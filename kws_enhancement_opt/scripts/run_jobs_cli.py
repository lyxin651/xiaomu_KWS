#!/usr/bin/env python3
"""分片推理引擎：每条实验任务在独立解释器进程里顺序执行（避免 fork+ORT 退化）。

用法（由 dispatch.py 调度）:
    python run_jobs_cli.py --jobs jobs.json --csv part.csv --npz part.npz

每个 job 字典字段:
    sid, label, cand, model, model_path, chunk_samples, variant
每条音频使用新的 detector（cache/fbank/cooldown 全复位），30ms chunk 默认。
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

CSV_FIELDS = [
    'job_key', 'sid', 'label', 'cand', 'model', 'variant', 'chunk_ms',
    'triggered', 'num_triggers', 'first_trigger_ms', 'trigger_times_ms',
    'trigger_scores', 'max_score', 'blank_logprob_mean', 'num_frames',
    'inference_ms', 'audio_duration_sec', 'audio_rms_raw', 'audio_rms_enh',
    'audio_rms_out', 'audio_rms_in', 'audio_peak_out', 'audio_scale',
    'audio_gain_db', 'audio_clip_samples', 'audio_n_samples',
    'audio_len_raw', 'audio_len_enh',
]


def limit_onnx_threads(n):
    """限制每进程 ORT 线程数：多进程并发时默认线程池会疯狂自旋（实测 20x 变慢）。"""
    import onnxruntime as ort
    if getattr(ort.InferenceSession, '_limited', False):
        return
    orig = ort.InferenceSession

    def patched(path_or_bytes, sess_options=None, providers=None,
                provider_options=None, **kwargs):
        so = sess_options or ort.SessionOptions()
        so.intra_op_num_threads = n
        so.inter_op_num_threads = 1
        try:
            so.add_session_config_entry('session.intra_op.allow_spinning', '0')
        except Exception:
            pass
        return orig(path_or_bytes, so, providers, provider_options, **kwargs)

    patched._limited = True
    ort.InferenceSession = patched


def make_evaluator(models=('sort', 'ctc')):
    return common.eval_module().Evaluator(common.DEPLOY_DIR, list(models))


def job_key(job):
    return (f"{job['model']}|{job.get('variant','')}|{job['cand']}|"
            f"{job['sid']}|{job['chunk_samples'] * 1000 // common.SR}")


def run_job(ev, job):
    raw = common.load_wav(job['raw_path'])
    enh = common.load_wav(job['enh_path'])
    audio, info = common.build_candidate_audio(job['cand'], raw, enh)
    t0 = time.perf_counter()
    res = common.run_stream(ev, job['model'], job['model_path'], audio,
                            job['chunk_samples'], keep_frames=True)
    infer_ms = (time.perf_counter() - t0) * 1000.0
    row = {
        'job_key': job_key(job),
        'sid': job['sid'], 'label': job['label'], 'cand': job['cand'],
        'model': job['model'], 'variant': job.get('variant', ''),
        'chunk_ms': job['chunk_samples'] * 1000 // common.SR,
        'triggered': res['triggered'], 'num_triggers': res['num_triggers'],
        'first_trigger_ms': res['first_trigger_ms'],
        'trigger_times_ms': '|'.join(f'{t:.1f}' for t in res['trigger_times_ms']),
        'trigger_scores': '|'.join(f'{s:.6f}' for s in res['trigger_scores']),
        'max_score': round(res['max_score'], 6),
        'blank_logprob_mean': (round(res.get('blank_logprob_mean', float('nan')), 6)
                               if job['model'] == 'ctc' else ''),
        'num_frames': res['num_frames'],
        'inference_ms': round(infer_ms, 1),
        'audio_duration_sec': round(len(audio) / common.SR, 4),
    }
    for k, v in info.items():
        row[f'audio_{k}'] = v
    arr = res['scores'] if job['model'] == 'sort' else res['logps']
    return row, job_key(job), np.asarray(arr, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', required=True)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--npz', default=None)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--threads', type=int, default=1)
    args = ap.parse_args()

    if args.threads > 0:
        limit_onnx_threads(args.threads)

    with open(args.jobs, encoding='utf-8') as f:
        jobs = json.load(f)
    if args.limit:
        jobs = jobs[:args.limit]

    models = sorted({j['model'] for j in jobs})
    ev = make_evaluator(models)
    out_csv = Path(args.csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_csv.exists():
        with open(out_csv, newline='', encoding='utf-8') as f:
            done = {r['job_key'] for r in csv.DictReader(f) if r.get('job_key')}
    exists = out_csv.exists()
    frames = {}
    npz_path = Path(args.npz) if args.npz else None
    if npz_path and npz_path.exists():
        with np.load(npz_path) as z:
            frames = {k: z[k] for k in z.files}

    rows_written = 0
    t0 = time.time()
    with open(out_csv, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            w.writeheader()
        for i, job in enumerate(jobs, 1):
            if job_key(job) in done:
                continue
            row, key, arr = run_job(ev, job)
            w.writerow({k: row.get(k, '') for k in CSV_FIELDS})
            if npz_path is not None:
                frames[key] = arr
            rows_written += 1
            if i % 100 == 0:
                f.flush()
                if npz_path is not None:
                    np.savez_compressed(npz_path, **frames)
                print(f'  [{out_csv.name}] {i}/{len(jobs)} {time.time()-t0:.0f}s',
                      flush=True)
    if npz_path is not None:
        np.savez_compressed(npz_path, **frames)
    print(f'[done] {out_csv.name}: +{rows_written} rows in {time.time()-t0:.0f}s',
          flush=True)


if __name__ == '__main__':
    main()
