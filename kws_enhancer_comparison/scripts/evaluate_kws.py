#!/usr/bin/env python3
"""批量离线流式推理：SORT / CTC × raw / enhanced。

设计要点
--------
* 直接复用部署包里的 `OnnxKWS`(deploy_sort) 与 `CtcKwsSession`(deploy_ctc)、
  `StreamingFbank`，不重写唤醒判决算法。
* 每条音频开始前重新构造检测器（cache 从 _init_cache.npy 重新载入、
  fbank / matcher / cooldown / recent 全部新建），因此样本之间零状态残留；
  ONNX InferenceSession 通过注入方式共享，只加载一次。
* 输入按 30 ms（480 samples）chunk 喂入，与部署端一致。
* 逐帧分数通过包裹 session.run 记录，不改变任何判决行为。
* 结果逐条落盘，支持断点续跑（已存在且 status=ok 的 sample_id 自动跳过）。

用法:
    python scripts/evaluate_kws.py \
        --raw-dir ../kws_dataset --enhanced-dir ../gtcrn_enhanced \
        --deploy-dir ../deploy --output-dir .. [--models sort ctc] \
        [--conditions raw enhanced] [--limit N] [--ids 0,1,7] [--no-resume]
"""
import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

SAMPLE_RATE = 16000
CHUNK_MS = 30
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)

MODEL_SPECS = {
    'sort': {
        'module_dir': 'deploy_sort',
        'module_file': 'mic_kws.py',
        'class': 'OnnxKWS',
        'model': 'kws_cmvn_g1_e100.onnx',
        'threshold': 0.84,
        'kwargs': {'cooldown_ms': 500.0, 'min_trigger_frames': 3, 'score_smooth': 3},
        'score_kind': 'frame_prob',
    },
    'ctc': {
        'module_dir': 'deploy_ctc',
        'module_file': 'mic_ctc.py',
        'class': 'CtcKwsSession',
        'model': 'kws_ctc_v2verifier_e7_int8.onnx',
        'threshold': -0.10,
        'kwargs': {'cooldown_ms': 500.0},
        'score_kind': 'path_score_nats_per_token',
    },
}

CSV_FIELDS = [
    'sample_id', 'label', 'model_name', 'model_file', 'audio_condition',
    'threshold', 'triggered', 'num_triggers', 'first_trigger_time_ms',
    'trigger_scores', 'audio_duration_sec', 'inference_time_ms', 'status',
    'num_frames', 'frame_score_max', 'frame_score_p95', 'score_kind',
    'trigger_times_ms', 'error',
]


class RecordingSession:
    """代理共享 InferenceSession，记录每帧模型输出（只读旁路）。"""

    def __init__(self, sess):
        self.sess = sess
        self.outputs = []

    def run(self, output_names, input_feed, *a, **kw):
        res = self.sess.run(output_names, input_feed, *a, **kw)
        self.outputs.append(np.asarray(res[0]))
        return res

    def __getattr__(self, item):
        return getattr(self.sess, item)


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f'kwsmod_{path.stem}_{id(path)}', str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class Evaluator:
    def __init__(self, deploy_dir: Path, models):
        self.deploy_dir = deploy_dir
        self.mods, self.sessions, self.recorders = {}, {}, {}
        # 两个部署包的 streaming_fbank.py 内容一致，统一加载一次
        fbank_path = deploy_dir / 'deploy_sort' / 'streaming_fbank.py'
        spec = importlib.util.spec_from_file_location('shared_streaming_fbank', str(fbank_path))
        self.fbank_mod = importlib.util.module_from_spec(spec)
        sys.modules['streaming_fbank'] = self.fbank_mod   # 供部署模块 import 复用
        spec.loader.exec_module(self.fbank_mod)

        for name in models:
            spec = MODEL_SPECS[name]
            d = deploy_dir / spec['module_dir']
            mod = load_module(d / spec['module_file'])
            model_path = d / spec['model']
            sess = self._get_session(model_path)
            self.mods[name] = mod
            self.sessions[name] = sess
            print(f'[init] {name}: {model_path.name} inputs='
                  f'{[i.name for i in sess.get_inputs()]}')

    def _get_session(self, model_path: Path):
        key = str(model_path)
        if key not in self.sessions:
            self.sessions[key] = ort.InferenceSession(
                key, providers=['CPUExecutionProvider'])
        return self.sessions[key]

    def build_detector(self, name, model_path):
        """新建检测器实例（cache/fbank 状态全复位），注入共享 session。"""
        spec = MODEL_SPECS[name]
        rec = RecordingSession(self._get_session(model_path))
        orig = ort.InferenceSession
        ort.InferenceSession = lambda *a, **kw: rec
        try:
            with open(os.devnull, 'w') as devnull, redirect_stdout(devnull):
                if name == 'sort':
                    det = self.mods[name].OnnxKWS(
                        str(model_path), threshold=spec['threshold'], **spec['kwargs'])
                else:
                    det = self.mods[name].CtcKwsSession(
                        str(model_path), threshold=spec['threshold'], **spec['kwargs'])
        finally:
            ort.InferenceSession = orig
        fbank = self.fbank_mod.StreamingFbank(
            sample_rate=SAMPLE_RATE, frame_length_ms=25, frame_shift_ms=10, n_mels=80)
        return det, fbank, rec


def replay_ctc_path_scores(rec: RecordingSession, mod):
    """用独立 matcher 重放 CTC log-prob，得到逐帧关键词路径分数（分析用）。

    注意：该曲线不做触发后复位，仅用于可视化；触发判定一律以
    CtcKwsSession.push_frames 的事件为准。
    """
    if not rec.outputs:
        return np.zeros(0, dtype=np.float32)
    logps = np.concatenate([o[0] for o in rec.outputs], axis=0)
    m = mod.KeywordPathMatcher()
    scores = np.empty(len(logps), dtype=np.float32)
    for i, lp in enumerate(logps):
        m.push(lp)
        scores[i] = m.score
    return scores


def run_one(ev: Evaluator, name, model_path, wav_path, label):
    det, fbank, rec = ev.build_detector(name, model_path)
    wav, sr = sf.read(str(wav_path), dtype='float32', always_2d=False)
    if wav.ndim > 1:
        wav = wav[:, 0]
    if sr != SAMPLE_RATE:
        raise ValueError(f'sample_rate={sr} != {SAMPLE_RATE}, refuse silent resampling')

    events = []
    t0 = time.perf_counter()
    for pos in range(0, len(wav), CHUNK_SAMPLES):
        frames = fbank.push(wav[pos:pos + CHUNK_SAMPLES])
        if frames:
            events.extend(det.push_frames(frames))
    infer_ms = (time.perf_counter() - t0) * 1000.0

    if name == 'sort':
        scores = (np.concatenate([o[0, :, 0] for o in rec.outputs]).astype(np.float32)
                  if rec.outputs else np.zeros(0, dtype=np.float32))
    else:
        scores = replay_ctc_path_scores(rec, ev.mods[name])

    spec = MODEL_SPECS[name]
    row = {
        'label': label,
        'model_name': name,
        'model_file': model_path.name,
        'threshold': spec['threshold'],
        'triggered': int(len(events) > 0),
        'num_triggers': len(events),
        'first_trigger_time_ms': round(events[0]['time_ms'], 1) if events else '',
        'trigger_scores': '|'.join(f'{e["score"]:.6f}' for e in events),
        'audio_duration_sec': round(len(wav) / SAMPLE_RATE, 4),
        'inference_time_ms': round(infer_ms, 2),
        'status': 'ok',
        'num_frames': int(det.total_frames),
        'frame_score_max': round(float(scores.max()), 6) if len(scores) else '',
        'frame_score_p95': round(float(np.percentile(scores, 95)), 6) if len(scores) else '',
        'score_kind': spec['score_kind'],
        'trigger_times_ms': '|'.join(f'{e["time_ms"]:.1f}' for e in events),
        'error': '',
    }
    return row, scores


def load_rows(path: Path):
    if not path.exists():
        return {}, []
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    done = {int(r['sample_id']): r for r in rows if r.get('status') == 'ok'}
    return done, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-dir', required=True)
    ap.add_argument('--enhanced-dir', required=True)
    ap.add_argument('--deploy-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--models', nargs='+', default=['sort', 'ctc'], choices=list(MODEL_SPECS))
    ap.add_argument('--conditions', nargs='+', default=['raw', 'enhanced'],
                    choices=['raw', 'enhanced'])
    ap.add_argument('--limit', type=int, default=None, help='仅跑前 N 条（调试）')
    ap.add_argument('--ids', default=None, help='逗号分隔的 sample_id 列表（调试）')
    ap.add_argument('--no-resume', action='store_true')
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    res_dir = out_dir / 'results'
    fs_dir = res_dir / 'frame_scores'
    fs_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / 'labels.csv', newline='', encoding='utf-8') as f:
        labels = [dict(r) for r in csv.DictReader(f)]
    labels.sort(key=lambda r: int(r['sample_id']))
    for r in labels:
        r['sample_id'] = int(r['sample_id'])
        r['label'] = int(r['label'])
    if args.ids:
        keep = {int(x) for x in args.ids.split(',') if x.strip()}
        labels = [r for r in labels if r['sample_id'] in keep]
    if args.limit:
        labels = labels[:args.limit]

    ev = Evaluator(Path(args.deploy_dir).resolve(), args.models)
    summary = {}

    for name in args.models:
        model_path = (Path(args.deploy_dir).resolve()
                      / MODEL_SPECS[name]['module_dir'] / MODEL_SPECS[name]['model'])
        for cond in args.conditions:
            out_csv = res_dir / f'{name}_{cond}.csv'
            npz_path = fs_dir / f'{name}_{cond}.npz'
            done, old_rows = ({}, []) if args.no_resume else load_rows(out_csv)
            if args.no_resume and out_csv.exists():
                out_csv.unlink()
            frame_store = {}
            if npz_path.exists() and not args.no_resume:
                with np.load(npz_path) as z:
                    frame_store = {k: z[k] for k in z.files}
            if not out_csv.exists():
                with open(out_csv, 'w', newline='', encoding='utf-8') as f:
                    csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

            key = lambda r: r.get('raw_path') if cond == 'raw' else r.get('enhanced_path')
            todo = [r for r in labels if r['sample_id'] not in done]
            print(f'[run] {name}/{cond}: total={len(labels)} todo={len(todo)} '
                  f'(resume skip={len(done)})')
            t_start = time.time()
            errors = 0
            for i, r in enumerate(todo, 1):
                wav_path = key(r)
                row = {'sample_id': r['sample_id']}
                try:
                    res, scores = run_one(ev, name, model_path, wav_path, r['label'])
                    row.update(res)
                    frame_store[str(r['sample_id'])] = scores.astype(np.float32)
                except Exception as e:  # 单样本错误记录并继续，绝不静默判为未触发
                    errors += 1
                    row.update({
                        'label': r['label'], 'model_name': name,
                        'model_file': model_path.name, 'audio_condition': cond,
                        'threshold': MODEL_SPECS[name]['threshold'],
                        'triggered': '', 'num_triggers': '', 'first_trigger_time_ms': '',
                        'trigger_scores': '', 'audio_duration_sec': '',
                        'inference_time_ms': '', 'status': 'error',
                        'num_frames': '', 'frame_score_max': '', 'frame_score_p95': '',
                        'score_kind': MODEL_SPECS[name]['score_kind'],
                        'trigger_times_ms': '', 'error': f'{type(e).__name__}: {e}',
                    })
                    print(f'[error] {name}/{cond} id={r["sample_id"]}: {e}', file=sys.stderr)
                row['audio_condition'] = cond
                with open(out_csv, 'a', newline='', encoding='utf-8') as f:
                    w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                    w.writerow(row)
                if i % 50 == 0:
                    with open(npz_path, 'wb') as f:
                        np.savez_compressed(f, **frame_store)
                    print(f'  ... {i}/{len(todo)}  {(time.time()-t_start):.1f}s')

            with open(npz_path, 'wb') as f:
                np.savez_compressed(f, **frame_store)

            done2, rows2 = load_rows(out_csv)
            ok_rows = [r for r in rows2 if r['status'] == 'ok']
            summary[f'{name}_{cond}'] = {
                'model': name, 'condition': cond,
                'model_file': model_path.name,
                'threshold': MODEL_SPECS[name]['threshold'],
                'total_rows': len(rows2), 'ok': len(ok_rows),
                'errors': len(rows2) - len(ok_rows),
                'triggered': sum(int(r['triggered']) for r in ok_rows),
                'positive': sum(1 for r in ok_rows if int(r['label']) == 1),
                'positive_triggered': sum(1 for r in ok_rows
                                          if int(r['label']) == 1 and int(r['triggered']) == 1),
                'negative': sum(1 for r in ok_rows if int(r['label']) == 0),
                'negative_triggered': sum(1 for r in ok_rows
                                          if int(r['label']) == 0 and int(r['triggered']) == 1),
                'run_seconds': round(time.time() - t_start, 1),
                'chunk_ms': CHUNK_MS,
                'cooldown_ms': MODEL_SPECS[name]['kwargs']['cooldown_ms'],
            }
            print(f'[done] {name}/{cond}: ok={len(ok_rows)} errors={len(rows2)-len(ok_rows)} '
                  f'triggered={summary[f"{name}_{cond}"]["triggered"]}')

    with open(res_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump({'config': {'chunk_ms': CHUNK_MS, 'sample_rate': SAMPLE_RATE,
                              'positive_count': sum(1 for r in labels if r['label'] == 1),
                              'negative_count': sum(1 for r in labels if r['label'] == 0),
                              'model_specs': MODEL_SPECS},
                   'runs': summary}, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
