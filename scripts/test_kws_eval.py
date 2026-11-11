#!/usr/bin/env python3
"""推理一致性与状态复位验证（不参与统计，只做正确性核查）。

覆盖:
  T1 标签/ID 校验
  T2 状态完全复位：同一样本连续/交错重复推理，逐帧分数逐位一致
  T3 流式等价性：10/30/100 ms chunk 输入，逐帧分数与触发事件一致
  T4 与官方 `mic_ctc.py --wav` 输出一致性（触发次数/时间/分数）
  T5 共享 session 与小样本四组推理核查（2 正 + 2 负）
"""
import argparse
import csv
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_kws as ek  # noqa: E402


def load_eval_module(name, deploy_dir, models):
    return ek.Evaluator(deploy_dir, models)


def run_with_chunks(ev, name, model_path, wav_path, chunk_samples):
    det, fbank, rec = ev.build_detector(name, model_path)
    wav, sr = sf.read(str(wav_path), dtype='float32', always_2d=False)
    events = []
    for pos in range(0, len(wav), chunk_samples):
        frames = fbank.push(wav[pos:pos + chunk_samples])
        if frames:
            events.extend(det.push_frames(frames))
    scores = (np.concatenate([o[0, :, 0] for o in rec.outputs]).astype(np.float32)
              if rec.outputs else np.zeros(0, dtype=np.float32))
    return scores, events


def t1_labels(output_dir: Path):
    with open(output_dir / 'labels.csv', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    ids = sorted(int(r['sample_id']) for r in rows)
    pos = sorted(int(r['sample_id']) for r in rows if int(r['label']) == 1)
    assert len(rows) == 392, len(rows)
    assert len(set(ids)) == 392
    assert len(pos) == 21, len(pos)
    assert set(ids) & {98, 99, 100, 222} == set()
    print(f'[T1] PASS labels=392 positives=21 missing=98,99,100,222')


def t2_reset(ev, labels, deploy_dir, ids=(0, 1, 2)):
    for name in ('sort', 'ctc'):
        mp = deploy_dir / ek.MODEL_SPECS[name]['module_dir'] / ek.MODEL_SPECS[name]['model']
        wav = labels[ids[0]]['raw_path']
        s1, e1 = run_with_chunks(ev, name, mp, wav, ek.CHUNK_SAMPLES)
        # 交错跑另一条样本，再回来跑同一条
        for other in ids[1:]:
            run_with_chunks(ev, name, mp, labels[other]['raw_path'], ek.CHUNK_SAMPLES)
        s2, e2 = run_with_chunks(ev, name, mp, wav, ek.CHUNK_SAMPLES)
        assert s1.shape == s2.shape and np.array_equal(s1, s2), f'{name} 分数未复位'
        assert e1 == e2, f'{name} 事件未复位'
        print(f'[T2] PASS {name} 重复推理逐位一致 (frames={len(s1)}, events={len(e1)})')


def t3_streaming(ev, labels, deploy_dir, ids=(0, 1, 2)):
    """SORT 对 chunk 长度不变；CTC 的 float 模型不变，INT8 模型随 chunk 变化。

    部署端固定 30 ms(3 帧)/chunk，因此主实验统一用 30 ms。INT8 的 chunk
    敏感性在此量化记录，用于报告中的适用性说明。
    """
    for sid in ids:
        mp = deploy_dir / 'deploy_sort' / ek.MODEL_SPECS['sort']['model']
        base_s, base_e = run_with_chunks(ev, 'sort', mp, labels[sid]['raw_path'], ek.CHUNK_SAMPLES)
        for cs in (160, 1600):
            s, e = run_with_chunks(ev, 'sort', mp, labels[sid]['raw_path'], cs)
            assert s.shape == base_s.shape, f'sort/{sid} chunk={cs} 帧数不同'
            assert np.allclose(s, base_s, atol=1e-5), f'sort/{sid} chunk={cs} 分数不一致'
            assert len(e) == len(base_e), f'sort/{sid} chunk={cs} 触发次数不同'
        print(f'[T3] PASS sort id={sid} chunk 10/30/100ms 一致')

    for model_file in ('kws_ctc_v2verifier_e7.onnx', 'kws_ctc_v2verifier_e7_int8.onnx'):
        mp = deploy_dir / 'deploy_ctc' / model_file
        for sid in ids[:1]:
            base_s, base_e = run_with_chunks(ev, 'ctc', mp, labels[sid]['raw_path'],
                                             ek.CHUNK_SAMPLES)
            diffs, wakes = [], []
            for cs in (160, 1600):
                s, e = run_with_chunks(ev, 'ctc', mp, labels[sid]['raw_path'], cs)
                diffs.append(float(np.max(np.abs(s - base_s))))
                wakes.append(len(e))
            print(f'[T3] ctc {model_file} id={sid} 分数 max|Δ| vs 30ms: '
                  f'{["%.4f" % d for d in diffs]}  触发次数 {len(base_e)} vs {wakes}')
            if 'int8' not in model_file:
                assert all(d < 1e-5 for d in diffs), 'float CTC 应 chunk 不变'
                assert all(w == len(base_e) for w in wakes)
                print(f'[T3] PASS ctc float id={sid} chunk 一致')
            else:
                print('[T3] NOTE ctc int8 输出随 chunk 长度变化（动态量化激活尺度），'
                      '部署与主实验均为 30ms，故统一固定 30ms')


def t4_official_ctc(labels, deploy_dir, ids=(0, 2, 101)):
    ctc_dir = deploy_dir / 'deploy_ctc'
    script = ctc_dir / 'mic_ctc.py'
    ev = ek.Evaluator(deploy_dir, ['ctc'])
    mp = ctc_dir / ek.MODEL_SPECS['ctc']['model']
    for sid in ids:
        wav = labels[sid]['raw_path']
        out = subprocess.run([sys.executable, str(script), '--wav', wav],
                             cwd=str(ctc_dir), capture_output=True, text=True)
        txt = out.stdout
        m = re.search(r'done: (\d+) frames, (\d+) wakes', txt)
        assert m, f'官方脚本输出异常: {txt}\n{out.stderr}'
        n_frames, n_wakes = int(m.group(1)), int(m.group(2))
        wake_lines = re.findall(r'WAKE  score=([+-][\d.]+)', txt)
        ws, events = run_with_chunks(ev, 'ctc', mp, wav, ek.CHUNK_SAMPLES)
        assert n_frames == len(ws), f'id={sid} 官方 frames={n_frames} vs 本实现 {len(ws)}'
        assert n_wakes == len(events), f'id={sid} 官方 wakes={n_wakes} vs {len(events)}'
        for a, b in zip(wake_lines, events):
            assert abs(float(a) - b['score']) < 1e-3, f'id={sid} score {a} vs {b["score"]}'
        print(f'[T4] PASS ctc id={sid} frames={n_frames} wakes={n_wakes} '
              f'scores={wake_lines} 与官方 --wav 一致')


def t5_small_matrix(ev, labels, deploy_dir, ids=(0, 1, 2, 3)):
    ok = True
    for name in ('sort', 'ctc'):
        mp = deploy_dir / ek.MODEL_SPECS[name]['module_dir'] / ek.MODEL_SPECS[name]['model']
        for cond in ('raw', 'enhanced'):
            for sid in ids:
                row, scores = ek.run_one(ev, name, mp, labels[sid][f'{cond}_path'], labels[sid]['label'])
                tag = 'OK' if row['status'] == 'ok' else 'FAIL'
                ok &= row['status'] == 'ok'
                print(f'[T5] {tag} {name:4s} {cond:8s} id={sid:3d} label={row["label"]} '
                      f'trig={row["triggered"]} n={row["num_triggers"]} '
                      f'first={row["first_trigger_time_ms"]}ms '
                      f'max={row["frame_score_max"]} frames={row["num_frames"]}')
    assert ok, '存在失败样本'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-dir', required=True)
    ap.add_argument('--enhanced-dir', required=True)
    ap.add_argument('--deploy-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()

    out = Path(args.output_dir).resolve()
    deploy = Path(args.deploy_dir).resolve()
    with open(out / 'labels.csv', newline='', encoding='utf-8') as f:
        labels = {int(r['sample_id']): r for r in csv.DictReader(f)}

    t1_labels(out)
    ev = ek.Evaluator(deploy, ['sort', 'ctc'])
    t2_reset(ev, labels, deploy)
    t3_streaming(ev, labels, deploy)
    t4_official_ctc(labels, deploy)
    t5_small_matrix(ev, labels, deploy)
    print('ALL TESTS PASSED')


if __name__ == '__main__':
    main()
