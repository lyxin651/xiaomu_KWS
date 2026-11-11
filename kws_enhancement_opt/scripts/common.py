#!/usr/bin/env python3
"""GTCRN 增强优化实验：公共路径、候选音频构造、流式推理封装、指标与阈值重放。

设计约束
--------
* 复用 kws_enhancement_eval/scripts/evaluate_kws.py 里的 Evaluator /
  RecordingSession / replay_ctc_path_scores，不重写唤醒判决算法。
* 推理判决语义完全沿用部署包：SORT 阈值 0.84 + 连续 3 帧 + 500ms 冷却；
  CTC 阈值 -0.10（关键词路径 mean per-token log-LR）+ 500ms 冷却。
* 所有候选音频在内存中生成，不落盘，避免 16-bit 重量化引入额外变量。
* 只读访问原始 WAV 与已有实验结果，不写入 kws_enhancement_eval。
"""
import csv
import importlib.util
import sys
from collections import deque
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path('/Users/lei/Documents/公牛沐光KWS')
OPT_DIR = ROOT / 'kws_enhancement_opt'
EVAL_DIR = ROOT / 'kws_enhancement_eval'
BASE_RESULTS = EVAL_DIR / 'results'
DEPLOY_DIR = ROOT / 'deploy'
RAW_DIR = ROOT / 'kws_dataset'
ENH_DIR = ROOT / 'gtcrn_enhanced'
LABELS_CSV = EVAL_DIR / 'labels.csv'

RESULTS_DIR = OPT_DIR / 'results'
FIG_DIR = OPT_DIR / 'figures'
REPORT_DIR = OPT_DIR / 'reports'

SR = 16000
CHUNK_MS = 30
CHUNK_SAMPLES = SR * CHUNK_MS // 1000
FRAME_SHIFT_MS = 10

SORT_MODEL = DEPLOY_DIR / 'deploy_sort' / 'kws_cmvn_g1_e100.onnx'
CTC_MODEL_INT8 = DEPLOY_DIR / 'deploy_ctc' / 'kws_ctc_v2verifier_e7_int8.onnx'
CTC_MODEL_FP32 = DEPLOY_DIR / 'deploy_ctc' / 'kws_ctc_v2verifier_e7.onnx'
SORT_THR = 0.84
CTC_THR = -0.10

# ---------------- 候选方案（顺序即报告顺序）----------------
CANDIDATE_ORDER = [
    'enh', 'enh_gain_p1', 'enh_gain_p2', 'enh_gain_p3', 'enh_gain_p4',
    'mix_a090', 'mix_a075', 'mix_a050',
    'raw_rms_to_enh', 'enh_rms_matched', 'raw',
]
CANDIDATE_SPECS = {
    'enh': {'kind': 'identity', 'desc': 'GTCRN 增强音频（现基线, α=1.0）'},
    'enh_gain_p1': {'kind': 'gain', 'db': 1.0, 'desc': '增强音频 +1 dB'},
    'enh_gain_p2': {'kind': 'gain', 'db': 2.0, 'desc': '增强音频 +2 dB'},
    'enh_gain_p3': {'kind': 'gain', 'db': 3.0, 'desc': '增强音频 +3 dB'},
    'enh_gain_p4': {'kind': 'gain', 'db': 4.0, 'desc': '增强音频 +4 dB'},
    'mix_a090': {'kind': 'mix', 'alpha': 0.90, 'desc': '0.90·enh + 0.10·raw'},
    'mix_a075': {'kind': 'mix', 'alpha': 0.75, 'desc': '0.75·enh + 0.25·raw'},
    'mix_a050': {'kind': 'mix', 'alpha': 0.50, 'desc': '0.50·enh + 0.50·raw'},
    'raw_rms_to_enh': {'kind': 'raw_scale_to_enh',
                       'desc': '原始音频按增强后 RMS 等比例缩放（诊断对照）'},
    'enh_rms_matched': {'kind': 'enh_scale_to_raw',
                        'desc': '增强音频逐条匹配原始 RMS（仅归因）'},
    'raw': {'kind': 'raw', 'desc': '原始音频'},
}


def eval_module():
    """按文件路径导入现有 evaluate_kws 模块（只读复用）。"""
    if 'evalkws' in sys.modules:
        return sys.modules['evalkws']
    path = EVAL_DIR / 'scripts' / 'evaluate_kws.py'
    spec = importlib.util.spec_from_file_location('evalkws', str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['evalkws'] = mod
    spec.loader.exec_module(mod)
    return mod


def ctc_module():
    """独立加载部署侧 mic_ctc.py（取 KeywordPathMatcher，不建会话）。"""
    if 'mic_ctc_local' in sys.modules:
        return sys.modules['mic_ctc_local']
    path = DEPLOY_DIR / 'deploy_ctc' / 'mic_ctc.py'
    spec = importlib.util.spec_from_file_location('mic_ctc_local', str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['mic_ctc_local'] = mod
    spec.loader.exec_module(mod)
    return mod


def load_labels():
    with open(LABELS_CSV, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['sample_id'] = int(r['sample_id'])
        r['label'] = int(r['label'])
    rows.sort(key=lambda r: r['sample_id'])
    return rows


def load_wav(path, expect_sr=SR):
    wav, sr = sf.read(str(path), dtype='float32', always_2d=False)
    if wav.ndim > 1:
        wav = wav[:, 0]
    if sr != expect_sr:
        raise ValueError(f'{path}: sample_rate={sr} != {expect_sr}')
    return np.ascontiguousarray(wav, dtype=np.float32)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-20))


def to_db(x):
    return 20.0 * np.log10(max(x, 1e-12))


def build_candidate_audio(cand, raw, enh):
    """返回 (audio, info)。不做静默归一化：所有缩放/混合系数写入 info。"""
    spec = CANDIDATE_SPECS[cand]
    kind = spec['kind']
    if len(raw) != len(enh):
        raise ValueError(f'length mismatch raw={len(raw)} enh={len(enh)}')
    scale = 1.0
    if kind == 'raw':
        out = raw.copy()
    elif kind == 'identity':
        out = enh.copy()
    elif kind == 'gain':
        scale = 10.0 ** (spec['db'] / 20.0)
        out = enh * scale
    elif kind == 'mix':
        a = spec['alpha']
        out = a * enh + (1.0 - a) * raw
    elif kind == 'raw_scale_to_enh':
        scale = rms(enh) / rms(raw)
        out = raw * scale
    elif kind == 'enh_scale_to_raw':
        scale = rms(raw) / rms(enh)
        out = enh * scale
    else:
        raise ValueError(kind)

    info = {
        'rms_raw': rms(raw), 'rms_enh': rms(enh), 'rms_out': rms(out),
        'rms_in': rms(raw if kind in ('raw', 'raw_scale_to_enh') else enh),
        'peak_out': float(np.max(np.abs(out))),
        'scale': scale,
        'gain_db': 20.0 * np.log10(scale) if scale > 0 else 0.0,
        'clip_samples': int(np.sum(np.abs(out) > 1.0)),
        'n_samples': int(len(out)),
        'len_raw': len(raw), 'len_enh': len(enh),
    }
    return out.astype(np.float32), info


def run_stream(ev, model_name, model_path, wav, chunk_samples=CHUNK_SAMPLES,
               keep_frames=False):
    """单条音频流式推理。返回事件、逐帧分数与（可选）逐帧模型输出。"""
    det, fbank, rec = ev.build_detector(model_name, model_path)
    events = []
    for pos in range(0, len(wav), chunk_samples):
        frames = fbank.push(wav[pos:pos + chunk_samples])
        if frames:
            events.extend(det.push_frames(frames))

    out = {
        'events': events,
        'num_frames': int(det.total_frames),
        'num_triggers': len(events),
        'triggered': int(len(events) > 0),
        'first_trigger_ms': round(events[0]['time_ms'], 1) if events else '',
        'trigger_times_ms': [round(e['time_ms'], 1) for e in events],
        'trigger_scores': [float(e['score']) for e in events],
    }
    if model_name == 'sort':
        frame = (np.concatenate([o[0, :, 0] for o in rec.outputs]).astype(np.float32)
                 if rec.outputs else np.zeros(0, np.float32))
        out['scores'] = frame
        out['max_score'] = float(frame.max()) if len(frame) else float('nan')
    else:
        frame = (np.concatenate([o[0] for o in rec.outputs]).astype(np.float32)
                 if rec.outputs else np.zeros((0, 6), np.float32))
        scores = eval_module().replay_ctc_path_scores(rec, ev.mods[model_name])
        out['scores'] = scores
        out['max_score'] = float(scores.max()) if len(scores) else float('nan')
        out['blank_logprob_mean'] = (float(frame[:, 0].mean())
                                     if len(frame) else float('nan'))
        if keep_frames:
            out['logps'] = frame
    return out


# ---------------- 阈值重放（与部署判决逐帧等价）----------------
def replay_sort(probs, threshold, smooth=3, min_frames=3, cooldown_frames=50):
    recent = deque(maxlen=smooth)
    cooldown = 0
    events = []
    for i, s in enumerate(probs):
        cooldown = max(0, cooldown - 1)
        recent.append(float(s))
        if cooldown > 0:
            continue
        if (len(recent) >= min_frames
                and all(x >= threshold for x in list(recent)[-min_frames:])):
            events.append({'time_ms': i * FRAME_SHIFT_MS,
                           'score': float(np.mean(list(recent)[-min_frames:]))})
            cooldown = cooldown_frames
    return events


def replay_ctc(logps, threshold, matcher_cls, cooldown_frames=50):
    m = matcher_cls()
    cooldown = 0
    events = []
    for i, lp in enumerate(logps):
        cooldown = max(0, cooldown - 1)
        m.push(lp)
        if m.score >= threshold and cooldown == 0:
            events.append({'time_ms': i * FRAME_SHIFT_MS, 'score': float(m.score)})
            cooldown = cooldown_frames
            m.reset()
    return events


def replay_ctc_maxscore(logps, matcher_cls):
    """不触发复位的单调 matcher 曲线（检测统计量，用于 AUC / 可分性）。"""
    m = matcher_cls()
    best = -np.inf
    for lp in logps:
        m.push(lp)
        if m.score > best:
            best = m.score
    return float(best)


# ---------------- 指标 ----------------
def metrics_from_flags(labels, triggered):
    tp = sum(1 for y, t in zip(labels, triggered) if y == 1 and t)
    fn = sum(1 for y, t in zip(labels, triggered) if y == 1 and not t)
    fp = sum(1 for y, t in zip(labels, triggered) if y == 0 and t)
    tn = sum(1 for y, t in zip(labels, triggered) if y == 0 and not t)
    npos, nneg = tp + fn, fp + tn
    return {
        'TP': tp, 'FN': fn, 'FP': fp, 'TN': tn,
        'n_pos': npos, 'n_neg': nneg,
        'recall': tp / npos if npos else float('nan'),
        'fpr': fp / nneg if nneg else float('nan'),
    }


def auc(pos_scores, neg_scores):
    """rank-based AUC（Mann-Whitney U）。"""
    from scipy.stats import rankdata
    pos = np.asarray(pos_scores, dtype=np.float64)
    neg = np.asarray(neg_scores, dtype=np.float64)
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    all_s = np.concatenate([pos, neg])
    r = rankdata(all_s)
    u = r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def read_rows(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def trigger_map(rows, condition=None):
    """CSV 行 -> {sample_id: triggered(int)}，可选按 audio_condition 过滤。"""
    out = {}
    for r in rows:
        if condition is not None and r.get('audio_condition') != condition:
            continue
        if r.get('status') != 'ok':
            continue
        out[int(r['sample_id'])] = int(r['triggered'])
    return out


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})


def write_json(path, obj):
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
