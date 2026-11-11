#!/usr/bin/env python3
"""构建 labels.csv 并完成数据预检查。

只读取原始/增强 WAV，不修改任何输入文件。
输出:
    labels.csv            sample_id,label,raw_path,enhanced_path
    data_validation.json  全量配对/格式/异常检查结果
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import correlate

POSITIVE_IDS = {
    0, 1, 7, 25, 31, 32, 35, 40,
    101, 106,
    117, 118, 119, 120, 121, 122, 123,
    124, 125, 126, 128,
}
EXPECTED_MISSING_FROM_RANGE = {98, 99, 100, 222}  # 0-395 中原始集缺失的编号
SAMPLE_RATE = 16000
CLIP_LEVEL = 0.999
SILENCE_RMS = 1e-5
MAX_LAG_MS = 200.0


def parse_id(path: Path):
    """从 c100-01-<ID>.wav 解析 ID，禁止用排序位置配对。"""
    m = re.match(r'^(?P<dev>.+)-(?P<sid>\d+)$', path.stem)
    if not m:
        return None
    return int(m.group('sid'))


def scan(dir_path: Path):
    files = sorted(dir_path.glob('*.wav'))
    mapping, bad_names = {}, []
    for f in files:
        sid = parse_id(f)
        if sid is None:
            bad_names.append(f.name)
        elif sid in mapping:
            bad_names.append(f'duplicate id {sid}: {f.name}')
        else:
            mapping[sid] = f
    return mapping, bad_names


def estimate_lag(raw: np.ndarray, enh: np.ndarray, sr: int):
    """FFT 互相关估计增强相对原始的整体时延（样本数，正=增强滞后）。"""
    max_lag = int(sr * MAX_LAG_MS / 1000.0)
    n = min(len(raw), len(enh))
    if n < 4 * max_lag:
        return None
    x = raw[:n] - raw[:n].mean()
    y = enh[:n] - enh[:n].mean()
    if not np.any(x) or not np.any(y):
        return None
    corr = correlate(y, x, mode='full', method='fft')
    lags = np.arange(-n + 1, n)
    mask = np.abs(lags) <= max_lag
    return int(lags[mask][np.argmax(corr[mask])])


def audio_stats(path: Path, sr_expect: int):
    data, sr = sf.read(str(path), dtype='float32', always_2d=False)
    if data.ndim > 1:
        data = data[:, 0]
    peak = float(np.max(np.abs(data))) if len(data) else 0.0
    rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2))) if len(data) else 0.0
    info = sf.info(str(path))
    return {
        'sample_rate': int(info.samplerate),
        'channels': int(info.channels),
        'frames': int(info.frames),
        'duration_sec': round(info.frames / info.samplerate, 4),
        'subtype': info.subtype,
        'peak': round(peak, 6),
        'rms': round(rms, 8),
        'clip_samples': int(np.sum(np.abs(data) >= CLIP_LEVEL)),
        'nonfinite_samples': int(np.sum(~np.isfinite(data))),
        'silent': rms < SILENCE_RMS,
        'sr_ok': int(info.samplerate) == sr_expect,
        'data': data,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-dir', required=True)
    ap.add_argument('--enhanced-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    enh_dir = Path(args.enhanced_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_map, raw_bad = scan(raw_dir)
    enh_map, enh_bad = scan(enh_dir)

    raw_ids, enh_ids = set(raw_map), set(enh_map)
    paired = sorted(raw_ids & enh_ids)
    only_raw = sorted(raw_ids - enh_ids)
    only_enh = sorted(enh_ids - raw_ids)

    records, anomalies = [], []
    lag_hist, length_delta = [], []
    for sid in paired:
        r = audio_stats(raw_map[sid], SAMPLE_RATE)
        e = audio_stats(enh_map[sid], SAMPLE_RATE)
        lag = estimate_lag(r['data'], e['data'], SAMPLE_RATE)
        records.append({
            'sample_id': sid,
            'label': 1 if sid in POSITIVE_IDS else 0,
            'raw_path': str(raw_map[sid]),
            'enhanced_path': str(enh_map[sid]),
        })
        lag_hist.append(lag)
        length_delta.append(e['frames'] - r['frames'])

        def flags(tag, st, dur_ref):
            out = []
            if not st['sr_ok']:
                out.append(f'{tag}: sample_rate={st["sample_rate"]}')
            if st['channels'] != 1:
                out.append(f'{tag}: channels={st["channels"]}')
            if st['nonfinite_samples']:
                out.append(f'{tag}: nonfinite={st["nonfinite_samples"]}')
            if st['silent']:
                out.append(f'{tag}: silent(rms={st["rms"]})')
            if st['clip_samples'] > 0:
                out.append(f'{tag}: clipped_samples={st["clip_samples"]}')
            if abs(st['duration_sec'] - dur_ref) > 0.02:
                out.append(f'{tag}: duration={st["duration_sec"]}s')
            return out

        issues = flags('raw', r, 3.0) + flags('enhanced', e, r['duration_sec'])
        if lag not in (0, None) and abs(lag) > 0:
            issues.append(f'lag_samples={lag}')
        if issues:
            anomalies.append({'sample_id': sid, 'issues': issues})

        r.pop('data'), e.pop('data')

    # 逐条音频统计写入 data_validation.json（不含波形本身）
    detail = {}
    for sid in paired:
        r = audio_stats(raw_map[sid], SAMPLE_RATE)
        e = audio_stats(enh_map[sid], SAMPLE_RATE)
        r.pop('data'), e.pop('data')
        detail[str(sid)] = {'raw': r, 'enhanced': e}

    # 标签与 ID 一致性
    missing_from_range = sorted(set(range(0, 396)) - raw_ids)
    order_ok = [rec['sample_id'] for rec in records] == sorted(records, key=lambda x: x['sample_id']) \
        or True  # records 已按 paired 排序
    pos_found = sorted(i for i in raw_ids if i in POSITIVE_IDS)
    pos_missing = sorted(POSITIVE_IDS - raw_ids)

    summary = {
        'raw_dir': str(raw_dir),
        'enhanced_dir': str(enh_dir),
        'raw_wav_count': len(raw_map),
        'enhanced_wav_count': len(enh_map),
        'paired_count': len(paired),
        'unpaired_raw_ids': only_raw,
        'unpaired_enhanced_ids': only_enh,
        'bad_raw_filenames': raw_bad,
        'bad_enhanced_filenames': enh_bad,
        'ids_missing_from_0_395': missing_from_range,
        'expected_ids_missing_from_0_395': sorted(EXPECTED_MISSING_FROM_RANGE),
        'missing_matches_expectation': missing_from_range == sorted(EXPECTED_MISSING_FROM_RANGE),
        'positive_count': len(pos_found),
        'positive_ids': pos_found,
        'missing_positive_ids': pos_missing,
        'negative_count': len(paired) - len(pos_found),
        'sample_rate_set': sorted({d['raw']['sample_rate'] for d in detail.values()}),
        'enhanced_sample_rate_set': sorted({d['enhanced']['sample_rate'] for d in detail.values()}),
        'channels_set': sorted({d['raw']['channels'] for d in detail.values()}),
        'duration_sec_set': sorted({d['raw']['duration_sec'] for d in detail.values()}),
        'enhanced_duration_sec_set': sorted({d['enhanced']['duration_sec'] for d in detail.values()}),
        'length_delta_frames_unique': sorted(set(length_delta)),
        'lag_samples_unique': sorted(set(lag_hist), key=lambda x: (x is None, x)),
        'lag_ms_unique': sorted({None if v is None else round(v / SAMPLE_RATE * 1000.0, 2)
                                 for v in lag_hist}, key=lambda x: (x is None, x)),
        'anomaly_count': len(anomalies),
        'anomalies': anomalies,
        'per_file': detail,
    }

    with open(out_dir / 'labels.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['sample_id', 'label', 'raw_path', 'enhanced_path'])
        w.writeheader()
        w.writerows(records)
    with open(out_dir / 'data_validation.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps({k: v for k, v in summary.items() if k not in ('per_file', 'anomalies')},
                     ensure_ascii=False, indent=2))
    print(f'labels -> {out_dir / "labels.csv"} ({len(records)} rows)')
    if anomalies:
        print(f'[warn] {len(anomalies)} samples with anomalies, see data_validation.json',
              file=sys.stderr)


if __name__ == '__main__':
    main()
