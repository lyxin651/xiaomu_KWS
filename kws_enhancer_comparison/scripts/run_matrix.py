#!/usr/bin/env python3
"""双 KWS 模型 × 双数据集 × 多增强方法 的批量离线流式评测。

数据与模型目录约定（均相对本仓库根目录，即 gongniu/）：

    data/raw/<dataset>/c100-01-<sid>.wav            原始音频（kws_dataset）
    data/raw/<dataset>/c100-01-<sid>_raw.wav        原始音频（kws_dataset_1016_5）
    data/enhanced/<method>/<dataset>/<同名文件>      增强音频
    deploy/deploy_sort, deploy/deploy_ctc           部署推理代码与权重
    results/<dataset>/<model>__<method>.csv         逐条结果
    results/frames/<dataset>__<model>__<method>.npz 逐帧分数

推理实现完全复用 scripts/evaluate_kws.py（已验证：30 ms chunk、逐样本复位、
SORT 阈值 0.84 连续 3 帧、CTC 阈值 -0.10、500 ms 冷却），本脚本只负责组织
「数据集 × 方法 × 模型」的循环、落盘与断点续跑。

用法:
    python3 scripts/run_matrix.py                     # 全量，自动续跑
    python3 scripts/run_matrix.py --dry-run           # 只校验文件、打印任务矩阵
    python3 scripts/run_matrix.py --datasets kws_dataset --methods raw FastEnhancer_T
    python3 scripts/run_matrix.py --models ctc --limit 5 --no-resume   # 冒烟测试
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / 'config.json'
RESULTS_DIR = ROOT / 'results'
FRAMES_DIR = RESULTS_DIR / 'frames'

CSV_FIELDS = [
    'sample_id', 'label', 'model_name', 'model_file', 'audio_condition',
    'threshold', 'triggered', 'num_triggers', 'first_trigger_time_ms',
    'trigger_scores', 'audio_duration_sec', 'inference_time_ms', 'status',
    'num_frames', 'frame_score_max', 'frame_score_p95', 'score_kind',
    'trigger_times_ms', 'error',
]


def load_config():
    with open(CFG_PATH, encoding='utf-8') as f:
        return json.load(f)


def get_dataset(cfg, name):
    for ds in cfg['datasets']:
        if ds['name'] == name:
            return ds
    raise SystemExit(f'config.json 中没有数据集 {name}')


def audio_path(cfg, ds, method, filename):
    if method == 'raw':
        return ROOT / ds['raw_dir'] / filename
    return ROOT / cfg['enhanced_root'] / method / ds['name'] / filename


def scan_dataset(ds):
    """扫描原始音频，按文件名解析 sample_id 与标签。"""
    raw_dir = ROOT / ds['raw_dir']
    if not raw_dir.is_dir():
        raise SystemExit(f'缺少原始音频目录: {raw_dir}')
    pat = re.compile(ds['filename_pattern'])
    pos = set(ds.get('positive_ids') or [])
    items = []
    for p in sorted(raw_dir.glob('*.wav')):
        m = pat.match(p.name)
        if not m:
            continue
        sid = int(m.group(1))
        label = 1 if ds['positives'] == 'all' or sid in pos else 0
        items.append({'sample_id': sid, 'filename': p.name, 'label': label})
    items.sort(key=lambda r: r['sample_id'])
    return items


def load_done(csv_path):
    if not csv_path.exists():
        return set()
    done = set()
    with open(csv_path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r.get('status') == 'ok' and r.get('sample_id', '').isdigit():
                done.add(int(r['sample_id']))
    return done


def ensure_header(csv_path):
    if csv_path.exists():
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def append_row(csv_path, row):
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(
            {k: row.get(k, '') for k in CSV_FIELDS})


def print_plan(cfg, datasets, methods, models):
    total = 0
    print(f'{"dataset":24s} {"method":16s} {"model":6s} {"可评样本":>8s} {"缺失":>5s}')
    for dname in datasets:
        ds = get_dataset(cfg, dname)
        items = scan_dataset(ds)
        for method in methods:
            missing = [it for it in items
                       if not audio_path(cfg, ds, method, it['filename']).exists()]
            n_ok = len(items) - len(missing)
            for model in models:
                total += n_ok
                print(f'{dname:24s} {method:16s} {model:6s} {n_ok:8d} {len(missing):5d}')
            if missing:
                print(f'  [警告] {dname}/{method} 缺 {len(missing)} 条，例如 '
                      f'{[it["filename"] for it in missing[:3]]}')
    print(f'\n合计推理次数: {total}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=None)
    ap.add_argument('--methods', nargs='+', default=None)
    ap.add_argument('--models', nargs='+', default=None)
    ap.add_argument('--limit', type=int, default=None, help='每组仅跑前 N 条（冒烟测试）')
    ap.add_argument('--no-resume', action='store_true', help='重跑并覆盖已有结果')
    ap.add_argument('--dry-run', action='store_true', help='只校验文件与打印计划')
    args = ap.parse_args()

    cfg = load_config()
    datasets = args.datasets or [d['name'] for d in cfg['datasets']]
    methods = args.methods or cfg['methods']
    models = args.models or cfg['models']
    for m in methods:
        if m not in cfg['methods']:
            raise SystemExit(f'未知方法 {m}，可选 {cfg["methods"]}')

    print_plan(cfg, datasets, methods, models)
    if args.dry_run:
        return

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import evaluate_kws as ek   # 延迟导入：dry-run 不需要 onnxruntime

    deploy_dir = ROOT / 'deploy'
    ev = ek.Evaluator(deploy_dir, models)
    RESULTS_DIR.mkdir(exist_ok=True)
    FRAMES_DIR.mkdir(exist_ok=True)

    for dname in datasets:
        ds = get_dataset(cfg, dname)
        items = scan_dataset(ds)
        if args.limit:
            items = items[:args.limit]
        for method in methods:
            for model in models:
                spec = ek.MODEL_SPECS[model]
                model_path = deploy_dir / spec['module_dir'] / spec['model']
                out_csv = RESULTS_DIR / dname / f'{model}__{method}.csv'
                npz_path = FRAMES_DIR / f'{dname}__{model}__{method}.npz'
                npz_path.parent.mkdir(parents=True, exist_ok=True)

                if args.no_resume and out_csv.exists():
                    out_csv.unlink()
                ensure_header(out_csv)
                done = set() if args.no_resume else load_done(out_csv)
                frames = {}
                if npz_path.exists() and not args.no_resume:
                    import numpy as np
                    with np.load(npz_path) as z:
                        frames = {k: z[k] for k in z.files}

                todo = [it for it in items if it['sample_id'] not in done]
                print(f'\n[run] {dname}/{model}__{method}: total={len(items)} '
                      f'todo={len(todo)} (已完成 {len(done)})', flush=True)
                t0 = time.time()
                errors = 0
                for i, it in enumerate(todo, 1):
                    wav = audio_path(cfg, ds, method, it['filename'])
                    row = {'sample_id': it['sample_id'], 'audio_condition': method}
                    if not wav.exists():
                        errors += 1
                        row.update({
                            'label': it['label'], 'model_name': model,
                            'model_file': model_path.name,
                            'threshold': spec['threshold'], 'status': 'error',
                            'score_kind': spec['score_kind'],
                            'error': f'FileNotFoundError: {wav}',
                        })
                        append_row(out_csv, row)
                        continue
                    try:
                        res, scores = ek.run_one(ev, model, model_path, wav, it['label'])
                        row.update(res)
                        frames[str(it['sample_id'])] = scores.astype('float32')
                    except Exception as e:   # 单条失败记录为 error，绝不计作“未触发”
                        errors += 1
                        row.update({
                            'label': it['label'], 'model_name': model,
                            'model_file': model_path.name,
                            'threshold': spec['threshold'], 'status': 'error',
                            'score_kind': spec['score_kind'],
                            'error': f'{type(e).__name__}: {e}',
                        })
                        print(f'  [error] id={it["sample_id"]}: {e}', file=sys.stderr)
                    append_row(out_csv, row)
                    if i % 100 == 0:
                        flush_frames(npz_path, frames)
                        print(f'  ... {i}/{len(todo)}  {time.time() - t0:.0f}s', flush=True)

                flush_frames(npz_path, frames)
                ok = len(load_done(out_csv))
                print(f'[done] {dname}/{model}__{method}: ok={ok} 新增错误={errors} '
                      f'用时={time.time() - t0:.0f}s', flush=True)

    with open(RESULTS_DIR / 'runs_meta.json', 'w', encoding='utf-8') as f:
        json.dump({'config': str(CFG_PATH.relative_to(ROOT)),
                   'datasets': datasets, 'methods': methods, 'models': models,
                   'chunk_ms': cfg['chunk_ms'], 'sample_rate': cfg['sample_rate']},
                  f, ensure_ascii=False, indent=2)


def flush_frames(npz_path, frames):
    import numpy as np
    with open(npz_path, 'wb') as f:
        np.savez_compressed(f, **frames)


if __name__ == '__main__':
    main()
