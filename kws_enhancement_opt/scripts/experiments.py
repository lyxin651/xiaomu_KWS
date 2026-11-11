#!/usr/bin/env python3
"""各实验任务清单构造（供 dispatch.py 分片执行）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

MODEL_PATHS = {
    'sort': common.SORT_MODEL,
    'ctc': common.CTC_MODEL_INT8,
    'ctc_fp32': common.CTC_MODEL_FP32,
}


def _base(r):
    return {'sid': r['sample_id'], 'label': r['label'],
            'raw_path': r['raw_path'], 'enh_path': r['enhanced_path']}


def build_jobs(exp, labels):
    jobs = []

    def add(model, model_path, cand, chunk_samples, variant=''):
        for r in labels:
            jobs.append({**_base(r), 'cand': cand, 'model': model,
                         'model_path': str(model_path),
                         'chunk_samples': chunk_samples, 'variant': variant})

    if exp == 'sweep':
        for model in ('sort', 'ctc'):
            for cand in ('raw', 'enh'):
                add(model, MODEL_PATHS[model], cand, common.CHUNK_SAMPLES)
    elif exp == 'candidates':
        for model in ('sort', 'ctc'):
            for cand in common.CANDIDATE_ORDER:
                add(model, MODEL_PATHS[model], cand, common.CHUNK_SAMPLES)
    elif exp == 'chunk':
        for variant, mpath in (('int8', MODEL_PATHS['ctc']),
                               ('fp32', MODEL_PATHS['ctc_fp32'])):
            for ms in (10, 30, 100):
                for cand in ('enh', 'raw'):
                    add('ctc', mpath, cand, common.SR * ms // 1000, variant)
    else:
        raise ValueError(exp)
    return jobs


EXP_DIR = {
    'sweep': 'sweep', 'candidates': 'candidates', 'chunk': 'chunk',
}
