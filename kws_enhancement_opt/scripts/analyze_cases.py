#!/usr/bin/env python3
"""任务 C：关键漏检 / 恢复样本归因。

对象：CTC 新增漏检 7, 25, 122, 123, 125；SORT 新增漏检 121；
      增强后恢复唤醒 119 (SORT), 32 (CTC)。
对比原始/增强的 RMS、Fbank 频带能量、逐帧分数、触发时刻、CTC 六类 token 概率，
并结合任务 B 的候选方案（电平/增益/干湿混合）判断原因归属。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

CASES = [
    (7, 'ctc_lost'), (25, 'ctc_lost'), (122, 'ctc_lost'),
    (123, 'ctc_lost'), (125, 'ctc_lost'),
    (121, 'sort_lost'), (119, 'sort_recovered'), (32, 'ctc_recovered'),
]
VOCAB = ['blank', 'fill', 'x', 'iao', 'm', 'u']
KW_IDX = [2, 3, 4, 5]


def load_fbank():
    import importlib.util
    p = common.DEPLOY_DIR / 'deploy_sort' / 'streaming_fbank.py'
    spec = importlib.util.spec_from_file_location('sfb_plot', str(p))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.StreamingFbank(sample_rate=16000, frame_length_ms=25,
                            frame_shift_ms=10, n_mels=80)


def band_energy_db(fb):
    mel = np.exp(fb.astype(np.float64))          # fbank 是 log-mel
    bands = {'low_0_1k': slice(0, 20), 'mid_1_2k': slice(20, 40),
             'mid_2_4k': slice(40, 60), 'high_4_8k': slice(60, 80)}
    return {k: float(10 * np.log10(mel[:, v].sum() / mel.shape[0] + 1e-12))
            for k, v in bands.items()}


def main():
    labels = common.load_labels()
    sid_map = {r['sample_id']: r for r in labels}
    with np.load(common.RESULTS_DIR / 'sweep' / 'frames.npz') as z:
        frames = {k: z[k] for k in z.files}
    cand_rows = {}
    cand_csv = common.RESULTS_DIR / 'candidates' / 'all.csv'
    if cand_csv.exists():
        for r in common.read_rows(cand_csv):
            cand_rows.setdefault((r['cand'], r['model'], int(r['sid'])), r)
    matcher_cls = common.ctc_module().KeywordPathMatcher

    base = {}
    for model in ('sort', 'ctc'):
        base[model] = {int(r['sample_id']): r
                       for r in common.read_rows(
                           common.BASE_RESULTS / f'{model}_enhanced.csv')}
        base[f'{model}_raw'] = {int(r['sample_id']): r
                                for r in common.read_rows(
                                    common.BASE_RESULTS / f'{model}_raw.csv')}

    fb = load_fbank()
    out_rows, notes = [], {}
    for sid, kind in CASES:
        rec = sid_map[sid]
        raw = common.load_wav(rec['raw_path'])
        enh = common.load_wav(rec['enhanced_path'])
        sort_raw = frames[f'sort||raw|{sid}|30']
        sort_enh = frames[f'sort||enh|{sid}|30']
        ctc_raw = frames[f'ctc||raw|{sid}|30']
        ctc_enh = frames[f'ctc||enh|{sid}|30']

        def path_curve(logps):
            m = matcher_cls()
            out = np.empty(len(logps), dtype=np.float32)
            for i, lp in enumerate(logps):
                m.push(lp)
                out[i] = m.score
            return out

        pc_raw, pc_enh = path_curve(ctc_raw), path_curve(ctc_enh)
        fb_raw, fb_enh = fb.compute_offline(raw), fb.compute_offline(enh)
        be_raw, be_enh = band_energy_db(fb_raw), band_energy_db(fb_enh)

        def token_stats(logps):
            p = np.exp(np.clip(logps.astype(np.float64), -20, 0))
            return {
                'kw_mean_prob': float(p[:, KW_IDX].mean()),
                'kw_max_prob': float(p[:, KW_IDX].max()),
                'blank_mean_prob': float(p[:, 0].mean()),
                'per_token_mean_prob': {VOCAB[i]: float(p[:, i].mean()) for i in range(6)},
            }

        note = {
            'kind': kind, 'label': rec['label'],
            'rms_raw': common.rms(raw), 'rms_enh': common.rms(enh),
            'rms_ratio_db': 20 * np.log10(common.rms(enh) / common.rms(raw)),
            'peak_raw': float(np.max(np.abs(raw))), 'peak_enh': float(np.max(np.abs(enh))),
            'band_energy_db_raw': be_raw, 'band_energy_db_enh': be_enh,
            'band_delta_db': {k: round(be_enh[k] - be_raw[k], 2) for k in be_raw},
            'sort': {
                'max_raw': float(sort_raw.max()), 'max_enh': float(sort_enh.max()),
                'triggered_raw': base['sort_raw'][sid]['triggered'],
                'triggered_enh': base['sort'][sid]['triggered'],
                'trigger_ms_raw': base['sort_raw'][sid]['trigger_times_ms'],
                'trigger_ms_enh': base['sort'][sid]['trigger_times_ms'],
            },
            'ctc': {
                'max_path_raw': float(pc_raw.max()), 'max_path_enh': float(pc_enh.max()),
                'triggered_raw': base['ctc_raw'][sid]['triggered'],
                'triggered_enh': base['ctc'][sid]['triggered'],
                'trigger_ms_raw': base['ctc_raw'][sid]['trigger_times_ms'],
                'trigger_ms_enh': base['ctc'][sid]['trigger_times_ms'],
                'token_stats_raw': token_stats(ctc_raw),
                'token_stats_enh': token_stats(ctc_enh),
            },
            'candidate_scores': {},
        }
        for cand in common.CANDIDATE_ORDER:
            for model in ('sort', 'ctc'):
                r = cand_rows.get((cand, model, sid))
                if r:
                    note['candidate_scores'].setdefault(cand, {})[model] = {
                        'max_score': float(r['max_score']),
                        'triggered': int(r['triggered']),
                    }
        notes[str(sid)] = note
        out_rows.append({
            'sid': sid, 'kind': kind,
            'rms_raw': round(common.rms(raw), 6), 'rms_enh': round(common.rms(enh), 6),
            'rms_ratio_db': round(note['rms_ratio_db'], 2),
            'sort_max_raw': round(note['sort']['max_raw'], 4),
            'sort_max_enh': round(note['sort']['max_enh'], 4),
            'ctc_max_raw': round(note['ctc']['max_path_raw'], 4),
            'ctc_max_enh': round(note['ctc']['max_path_enh'], 4),
            'low_band_delta_db': note['band_delta_db']['low_0_1k'],
            'high_band_delta_db': note['band_delta_db']['high_4_8k'],
            'kw_mean_prob_raw': round(note['ctc']['token_stats_raw']['kw_mean_prob'], 6),
            'kw_mean_prob_enh': round(note['ctc']['token_stats_enh']['kw_mean_prob'], 6),
            'kw_max_prob_raw': round(note['ctc']['token_stats_raw']['kw_max_prob'], 6),
            'kw_max_prob_enh': round(note['ctc']['token_stats_enh']['kw_max_prob'], 6),
            'enh_rms_matched_ctc_max': note['candidate_scores']
                .get('enh_rms_matched', {}).get('ctc', {}).get('max_score', ''),
            'gain_p4_ctc_max': note['candidate_scores']
                .get('enh_gain_p4', {}).get('ctc', {}).get('max_score', ''),
            'mix_a050_ctc_max': note['candidate_scores']
                .get('mix_a050', {}).get('ctc', {}).get('max_score', ''),
            'mix_a050_sort_max': note['candidate_scores']
                .get('mix_a050', {}).get('sort', {}).get('max_score', ''),
        })

        # -------- 图：每条样本一页 --------
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 2, figsize=(13, 9))
        vmin = min(fb_raw.min(), fb_enh.min())
        vmax = max(fb_raw.max(), fb_enh.max())
        for ax, f, title in ((axes[0, 0], fb_raw, 'raw'), (axes[0, 1], fb_enh, 'GTCRN enhanced')):
            im = ax.imshow(f.T, origin='lower', aspect='auto', vmin=vmin, vmax=vmax,
                           cmap='magma', extent=[0, len(f) / 100, 0, 80])
            ax.set_title(f'id={sid} {title} log-mel (RMS {common.rms(raw if title=="raw" else enh):.4f})')
            ax.set_ylabel('mel bin')
            fig.colorbar(im, ax=ax, fraction=0.03)
        ax = axes[1, 0]
        ax.plot(np.arange(len(sort_raw)) * 10, sort_raw, label='raw', color='#7f8c8d', lw=1.2)
        ax.plot(np.arange(len(sort_enh)) * 10, sort_enh, label='enh', color='#c0392b', lw=1.2)
        ax.axhline(0.84, ls='--', color='k', lw=0.8)
        ax.set_title(f'SORT frame prob (trigger raw={base["sort_raw"][sid]["triggered"]}, '
                     f'enh={base["sort"][sid]["triggered"]})')
        ax.set_ylabel('prob'); ax.set_xlabel('ms'); ax.legend(fontsize=8); ax.grid(alpha=0.25)
        ax = axes[1, 1]
        ax.plot(np.arange(len(pc_raw)) * 10, pc_raw, label='raw', color='#7f8c8d', lw=1.2)
        ax.plot(np.arange(len(pc_enh)) * 10, pc_enh, label='enh', color='#c0392b', lw=1.2)
        ax.axhline(-0.10, ls='--', color='k', lw=0.8)
        ax.set_title(f'CTC keyword path score (trigger raw={base["ctc_raw"][sid]["triggered"]}, '
                     f'enh={base["ctc"][sid]["triggered"]})')
        ax.set_ylabel('nats/token'); ax.set_xlabel('ms'); ax.legend(fontsize=8); ax.grid(alpha=0.25)
        ax = axes[2, 0]
        w = 0.38
        x = np.arange(6)
        pr = [note['ctc']['token_stats_raw']['per_token_mean_prob'][v] for v in VOCAB]
        pe = [note['ctc']['token_stats_enh']['per_token_mean_prob'][v] for v in VOCAB]
        ax.bar(x - w / 2, pr, w, label='raw', color='#7f8c8d')
        ax.bar(x + w / 2, pe, w, label='enh', color='#c0392b')
        ax.set_xticks(x); ax.set_xticklabels(VOCAB)
        ax.set_yscale('log'); ax.set_ylim(1e-5, 1.5)
        ax.set_title('CTC token mean prob (log); kw-mean raw=%.4f enh=%.4f'
                     % (note['ctc']['token_stats_raw']['kw_mean_prob'],
                        note['ctc']['token_stats_enh']['kw_mean_prob']),
                     fontsize=9)
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
        ax = axes[2, 1]
        bands = list(be_raw)
        x = np.arange(len(bands)); w = 0.38
        ax.bar(x - w / 2, [be_raw[b] for b in bands], w, label='raw', color='#7f8c8d')
        ax.bar(x + w / 2, [be_enh[b] for b in bands], w, label='enh', color='#c0392b')
        ax.set_xticks(x); ax.set_xticklabels(bands, fontsize=8)
        ax.set_ylabel('dB'); ax.set_title('Fbank band energy')
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
        fig.tight_layout()
        out = common.FIG_DIR / f'case_{sid}.png'
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print('saved', out)

    common.write_csv(common.RESULTS_DIR / 'cases' / 'case_table.csv', out_rows)

    # 全局：分数变化与电平变化的相关性（正样本）
    s = json.load(open(common.RESULTS_DIR / 'sweep' / 'sweep_summary.json'))
    import csv as _csv
    rows = [(r['sample_id'], r['label'], r['raw_path'], r['enhanced_path']) for r in labels]
    d_sort, d_ctc, d_rms = [], [], []
    for sid, lab, _, _ in rows:
        sr = s['separability']['sort_raw']['max_scores'][str(sid)]
        se = s['separability']['sort_enh']['max_scores'][str(sid)]
        cr = s['separability']['ctc_raw']['max_scores'][str(sid)]
        ce = s['separability']['ctc_enh']['max_scores'][str(sid)]
        rec = sid_map[sid]
        raw = common.load_wav(rec['raw_path'])
        enh = common.load_wav(rec['enhanced_path'])
        d_sort.append(se - sr)
        d_ctc.append(ce - cr)
        d_rms.append(20 * np.log10(common.rms(enh) / common.rms(raw)))
    from scipy.stats import spearmanr
    notes['_global_level_correlation'] = {
        'n': len(d_rms),
        'spearman_sort_delta_vs_rms_db': {
            'r': float(spearmanr(d_rms, d_sort).correlation),
            'p': float(spearmanr(d_rms, d_sort).pvalue)},
        'spearman_ctc_delta_vs_rms_db': {
            'r': float(spearmanr(d_rms, d_ctc).correlation),
            'p': float(spearmanr(d_rms, d_ctc).pvalue)},
        'rms_change_db_median': float(np.median(d_rms)),
        'rms_change_db_p10': float(np.percentile(d_rms, 10)),
        'rms_change_db_p90': float(np.percentile(d_rms, 90)),
    }
    common.write_json(common.RESULTS_DIR / 'cases' / 'summary.json', notes)
    print(json.dumps(notes['_global_level_correlation'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
