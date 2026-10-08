"""Wave L3R — LLM-native калибровка и независимые length-matched повторы (мега-скрипт VM).

Секции: env → extract (кэш эмбеддингов с чекпоинтами) → pytest → load →
null (dev 500×n-grid + val 200×n-grid, correct-vs-correct) → contrast
(dev/val/test × alpha × 30 length-matched репов, n=500) → permutation
(500 сплитов меток канонического L3d) → analysis → figures → summary.
RTD не вычисляется (secondary по протоколу, бюджет времени). Resume —
по experiment_id в rows.jsonl. SystemExit не используется.
"""
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

RAW = 'results/raw/l3r'
TABLES = 'results/tables'
FIGURES = 'results/figures'
LOG_PATH = 'logs/l3r_run.log'
ROWS_PATH = os.path.join(RAW, 'rows.jsonl')
PROTOCOL_PATH = 'protocols/l3r_protocol.json'
CACHE_DIR = 'embeddings/l3r_cache'
EXTRACT_SCRIPT = 'scripts/extract_llm_embeddings.py'

N_GRID = [50, 100, 250, 500, 1000]
NULL_DEV_PAIRS = 500
NULL_VAL_PAIRS = 200
N_CLOUD = 500
REPS = 30
ALPHA_GRID = [0.0, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
N_PERMS = 500
LAYER = 18
PCA_DIM = 16

METRICS = ['mtd_PQ', 'mtd_QP', 'ntd_PQ', 'ntd_QP',
           'precision@3', 'recall@3', 'mmd', 'frechet', 'js']
SIMILARITY = {'precision@3', 'recall@3'}
PRIMARY = 'precision@3'

SECTION_STATUS = {}
_STARTED = time.perf_counter()


class Tee:
    def __init__(self, path):
        self._file = open(path, 'a', encoding='utf-8')

    def write(self, text):
        sys.__stdout__.write(text)
        self._file.write(text)

    def flush(self):
        sys.__stdout__.flush()
        self._file.flush()


def progress(text):
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(65536), b''):
            digest.update(chunk)
    return digest.hexdigest()


def section(name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            progress(f'=== SECTION {name} START ===')
            started = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                SECTION_STATUS[name] = 'ok'
                progress(f'=== SECTION {name} OK ({time.perf_counter() - started:.1f}s) ===')
                return result
            except Exception:
                SECTION_STATUS[name] = 'failed: ' + traceback.format_exc(limit=3)
                progress(f'=== SECTION {name} FAIL ===')
                progress(traceback.format_exc(limit=3))
                return None
        return wrapper
    return decorator


def append_row(row, handle):
    handle.write(json.dumps(json_safe(row)) + '\n')
    handle.flush()


def metrics_of(metrics_obj, P, Q):
    P_pca = pca_reduce(P, P, PCA_DIM)
    Q_pca = pca_reduce(P, Q, PCA_DIM)
    result = metrics_obj.compute_all(P_pca, Q_pca, skip=('rtd',))
    out = {}
    for key, value in result.items():
        if key in ('precision@1', 'precision@10', 'recall@1', 'recall@10'):
            continue
        out[key] = None if value is None else float(value)
    return out


@section('env')
def section_env(_):
    import numpy
    versions = {'python': sys.version.split()[0], 'numpy': numpy.__version__}
    for name in ('pandas', 'scipy', 'torch', 'transformers', 'pyarrow'):
        try:
            module = __import__(name)
            versions[name] = getattr(module, '__version__', '?')
        except Exception:
            versions[name] = None
    progress(f'versions: {versions}')
    progress(f'protocol sha256: {_sha256_file(PROTOCOL_PATH)}')
    return {'versions': versions}


@section('extract')
def section_extract(_):
    cache_marker = os.path.join(CACHE_DIR, 'vectors.npy')
    if os.path.exists(cache_marker):
        progress(f'кэш {CACHE_DIR} существует — извлечение пропущено')
        return
    proc = subprocess.run(
        [sys.executable, EXTRACT_SCRIPT, '--out-dir', CACHE_DIR],
        capture_output=True, text=True,
    )
    print(proc.stdout[-3000:])
    print(proc.stderr[-500:])
    if proc.returncode != 0 or not os.path.exists(cache_marker):
        raise RuntimeError(f'extract rc={proc.returncode}')
    progress('извлечение эмбеддингов завершено')


@section('pytest_l3r')
def section_pytest(_):
    proc = subprocess.run(
        [sys.executable, '-m', 'pytest', 'tests/test_sample_length_matched_pair.py', '-q'],
        capture_output=True, text=True, timeout=900,
    )
    print(proc.stdout[-1500:])
    if proc.returncode != 0:
        raise RuntimeError(f'pytest rc={proc.returncode}')
    progress('pytest L3R OK')


def load_state():
    from tda_metrics.embedding_cache import load_cache
    from tda_metrics.llm_experiments import prompt_id_split
    meta, vectors, manifest = load_cache(
        CACHE_DIR,
        expect={'model': 'Qwen/Qwen2.5-3B-Instruct',
                'input_format': 'chat_knowledge_v1',
                'pooling': 'mean_answer'},
    )
    layer_indices = manifest['layer_indices']
    if LAYER not in layer_indices:
        raise RuntimeError(f'layer {LAYER} нет в кэше: {layer_indices}')
    pos = layer_indices.index(LAYER)
    meta_l = meta[meta['layer_index'] == LAYER].reset_index(drop=True)
    vectors_l = vectors[:, pos, :]
    splits = prompt_id_split(meta_l, seed=0)
    return meta_l, vectors_l, splits, manifest


def select_items(meta_l, vectors_l, prompts, kind):
    import numpy as np
    allowed = set(prompts)
    mask = (meta_l['answer_kind'] == kind) & meta_l['prompt_id'].isin(allowed)
    return vectors_l[mask.to_numpy()]


def pair_clouds(meta, vectors, prompts_a, prompts_b, n, rng):
    pa = sorted(rng.choice(len(prompts_a), size=n, replace=False).tolist())
    pb = sorted(rng.choice(len(prompts_b), size=n, replace=False).tolist())
    P = select_items(meta, vectors, [prompts_a[i] for i in pa], 'correct')
    Q = select_items(meta, vectors, [prompts_b[i] for i in pb], 'correct')
    return P, Q


@section('null')
def section_null(deps):
    import numpy as np
    meta, vectors, splits, _ = deps['state']
    metrics_obj = deps['metrics_obj']
    os.makedirs(RAW, exist_ok=True)
    done = load_done_ids()
    handle = open(ROWS_PATH, 'a', encoding='utf-8')
    dev = splits['dev']
    val = splits['val']
    dev_a, dev_b = dev[::2], dev[1::2]
    val_a, val_b = val[::2], val[1::2]
    counter = 0
    t0 = time.perf_counter()
    for split_name, half_a, half_b, n_pairs in (
        ('dev', dev_a, dev_b, NULL_DEV_PAIRS),
        ('val', val_a, val_b, NULL_VAL_PAIRS),
    ):
        for n in N_GRID:
            for i in range(n_pairs):
                eid = f'null/{split_name}/n{n}/p{i}'
                if eid in done:
                    continue
                rng = np.random.default_rng(1000 + i if split_name == 'dev' else 5000 + i)
                try:
                    P, Q = pair_clouds(meta, vectors, half_a, half_b, n, rng)
                    row = {'experiment_id': eid, 'kind': 'null', 'split': split_name,
                           'n': n, 'pair_idx': i, 'alpha': None, 'rep': None,
                           'status': 'ok', 'error_message': ''}
                    row.update(metrics_of(metrics_obj, P, Q))
                except Exception as exc:
                    row = {'experiment_id': eid, 'kind': 'null', 'split': split_name,
                           'n': n, 'pair_idx': i, 'alpha': None, 'rep': None,
                           'status': 'failed', 'error_message': f'{type(exc).__name__}: {exc}'}
                append_row(row, handle)
                counter += 1
                if counter % 50 == 0:
                    elapsed = time.perf_counter() - t0
                    progress(f'null: {counter} пар записано, {elapsed / counter:.1f}s/пара')
    handle.close()
    progress(f'null: {counter} новых пар')


@section('contrast')
def section_contrast(deps):
    import numpy as np
    from tda_metrics.llm_experiments import independent_mixture
    meta_l, vectors_l, splits, _ = deps['state']
    metrics_obj = deps['metrics_obj']
    done = load_done_ids()
    handle = open(ROWS_PATH, 'a', encoding='utf-8')
    counter = 0
    t0 = time.perf_counter()
    for split_name in ('dev', 'val', 'test'):
        prompts = splits[split_name]
        for rep in range(REPS):
            eid_pair = f'lm/{split_name}/r{rep}'
            try:
                correct_m, hall_m, meta_c, meta_h, info = sample_lm_pair(
                    meta_l, vectors_l, prompts, seed=2000 + rep, n_cloud=N_CLOUD)
                length_diag = info['length_diagnostics']
            except Exception as exc:
                row = {'experiment_id': eid_pair, 'kind': 'lm_pair', 'split': split_name,
                       'n': N_CLOUD, 'rep': rep, 'alpha': None, 'status': 'failed',
                       'error_message': f'{type(exc).__name__}: {exc}'}
                append_row(row, handle)
                counter += 1
                continue
            row_pair = {'experiment_id': eid_pair, 'kind': 'lm_pair', 'split': split_name,
                         'n': N_CLOUD, 'rep': rep, 'alpha': None, 'status': 'ok',
                         'error_message': '',
                         'n_selected': int(info['n_selected']),
                         'ks_stat': float(length_diag['ks_stat']),
                         'ks_p': float(length_diag['ks_p']),
                         'mean_correct': float(length_diag['mean_correct']),
                         'mean_hallucinated': float(length_diag['mean_hallucinated'])}
            if eid_pair not in done:
                append_row(row_pair, handle)
            perm = np.random.default_rng(3000 + rep).permutation(len(correct_m))
            for alpha in ALPHA_GRID:
                eid = f'contrast/{split_name}/a{alpha:.4g}/r{rep}'
                if eid in done:
                    continue
                try:
                    Q_mix = independent_mixture(correct_m, hall_m, alpha, perm)
                    row = {'experiment_id': eid, 'kind': 'contrast', 'split': split_name,
                           'n': N_CLOUD, 'rep': rep, 'alpha': alpha, 'status': 'ok',
                           'error_message': ''}
                    row.update(metrics_of(metrics_obj, correct_m, Q_mix))
                except Exception as exc:
                    row = {'experiment_id': eid, 'kind': 'contrast', 'split': split_name,
                           'n': N_CLOUD, 'rep': rep, 'alpha': alpha, 'status': 'failed',
                           'error_message': f'{type(exc).__name__}: {exc}'}
                append_row(row, handle)
                counter += 1
                if counter % 60 == 0:
                    elapsed = time.perf_counter() - t0
                    progress(f'contrast: {counter} строк, {elapsed / counter:.1f}s/строка')
    handle.close()
    progress(f'contrast: {counter} новых строк')


def sample_lm_pair(meta_l, vectors_l, prompts, seed, n_cloud):
    from tda_metrics.llm_experiments import sample_length_matched_pair
    return sample_length_matched_pair(
        meta_l, vectors_l, prompts, LAYER, seed=seed, n_cloud=n_cloud, bins=10)


@section('permutation')
def section_permutation(deps):
    import numpy as np
    from tda_metrics.llm_experiments import shuffled_label_pair
    meta_l, vectors_l, splits, _ = deps['state']
    metrics_obj = deps['metrics_obj']
    done = load_done_ids()
    correct_m, hall_m, _, _, _ = sample_lm_pair(
        meta_l, vectors_l, splits['dev'], seed=2000, n_cloud=N_CLOUD)
    handle = open(ROWS_PATH, 'a', encoding='utf-8')
    counter = 0
    t0 = time.perf_counter()
    joint = np.concatenate([correct_m, hall_m])
    labels = np.concatenate([np.zeros(len(correct_m)), np.ones(len(hall_m))])
    from tda_metrics.statistics import paired_permutation_labels
    for i in range(N_PERMS):
        eid = f'perm/dev/p{i}'
        if eid in done:
            continue
        signs = paired_permutation_labels(len(correct_m), 1, seed=7000 + i)[0]
        labels_perm = labels.copy()
        flip = signs < 0
        labels_perm[:len(correct_m)][flip] = 1.0
        labels_perm[len(correct_m):][flip] = 0.0
        P = joint[labels_perm == 0]
        Q = joint[labels_perm == 1]
        try:
            row = {'experiment_id': eid, 'kind': 'perm', 'split': 'dev',
                   'n': N_CLOUD, 'rep': None, 'alpha': None, 'status': 'ok',
                   'error_message': ''}
            row.update(metrics_of(metrics_obj, P, Q))
        except Exception as exc:
            row = {'experiment_id': eid, 'kind': 'perm', 'split': 'dev',
                   'n': N_CLOUD, 'rep': None, 'alpha': None, 'status': 'failed',
                   'error_message': f'{type(exc).__name__}: {exc}'}
        append_row(row, handle)
        counter += 1
        if counter % 50 == 0:
            elapsed = time.perf_counter() - t0
            progress(f'perm: {counter}/{N_PERMS}, {elapsed / counter:.1f}s/перм')
    handle.close()
    progress(f'perm: {counter} новых перестановок')


def load_done_ids():
    done = set()
    if os.path.exists(ROWS_PATH):
        with open(ROWS_PATH, encoding='utf-8') as handle:
            for line in handle:
                try:
                    rec = json.loads(line)
                    if rec.get('status') == 'ok':
                        done.add(rec['experiment_id'])
                except Exception:
                    continue
    return done


def read_rows():
    import pandas as pd
    rows = []
    with open(ROWS_PATH, encoding='utf-8') as handle:
        for line in handle:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


@section('analysis')
def section_analysis(deps):
    import numpy as np
    import pandas as pd
    from tda_metrics.statistics import benjamini_hochberg
    from scipy.stats import binomtest

    frame = read_rows()
    if frame.empty:
        raise RuntimeError('rows.jsonl пуст')
    null = frame[(frame['kind'] == 'null') & (frame['status'] == 'ok')]
    contrast = frame[(frame['kind'] == 'contrast') & (frame['status'] == 'ok')]
    perm = frame[(frame['kind'] == 'perm') & (frame['status'] == 'ok')]
    if null.empty or contrast.empty:
        raise RuntimeError('null/contrast пусты — analysis невозможен')

    thresholds = []
    for n in N_GRID:
        for metric in METRICS:
            dev_vals = null[(null['split'] == 'dev') & (null['n'] == n)][metric].dropna()
            if len(dev_vals) < 50:
                continue
            mu, sd = float(dev_vals.mean()), float(dev_vals.std(ddof=1))
            degenerate = sd <= 0.0
            z = (dev_vals - mu) / (sd + 1e-12)
            z_crit = float(np.quantile(np.abs(z), 0.95))
            row = {'n': n, 'metric': metric, 'null_mean': mu, 'null_sd': sd,
                   'z_crit': z_crit, 'n_null_dev': int(len(dev_vals)),
                   'degenerate': degenerate}
            val_vals = null[(null['split'] == 'val') & (null['n'] == n)][metric].dropna()
            if len(val_vals) >= 50:
                z_val = np.abs((val_vals - mu) / (sd + 1e-12))
                fpr = float(np.mean(z_val > z_crit))
                ci = binomtest(int(round(fpr * len(val_vals))), len(val_vals), 0.05
                               ).proportion_ci(0.95)
                row['fpr_val'] = fpr
                row['fpr_val_ci_low'] = float(ci.low)
                row['fpr_val_ci_high'] = float(ci.high)
                row['n_null_val'] = int(len(val_vals))
            thresholds.append(row)
    thr = pd.DataFrame(thresholds)

    base = thr[thr['n'] == N_CLOUD].set_index('metric')
    power_rows = []
    pvals = []
    for split_name in ('dev', 'val', 'test'):
        block = contrast[contrast['split'] == split_name]
        for alpha in ALPHA_GRID:
            for metric in METRICS:
                vals = block[block['alpha'] == alpha][metric].dropna()
                if vals.empty or metric not in base.index:
                    continue
                mu = float(base.loc[metric, 'null_mean'])
                sd = float(base.loc[metric, 'null_sd'])
                zc = float(base.loc[metric, 'z_crit'])
                sign = -1.0 if metric in SIMILARITY else 1.0
                z_obs = (vals - mu) * sign / (sd + 1e-12)
                power = float(np.mean(np.abs(z_obs) > zc))
                mean_z = float(np.mean(z_obs))
                pvals.append(abs(mean_z))
                power_rows.append({
                    'split': split_name, 'alpha': alpha, 'metric': metric,
                    'mean': float(vals.mean()), 'mean_z': mean_z,
                    'power': power, 'n_reps': int(len(vals)),
                    'z_crit': zc, 'null_sd': float(base.loc[metric, 'null_sd']),
                })
    power = pd.DataFrame(power_rows)
    p = np.array(pvals)
    p_values = np.clip(2 * (1 - _norm_sf(p)), 0.0, 1.0)
    mask = power['metric'] != PRIMARY
    power['p_value'] = p_values
    power['q_bh'] = np.nan
    if mask.any():
        _, adjusted = benjamini_hochberg(power.loc[mask, 'p_value'].to_numpy(), alpha=0.05)
        power.loc[mask, 'q_bh'] = adjusted

    perm_rows = []
    canonical = contrast[(contrast['split'] == 'dev') & (contrast['alpha'] == 1.0)]
    for metric in METRICS:
        if metric not in base.index:
            continue
        obs = canonical[metric].dropna()
        pv = perm[metric].dropna()
        if obs.empty or pv.empty:
            continue
        mu = float(base.loc[metric, 'null_mean'])
        sd = float(base.loc[metric, 'null_sd'])
        z_obs = float((obs.mean() - mu) / (sd + 1e-12))
        z_perm = np.abs((pv - mu) / (sd + 1e-12))
        perm_rows.append({
            'metric': metric, 'z_obs': z_obs,
            'p_perm': float(np.mean(np.abs(z_perm) >= abs(z_obs))),
            'n_perm': int(len(pv)),
        })

    lm = frame[(frame['kind'] == 'lm_pair') & (frame['status'] == 'ok')]
    balance = lm.groupby('split').agg(
        n_reps=('ks_stat', 'size'),
        ks_stat_mean=('ks_stat', 'mean'), ks_stat_max=('ks_stat', 'max'),
        ks_p_min=('ks_p', 'min'), ks_p_median=('ks_p', 'median'),
        mean_correct=('mean_correct', 'mean'),
        mean_hallucinated=('mean_hallucinated', 'mean'),
    ).reset_index()

    os.makedirs(TABLES, exist_ok=True)
    thr.to_csv(os.path.join(TABLES, 'l3r_null_power.csv'), index=False)
    power.to_csv(os.path.join(TABLES, 'l3r_power_by_alpha.csv'), index=False)
    pd.DataFrame(perm_rows).to_csv(
        os.path.join(TABLES, 'l3r_permutation.csv'), index=False)
    balance.to_csv(os.path.join(TABLES, 'l3r_length_balance.csv'), index=False)

    primary_rows = power[(power['metric'] == PRIMARY) & (power['alpha'] == 1.0)]
    rng = np.random.default_rng(0)
    primary = primary_rows.set_index('split')['power'].to_dict()
    primary_ci = {}
    for split_name in ('dev', 'val', 'test'):
        vals = contrast[(contrast['split'] == split_name) & (contrast['alpha'] == 1.0)][PRIMARY].dropna()
        if vals.empty:
            continue
        boots = []
        arr = vals.to_numpy()
        for _ in range(1000):
            idx = rng.integers(0, len(arr), len(arr))
            boots.append(float(np.mean(arr[idx])))
        primary_ci[split_name] = [float(np.quantile(boots, 0.025)),
                                  float(np.quantile(boots, 0.975))]

    return {
        'thresholds': thr.to_dict('records'),
        'primary_endpoint': {
            'metric': PRIMARY, 'alpha': 1.0, 'n': N_CLOUD,
            'power_by_split': primary,
            'power_bootstrap_ci': primary_ci,
        },
        'permutation': perm_rows,
        'length_balance': balance.to_dict('records'),
    }


def _norm_sf(x):
    from scipy.stats import norm
    return norm.sf(x)


@section('figures')
def section_figures(deps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd

    analysis = deps.get('analysis')
    if analysis is None:
        raise RuntimeError('analysis недоступен')
    power = pd.read_csv(os.path.join(TABLES, 'l3r_power_by_alpha.csv'))
    os.makedirs(FIGURES, exist_ok=True)
    metrics_plot = ['precision@3', 'ntd_QP', 'ntd_PQ', 'mtd_QP', 'recall@3']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, split_name in zip(axes, ('dev', 'val', 'test')):
        for metric in metrics_plot:
            block = power[(power['split'] == split_name)
                          & (power['metric'] == metric)].sort_values('alpha')
            ax.plot(block['alpha'], block['power'], marker='o', label=metric)
        ax.set_title(split_name)
        ax.set_xlabel('alpha (доля галлюцинаций)')
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel('power @ |z|>z_crit(5%)')
    axes[0].legend(fontsize=8)
    fig.suptitle('L3R: мощность по alpha (length-matched, n=500, PCA-16, layer 18)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'l3r_power_curves.png'), dpi=130)
    plt.close(fig)

    lm = pd.read_csv(os.path.join(TABLES, 'l3r_length_balance.csv'))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(lm['split'], lm['ks_stat_mean'], yerr=lm['ks_stat_max'] - lm['ks_stat_mean'])
    ax.set_ylabel('KS-статистика длин (mean, усы: max)')
    ax.set_title('L3R: length-balance по сплитам')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'l3r_length_balance.png'), dpi=130)
    plt.close(fig)
    progress('figures: l3r_power_curves.png, l3r_length_balance.png')


@section('summary')
def section_summary(deps):
    from tda_metrics.llm_experiments import prompt_ids_hash
    analysis = deps.get('analysis')
    _, _, splits, cache_manifest = deps['state']
    summary = {
        'created': datetime.now(timezone.utc).isoformat(),
        'sections': SECTION_STATUS,
        'primary_endpoint': (analysis or {}).get('primary_endpoint'),
        'permutation': (analysis or {}).get('permutation'),
        'length_balance': (analysis or {}).get('length_balance'),
        'split_hashes': {k: prompt_ids_hash(v) for k, v in splits.items()},
        'cache_manifest_keys': {k: cache_manifest.get(k) for k in
                                ('model', 'input_format', 'pooling', 'layer_indices')},
    }
    manifest = {
        'created': summary['created'],
        'protocol_path': PROTOCOL_PATH,
        'protocol_sha256': _sha256_file(PROTOCOL_PATH),
        'sections': SECTION_STATUS,
        'grids': {
            'n_grid': N_GRID, 'null_dev': NULL_DEV_PAIRS, 'null_val': NULL_VAL_PAIRS,
            'n_cloud': N_CLOUD, 'reps': REPS, 'alpha_grid': ALPHA_GRID,
            'n_permutations': N_PERMS, 'layer': LAYER, 'pca_dim': PCA_DIM,
            'rtd': 'не вычислялся (secondary, бюджет времени)',
        },
        'split_hashes': summary['split_hashes'],
        'runtime_seconds': time.perf_counter() - _STARTED,
    }
    with open(os.path.join(RAW, 'summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(json_safe(summary), handle, ensure_ascii=False, indent=2)
    with open(os.path.join(RAW, 'manifest.json'), 'w', encoding='utf-8') as handle:
        json.dump(json_safe(manifest), handle, ensure_ascii=False, indent=2)
    progress('summary/manifest записаны')


def main():
    deps = {}
    protocol_missing = not os.path.exists(PROTOCOL_PATH)
    progress(f'L3R start: protocol_missing={protocol_missing}')
    try:
        import numpy as np
        from tda_metrics.metrics import TopologyMetrics
        globals()['np'] = np
        globals()['pca_reduce'] = __import__(
            'tda_metrics.llm_experiments', fromlist=['pca_reduce']).pca_reduce
        deps['metrics_obj'] = TopologyMetrics(seed=0)
    except Exception:
        progress('HEAVY IMPORT FAIL:\n' + traceback.format_exc(limit=3))
        print('L3R DONE rc=1')
        return
    section_env(None)
    section_extract(None)
    section_pytest(None)
    try:
        deps['state'] = load_state()
    except Exception:
        progress('LOAD FAIL:\n' + traceback.format_exc(limit=3))
        print('L3R DONE rc=1')
        return
    section_null(deps)
    section_contrast(deps)
    section_permutation(deps)
    analysis = section_analysis(deps)
    deps['analysis'] = analysis
    section_figures(deps)
    section_summary(deps)
    ok = SECTION_STATUS.get('null') == 'ok' and SECTION_STATUS.get('contrast') == 'ok'
    analysis_ok = SECTION_STATUS.get('analysis') == 'ok'
    print('L3R DONE rc=' + ('0' if (ok and analysis_ok) else '1'))


if __name__ == '__main__':
    os.makedirs('logs', exist_ok=True)
    os.makedirs(RAW, exist_ok=True)
    sys.stdout = Tee(LOG_PATH)
    main()
