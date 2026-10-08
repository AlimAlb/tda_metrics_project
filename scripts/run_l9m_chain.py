"""L9M: length-matched packet dose response без confidence-признаков (wave L9M).

Запускается на Colab VM (heavy-стек: torch не нужен — только кэш экстракции,
TF-метрики, ripser++/MTopDiv, pandas/scipy/sklearn). Поток: восстановление
кэша -> pytest-гейт -> пилот биннинга (лок n_bins/n_cloud по всем
запланированным сидам) -> same-law null-пары (dev/val/test) и length-matched
пакеты (dev: 5 фолдов x 4 повтора x 8 alpha; val/test: без фолдов,
4 повтора x 8 alpha; alpha одного (fold, repeat) вложены — дозовые кривые,
не независимые пакеты) -> калибровка порогов на dev-null (направленный
threshold, coverage -> 'less', дистанции -> 'greater') -> FPR с binomial CI
(val/test null) и power по alpha -> dose-response, single-feature OOF,
length-balance -> figures/verdict/summary.

Отклонение от пакета: rtd исключён из батареи (стоимость ~x2 и шумовой пол;
прецедент L3R) — topology_only = {mtd_PQ, mtd_QP, ntd_PQ, ntd_QP}; задокументировано
в протоколе. Resume-safe: строки дописываются в rows.jsonl по experiment_id;
SystemExit не поднимается (только печать rc).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

PROTOCOL_PATH = 'protocols/l9m_protocol.json'
CACHE_DIR = 'embeddings/l9m_cache'
RAW = 'results/raw/l9m'
TABLES = 'results/tables'
FIGURES = 'results/figures'
ROWS_PATH = os.path.join(RAW, 'rows.jsonl')
LOG_PATH = 'logs/l9m_run.log'

LAYER = 18
PCA_DIM = 16
N_CLOUD_TARGET = 500
N_CLOUD_FLOOR = 250
BIN_LADDER = (10, 8, 6, 5, 4, 3, 2)
N_FOLDS = 5
FOLD_SEED = 500
REPEATS = 4
ALPHA_GRID = [0.0, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0]
NULL_DEV_PER_FOLD = 100
NULL_VAL = 200
NULL_TEST = 200

METRICS = ['mtd_PQ', 'mtd_QP', 'ntd_PQ', 'ntd_QP',
           'precision@1', 'recall@1', 'precision@3', 'recall@3',
           'precision@10', 'recall@10', 'mmd', 'frechet', 'js']
COVERAGE = {'precision@1', 'recall@1', 'precision@3', 'recall@3',
            'precision@10', 'recall@10'}
GROUPS_OOF = {
    'topology_only': ['mtd_PQ', 'mtd_QP', 'ntd_PQ', 'ntd_QP'],
    'coverage_only': ['precision@1', 'recall@1', 'precision@3', 'recall@3',
                      'precision@10', 'recall@10'],
    'statistics_only': ['mmd', 'frechet', 'js'],
}
PRIMARY = 'precision@3'
Q_LEN_AUROC_MAX = 0.55
SMD_TOLERANCE = 0.1

EXPECTED_SPLIT_HASHES = {
    'dev': 'debfc629f24af648526470f23db8fe2d837ea40e06eeaea8b24d32c54559a3cf',
    'val': '98c55b0de59675e6f9ba8ccc33e0d0a3ac79f71069345b5cf2a71371b3b1a1a4',
    'test': 'f4145eb02aa948656c9ce44adfd4f10dcc7ba51a568ee24e454add962708d5f7',
}

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


def dev_packet_seeds(fold, repeat):
    return (100000 + 1000 * fold + repeat, 200000 + 1000 * fold + repeat,
            300000 + 1000 * fold + repeat)


def dev_null_seeds(fold, pair):
    return (400000 + 1000 * fold + pair, 500000 + 1000 * fold + pair)


def val_packet_seeds(repeat):
    return (700000 + repeat, 800000 + repeat, 900000 + repeat)


def val_null_seeds(pair):
    return (1000000 + pair, 1100000 + pair)


def test_packet_seeds(repeat):
    return (1200000 + repeat, 1300000 + repeat, 1400000 + repeat)


def test_null_seeds(pair):
    return (1500000 + pair, 1600000 + pair)


def fold_pools(meta_l, x16, prompts):
    import numpy as np
    allowed = set(prompts)
    prompt_np = meta_l['prompt_id'].to_numpy()
    kind_np = meta_l['answer_kind'].to_numpy()
    len_np = meta_l['n_pooled'].to_numpy()
    mask_c = np.isin(prompt_np, list(allowed)) & (kind_np == 'correct')
    mask_h = np.isin(prompt_np, list(allowed)) & (kind_np == 'hallucinated')
    rows_c = np.flatnonzero(mask_c)
    rows_h = np.flatnonzero(mask_h)
    if len(rows_c) == 0 or len(rows_h) == 0:
        raise RuntimeError(f'пул пуст: correct {len(rows_c)}, hallucinated {len(rows_h)}')
    return {
        'correct_matrix': x16[rows_c],
        'correct_index': rows_c,
        'correct_lengths': len_np[rows_c],
        'correct_prompts': prompt_np[rows_c],
        'hall_matrix': x16[rows_h],
        'hall_index': rows_h,
        'hall_lengths': len_np[rows_h],
        'hall_prompts': prompt_np[rows_h],
    }


def sample_pools(pools, n_cloud, seed_half, seed_sample, n_bins):
    from tda_metrics.l9m_pools import sample_length_matched_pools
    return sample_length_matched_pools(
        pools['correct_matrix'], pools['correct_index'],
        pools['correct_lengths'], pools['correct_prompts'],
        pools['hall_matrix'], pools['hall_index'],
        pools['hall_lengths'], pools['hall_prompts'],
        n_cloud=n_cloud, seed_half=seed_half, seed_sample=seed_sample,
        n_bins=n_bins, bin_ladder=())


def composition_hash(packet):
    p_part = '|'.join(f'{kind}:{index}' for kind, index in packet['P_items'])
    q_part = '|'.join(f'{kind}:{index}' for kind, index in packet['Q_items'])
    return hashlib.sha256((p_part + '||' + q_part).encode()).hexdigest()[:16]


def metrics_of(metrics_obj, P, Q):
    import numpy as np
    result = metrics_obj.compute_all(
        np.asarray(P, dtype=float), np.asarray(Q, dtype=float), skip=('rtd',))
    out = {}
    for key, value in result.items():
        out[key] = None if value is None else float(value)
    return out


def packet_row(eid, kind, split, fold, repeat, pair_idx, alpha, seeds,
               pools_obj, packet, metrics, seconds):
    import numpy as np
    diagnostics = pools_obj['diagnostics']
    return {
        'experiment_id': eid, 'kind': kind, 'split': split, 'fold': fold,
        'repeat': repeat, 'pair_idx': pair_idx, 'alpha': alpha,
        'seed_half': seeds[0], 'seed_sample': seeds[1], 'seed_mix': seeds[2],
        'n_cloud_actual': pools_obj['n_cloud_actual'],
        'n_bins_actual': pools_obj['n_bins_actual'],
        'bin_quotas': pools_obj['bin_quotas'],
        'n_hall': packet['n_hall'],
        'p_len_mean': float(np.mean(packet['p_lengths'])),
        'q_len_mean': float(np.mean(packet['q_lengths'])),
        'smd_pq': float(diagnostics['smd_pq']),
        'p_lengths': [int(v) for v in packet['p_lengths']],
        'q_lengths': [int(v) for v in packet['q_lengths']],
        'composition_hash': composition_hash(packet),
        'seconds': seconds, 'status': 'ok', 'error_message': '',
        **metrics,
    }


@section('env')
def section_env(_):
    import numpy
    versions = {'python': sys.version.split()[0], 'numpy': numpy.__version__}
    for name in ('pandas', 'scipy', 'sklearn', 'pyarrow'):
        try:
            module = __import__(name)
            versions[name] = getattr(module, '__version__', '?')
        except Exception:
            versions[name] = None
    progress(f'versions: {versions}')
    progress(f'protocol sha256: {_sha256_file(PROTOCOL_PATH)}')
    return {'versions': versions}


@section('pytest')
def section_pytest(_):
    proc = subprocess.run(
        [sys.executable, '-m', 'pytest',
         'tests/test_l9m_pools.py', 'tests/test_l9_tasks.py',
         'tests/test_llm_experiments.py', 'tests/test_sample_length_matched_pair.py',
         '-q'],
        capture_output=True, text=True, timeout=1800,
    )
    print(proc.stdout[-2000:])
    if proc.returncode != 0:
        raise RuntimeError(f'pytest rc={proc.returncode}')
    progress('pytest L9M-гейт OK')


def load_state():
    import numpy as np
    from tda_metrics.embedding_cache import load_cache
    from tda_metrics.llm_experiments import prompt_id_split, prompt_ids_hash
    from tda_metrics.reduction import pca_fit, pca_transform
    from tda_metrics.l9_tasks import prompt_folds

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

    split_hashes = {name: prompt_ids_hash(ids) for name, ids in splits.items()}
    for name, expected in EXPECTED_SPLIT_HASHES.items():
        if split_hashes[name] != expected:
            raise RuntimeError(
                f'сплит {name} не совпал с протоколом: {split_hashes[name]} != {expected}')
    progress(f'split hashes совпали с L3R: {split_hashes}')

    dev_mask = (meta_l['prompt_id'].isin(set(splits['dev']))
                & (meta_l['answer_kind'] == 'correct')).to_numpy()
    dev_correct = vectors_l[dev_mask]
    progress(f'PCA-{PCA_DIM}: fit на dev-correct ({len(dev_correct)} строк)')
    model = pca_fit(dev_correct, n_components=PCA_DIM)
    x16, info = pca_transform(vectors_l, model=model)
    progress(f'PCA explained total: {info.get("total")}')
    x16 = np.asarray(x16, dtype=float)

    folds = prompt_folds(splits['dev'], n_folds=N_FOLDS, seed=FOLD_SEED)
    pools = {
        ('dev', fold): fold_pools(meta_l, x16, prompts)
        for fold, prompts in enumerate(folds)
    }
    pools[('val', None)] = fold_pools(meta_l, x16, splits['val'])
    pools[('test', None)] = fold_pools(meta_l, x16, splits['test'])
    for key, value in pools.items():
        progress(f'пул {key}: correct {len(value["correct_index"])}, '
                 f'hall {len(value["hall_index"])}')
    return {
        'meta_l': meta_l, 'splits': splits, 'split_hashes': split_hashes,
        'manifest': manifest, 'pools': pools,
    }


@section('pilot')
def section_pilot(deps):
    import numpy as np
    pools = deps['pools']
    planned = []
    for fold in range(N_FOLDS):
        for repeat in range(REPEATS):
            planned.append((('dev', fold), dev_packet_seeds(fold, repeat)))
        for pair in range(NULL_DEV_PER_FOLD):
            planned.append((('dev', fold), dev_null_seeds(fold, pair)))
    for repeat in range(REPEATS):
        planned.append((('val', None), val_packet_seeds(repeat)))
    for pair in range(NULL_VAL):
        planned.append((('val', None), val_null_seeds(pair)))
    for repeat in range(REPEATS):
        planned.append((('test', None), test_packet_seeds(repeat)))
    for pair in range(NULL_TEST):
        planned.append((('test', None), test_null_seeds(pair)))
    progress(f'пилот: {len(planned)} сэмплов по лестнице {BIN_LADDER}')

    lock = None
    ladder_report = []
    for n_bins in BIN_LADDER:
        totals = np.empty(len(planned), dtype=np.int64)
        for i, (pool_key, seeds) in enumerate(planned):
            pools_obj = sample_pools(pools[pool_key], N_CLOUD_TARGET,
                                      seeds[0], seeds[1], n_bins)
            totals[i] = pools_obj['n_cloud_actual']
        min_total = int(totals.min())
        ladder_report.append({'n_bins': n_bins, 'min_total': min_total})
        progress(f'пилот n_bins={n_bins}: min_total={min_total}')
        if min_total >= N_CLOUD_TARGET:
            lock = {'n_bins': n_bins, 'n_cloud': N_CLOUD_TARGET}
            break
    if lock is None:
        best = max(ladder_report, key=lambda item: item['min_total'])
        if best['min_total'] < N_CLOUD_FLOOR:
            raise RuntimeError(
                f'пилот: лучший min_total={best["min_total"]} ниже пола '
                f'{N_CLOUD_FLOOR} — дизайн невыполним на этих пулах')
        lock = {'n_bins': best['n_bins'], 'n_cloud': best['min_total']}
    progress(f'LOCK: n_bins={lock["n_bins"]}, n_cloud={lock["n_cloud"]}')
    deps['lock'] = lock
    return {'lock': lock, 'ladder_report': ladder_report}


def planned_items():
    items = []
    for fold in range(N_FOLDS):
        for pair in range(NULL_DEV_PER_FOLD):
            items.append({
                'eid': f'null/dev/f{fold}/p{pair}', 'kind': 'null', 'split': 'dev',
                'fold': fold, 'repeat': None, 'pair_idx': pair, 'alpha': 0.0,
                'seeds': (*dev_null_seeds(fold, pair), 0), 'pool_key': ('dev', fold),
            })
    for pair in range(NULL_VAL):
        items.append({
            'eid': f'null/val/x/p{pair}', 'kind': 'null', 'split': 'val',
            'fold': None, 'repeat': None, 'pair_idx': pair, 'alpha': 0.0,
            'seeds': (*val_null_seeds(pair), 0), 'pool_key': ('val', None),
        })
    for pair in range(NULL_TEST):
        items.append({
            'eid': f'null/test/x/p{pair}', 'kind': 'null', 'split': 'test',
            'fold': None, 'repeat': None, 'pair_idx': pair, 'alpha': 0.0,
            'seeds': (*test_null_seeds(pair), 0), 'pool_key': ('test', None),
        })
    for fold in range(N_FOLDS):
        for repeat in range(REPEATS):
            for alpha in ALPHA_GRID:
                items.append({
                    'eid': f'packet/dev/f{fold}/r{repeat}/a{alpha}', 'kind': 'packet',
                    'split': 'dev', 'fold': fold, 'repeat': repeat, 'pair_idx': None,
                    'alpha': alpha, 'seeds': dev_packet_seeds(fold, repeat),
                    'pool_key': ('dev', fold), 'per_base': (fold, repeat),
                })
    for split_name, seeds_fn in (('val', val_packet_seeds), ('test', test_packet_seeds)):
        for repeat in range(REPEATS):
            for alpha in ALPHA_GRID:
                items.append({
                    'eid': f'packet/{split_name}/x/r{repeat}/a{alpha}', 'kind': 'packet',
                    'split': split_name, 'fold': None, 'repeat': repeat,
                    'pair_idx': None, 'alpha': alpha, 'seeds': seeds_fn(repeat),
                    'pool_key': (split_name, None), 'per_base': (None, repeat),
                })
    return items


@section('grid')
def section_grid(deps):
    from tda_metrics.l9m_pools import build_length_matched_packet

    lock = deps['lock']
    pools = deps['pools']
    metrics_obj = deps['metrics_obj']
    os.makedirs(RAW, exist_ok=True)
    done = load_done_ids()
    handle = open(ROWS_PATH, 'a', encoding='utf-8')
    bases = {}
    counter = 0
    t0 = time.perf_counter()
    try:
        for item in planned_items():
            if item.get('once') and item['eid'] in done:
                continue
            if item['eid'] in done:
                continue
            pool_key = item['pool_key']
            base_key = item.get('per_base')
            started = time.perf_counter()
            try:
                if base_key is not None and base_key in bases:
                    pools_obj = bases[base_key]
                else:
                    pools_obj = sample_pools(
                        pools[pool_key], lock['n_cloud'],
                        item['seeds'][0], item['seeds'][1], lock['n_bins'])
                    if pools_obj['n_cloud_actual'] != lock['n_cloud']:
                        raise RuntimeError(
                            f'n_cloud_actual {pools_obj["n_cloud_actual"]} != lock '
                            f'{lock["n_cloud"]} — пилот и прод расходятся')
                    if base_key is not None:
                        bases[base_key] = pools_obj
                packet = build_length_matched_packet(pools_obj, item['alpha'], item['seeds'][2])
                metrics = metrics_of(metrics_obj, packet['P'], packet['Q'])
                row = packet_row(
                    item['eid'], item['kind'], item['split'], item['fold'],
                    item['repeat'], item['pair_idx'], item['alpha'], item['seeds'],
                    pools_obj, packet, metrics, time.perf_counter() - started)
            except Exception as exc:
                row = {
                    'experiment_id': item['eid'], 'kind': item['kind'],
                    'split': item['split'], 'fold': item['fold'],
                    'repeat': item['repeat'], 'pair_idx': item['pair_idx'],
                    'alpha': item['alpha'], 'status': 'failed',
                    'error_message': f'{type(exc).__name__}: {exc}',
                }
            append_row(row, handle)
            counter += 1
            if counter % 50 == 0:
                elapsed = time.perf_counter() - t0
                progress(f'grid: {counter} строк, {elapsed / counter:.1f}s/строка')
    finally:
        handle.close()
    progress(f'grid: {counter} новых строк')


@section('analysis')
def section_analysis(deps):
    import numpy as np
    import pandas as pd
    from scipy.stats import binomtest, ks_2samp
    from sklearn.metrics import roc_auc_score
    from tda_metrics.l9_tasks import evaluate_detection, evaluate_regression

    lock = deps['lock']
    frame = read_rows()
    if frame.empty:
        raise RuntimeError('rows.jsonl пуст')
    null = frame[(frame['kind'] == 'null') & (frame['status'] == 'ok')]
    packet = frame[(frame['kind'] == 'packet') & (frame['status'] == 'ok')]
    failed = frame[frame['status'] == 'failed']
    if null.empty or packet.empty:
        raise RuntimeError('null/packet пусты — analysis невозможен')
    progress(f'строк: null {len(null)}, packet {len(packet)}, failed {len(failed)}')

    def direction_of(metric):
        return 'less' if metric in COVERAGE else 'greater'

    def crossing(values, metric, threshold):
        values = np.asarray(values, dtype=float)
        if direction_of(metric) == 'less':
            return values < threshold
        return values > threshold

    thresholds = []
    for metric in METRICS:
        dev_vals = null[null['split'] == 'dev'][metric].dropna()
        if len(dev_vals) < 100:
            progress(f'{metric}: дев-нулей {len(dev_vals)} < 100 — пропуск')
            continue
        mu, sd = float(dev_vals.mean()), float(dev_vals.std(ddof=1))
        degenerate = sd <= 0.0
        if direction_of(metric) == 'less':
            thr = float(np.quantile(dev_vals, 0.05, method='lower'))
        else:
            thr = float(np.quantile(dev_vals, 0.95, method='higher'))
        row = {'metric': metric, 'direction': direction_of(metric),
               'null_mean': mu, 'null_sd': sd, 'threshold': thr,
               'degenerate': degenerate, 'n_null_dev': int(len(dev_vals))}
        for split_name in ('val', 'test'):
            split_nulls = null[null['split'] == split_name][metric].dropna()
            if len(split_nulls) >= 50:
                fpr = float(np.mean(crossing(split_nulls, metric, thr)))
                k = int(np.sum(crossing(split_nulls, metric, thr)))
                ci = binomtest(k, len(split_nulls), 0.05).proportion_ci(0.95)
                row[f'fpr_{split_name}'] = fpr
                row[f'fpr_{split_name}_ci_low'] = float(ci.low)
                row[f'fpr_{split_name}_ci_high'] = float(ci.high)
                row[f'n_null_{split_name}'] = int(len(split_nulls))
        thresholds.append(row)
    thr_frame = pd.DataFrame(thresholds).set_index('metric')
    progress(f'порогов: {len(thr_frame)} (degenerate: '
             f'{int(thr_frame["degenerate"].sum())})')

    power_rows = []
    for split_name in ('dev', 'val', 'test'):
        block = packet[packet['split'] == split_name]
        for alpha in ALPHA_GRID:
            for metric in thr_frame.index:
                vals = block[block['alpha'] == alpha][metric].dropna()
                if vals.empty:
                    continue
                mu = float(thr_frame.loc[metric, 'null_mean'])
                sd = float(thr_frame.loc[metric, 'null_sd'])
                thr = float(thr_frame.loc[metric, 'threshold'])
                sign = -1.0 if direction_of(metric) == 'less' else 1.0
                z_obs = (vals - mu) * sign / (sd + 1e-12)
                power_rows.append({
                    'split': split_name, 'alpha': alpha, 'metric': metric,
                    'mean': float(vals.mean()), 'sd': float(vals.std(ddof=1)),
                    'mean_z': float(z_obs.mean()), 'power': float(
                        np.mean(crossing(vals, metric, thr))),
                    'n_reps': int(len(vals)), 'degenerate': bool(
                        thr_frame.loc[metric, 'degenerate']),
                })
    power = pd.DataFrame(power_rows)

    min_alpha = {}
    for split_name in ('val', 'test'):
        for metric in thr_frame.index:
            block = power[(power['split'] == split_name) & (power['metric'] == metric)]
            block = block[~block['degenerate']]
            hit = block[block['power'] >= 0.8].sort_values('alpha')
            min_alpha[(split_name, metric)] = (
                float(hit['alpha'].iloc[0]) if len(hit) else None)

    dev_packets = packet[packet['split'] == 'dev'].copy()
    y_alpha = (dev_packets['alpha'] > 0).astype(int).to_numpy()
    q_len_auroc = float(roc_auc_score(y_alpha, dev_packets['q_len_mean'].to_numpy()))
    per_alpha_len = dev_packets.groupby('alpha').agg(
        q_len_mean=('q_len_mean', 'mean'), p_len_mean=('p_len_mean', 'mean')).reset_index()
    progress(f'q_len_mean AUROC(alpha>0) на dev-пакетах: {q_len_auroc:.3f}')

    ks_rows = []
    for _, row in packet.iterrows():
        ks = ks_2samp(row['p_lengths'], row['q_lengths'])
        ks_rows.append({'experiment_id': row['experiment_id'], 'split': row['split'],
                        'alpha': row['alpha'], 'ks_stat': float(ks.statistic),
                        'ks_p': float(ks.pvalue), 'smd_pq': float(row['smd_pq'])})
    ks_frame = pd.DataFrame(ks_rows)
    length_balance = ks_frame.groupby('split').agg(
        n_rows=('ks_stat', 'size'), smd_mean=('smd_pq', 'mean'),
        smd_max_abs=('smd_pq', lambda s: float(np.max(np.abs(s)))),
        ks_stat_mean=('ks_stat', 'mean'), ks_stat_max=('ks_stat', 'max'),
        ks_p_median=('ks_p', 'median')).reset_index()
    smd_max_abs = float(np.max(np.abs(ks_frame['smd_pq'])))
    progress(f'length balance: max|SMD|={smd_max_abs:.4f}, '
             f'KS mean={ks_frame["ks_stat"].mean():.4f}')

    packet_table = packet[packet['split'] == 'dev'][
        ['alpha', 'fold', *METRICS]].copy()
    oof = {'detection': {}, 'regression': {},
           'single_feature': {}, 'small_alpha': {}}
    oof['detection'] = evaluate_detection(packet_table, feature_groups=GROUPS_OOF)
    oof['regression'] = evaluate_regression(packet_table, feature_groups=GROUPS_OOF)
    for metric in METRICS:
        single = evaluate_detection(packet_table, feature_groups={metric: [metric]})
        oof['single_feature'][metric] = single[metric]
    small = packet_table[packet_table['alpha'] <= 0.1 + 1e-9].copy()
    oof['small_alpha']['detection'] = evaluate_detection(small, feature_groups=GROUPS_OOF)
    oof['small_alpha']['regression'] = evaluate_regression(small, feature_groups=GROUPS_OOF)
    oof['small_alpha']['n_rows'] = int(len(small))

    analysis = {
        'thresholds': thr_frame.reset_index().to_dict('records'),
        'power': power,
        'min_alpha': {f'{split}/{metric}': value
                      for (split, metric), value in min_alpha.items()},
        'q_len_auroc': q_len_auroc,
        'per_alpha_len': per_alpha_len,
        'length_balance': length_balance,
        'ks_frame': ks_frame,
        'smd_max_abs': smd_max_abs,
        'oof': oof,
        'failed_rows': int(len(failed)),
    }
    deps['analysis'] = analysis
    return analysis


@section('figures')
def section_figures(deps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    analysis = deps.get('analysis')
    if analysis is None:
        raise RuntimeError('analysis отсутствует')
    os.makedirs(FIGURES, exist_ok=True)
    thr = {row['metric']: row for row in analysis['thresholds']}
    power = analysis['power']
    metrics = [row['metric'] for row in analysis['thresholds']]

    dev_power = power[power['split'] == 'dev']
    n_rows, n_cols = 4, 4
    figure, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12))
    for position, metric in enumerate(metrics):
        axis = axes[position // n_cols][position % n_cols]
        block = dev_power[dev_power['metric'] == metric].sort_values('alpha')
        axis.errorbar(block['alpha'], block['mean'], yerr=block['sd'], fmt='o-',
                      color='tab:blue', capsize=3, label='dev packets')
        spec = thr[metric]
        axis.axhspan(spec['null_mean'] - spec['null_sd'],
                     spec['null_mean'] + spec['null_sd'], color='gray', alpha=0.25)
        axis.axhline(spec['threshold'], color='red', linestyle='--', linewidth=1,
                     label='threshold')
        axis.set_title(f"{metric} [{spec['direction']}]", fontsize=9)
        axis.set_xlabel('alpha', fontsize=8)
        axis.tick_params(labelsize=7)
    for position in range(len(metrics), n_rows * n_cols):
        axes[position // n_cols][position % n_cols].axis('off')
    axes[0][0].legend(fontsize=7)
    figure.suptitle('L9M: dose response, dev-пакеты (mean±sd, 5 фолдов x 4 повтора), '
                    'заливка — dev-null band')
    figure.tight_layout()
    figure.savefig(os.path.join(FIGURES, 'l9m_dose_response.png'), dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12))
    for position, metric in enumerate(metrics):
        axis = axes[position // n_cols][position % n_cols]
        for split_name, style, color in (('val', 'o-', 'tab:orange'),
                                         ('test', 's--', 'tab:green')):
            block = power[(power['split'] == split_name)
                          & (power['metric'] == metric)].sort_values('alpha')
            if not block.empty:
                axis.plot(block['alpha'], block['power'], style, color=color,
                          label=split_name, markersize=4)
        axis.axhline(0.8, color='red', linestyle=':', linewidth=1)
        axis.set_title(metric, fontsize=9)
        axis.set_ylim(-0.05, 1.05)
        axis.set_xlabel('alpha', fontsize=8)
        axis.tick_params(labelsize=7)
    for position in range(len(metrics), n_rows * n_cols):
        axes[position // n_cols][position % n_cols].axis('off')
    axes[0][0].legend(fontsize=7)
    figure.suptitle('L9M: power по alpha при locked threshold (dev-null, FPR 0.05)')
    figure.tight_layout()
    figure.savefig(os.path.join(FIGURES, 'l9m_power_by_alpha.png'), dpi=150)
    plt.close(figure)

    frame = read_rows()
    packet = frame[(frame['kind'] == 'packet') & (frame['status'] == 'ok')]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    p_len = np.concatenate(packet[packet['alpha'] == 0.0]['p_lengths'].to_numpy())
    q_correct_len = np.concatenate(packet[packet['alpha'] == 0.0]['q_lengths'].to_numpy())
    h_len = np.concatenate(packet[packet['alpha'] == 1.0]['q_lengths'].to_numpy())
    for values, label, color in ((p_len, 'P (correct, half A)', 'tab:blue'),
                                 (q_correct_len, 'Q_correct (half B)', 'tab:cyan'),
                                 (h_len, 'H (hallucinated, half B)', 'tab:red')):
        axes[0].hist(values, bins=30, alpha=0.5, label=label, color=color, density=True)
    axes[0].set_title('Распределения длин n_pooled')
    axes[0].set_xlabel('n_pooled')
    axes[0].legend(fontsize=8)
    for split_name, color in (('dev', 'tab:blue'), ('val', 'tab:orange'),
                              ('test', 'tab:green')):
        block = packet[packet['split'] == split_name]
        axes[1].scatter(block['alpha'], block['smd_pq'], s=12, alpha=0.6,
                        color=color, label=split_name)
    axes[1].axhline(SMD_TOLERANCE, color='red', linestyle='--', linewidth=1)
    axes[1].axhline(-SMD_TOLERANCE, color='red', linestyle='--', linewidth=1)
    axes[1].set_title(f'SMD длин P vs Q_alpha по строкам (tolerance ±{SMD_TOLERANCE})')
    axes[1].set_xlabel('alpha')
    axes[1].set_ylabel('SMD')
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(os.path.join(FIGURES, 'l9m_length_balance.png'), dpi=150)
    plt.close(figure)
    progress('figures записаны')


@section('tables')
def section_tables(deps):
    import pandas as pd
    analysis = deps.get('analysis')
    if analysis is None:
        raise RuntimeError('analysis отсутствует')
    os.makedirs(TABLES, exist_ok=True)

    thr_power = pd.DataFrame(analysis['thresholds'])
    power = analysis['power']
    for split_name in ('val', 'test'):
        min_col = []
        for metric in thr_power['metric']:
            min_col.append(analysis['min_alpha'].get(f'{split_name}/{metric}'))
        thr_power[f'min_alpha_power08_{split_name}'] = min_col
    thr_power.to_csv(os.path.join(TABLES, 'l9m_thresholds_power.csv'), index=False)

    dose = power[power['split'] == 'dev'][
        ['metric', 'alpha', 'mean', 'sd', 'mean_z', 'power', 'n_reps', 'degenerate']]
    dose.to_csv(os.path.join(TABLES, 'l9m_dose_response.csv'), index=False)

    single_rows = []
    for metric, entry in analysis['oof']['single_feature'].items():
        single_rows.append({'metric': metric, **entry})
    for group, entry in analysis['oof']['detection'].items():
        single_rows.append({'metric': f'GROUP:{group}', **entry})
    pd.DataFrame(single_rows).to_csv(
        os.path.join(TABLES, 'l9m_single_feature.csv'), index=False)

    analysis['length_balance'].to_csv(
        os.path.join(TABLES, 'l9m_length_balance.csv'), index=False)

    frame = read_rows()
    frame.to_parquet(os.path.join(RAW, 'results.parquet'))
    packets_flat = frame[(frame['kind'] == 'packet') & (frame['status'] == 'ok')][
        ['experiment_id', 'split', 'fold', 'repeat', 'alpha', 'n_hall',
         'n_cloud_actual', 'n_bins_actual', 'bin_quotas', 'q_len_mean',
         'p_len_mean', 'smd_pq', 'composition_hash', *METRICS]]
    packets_flat.to_parquet(os.path.join(RAW, 'packet_table.parquet'))
    progress('tables + parquet записаны')


@section('verdict')
def section_verdict(deps):
    analysis = deps.get('analysis')
    lock = deps.get('lock') or {}
    if analysis is None:
        raise RuntimeError('analysis отсутствует')
    smd_ok = analysis['smd_max_abs'] < SMD_TOLERANCE
    q_len_ok = analysis['q_len_auroc'] <= Q_LEN_AUROC_MAX
    n_cloud_ok = True
    frame = read_rows()
    ok_rows = frame[frame['status'] == 'ok']
    if 'n_cloud_actual' in ok_rows and len(ok_rows):
        n_cloud_ok = bool((ok_rows['n_cloud_actual'] == lock.get('n_cloud')).all())
    checks = {
        'length_smd_within_tolerance': {
            'value': analysis['smd_max_abs'], 'tolerance': SMD_TOLERANCE,
            'pass': bool(smd_ok)},
        'q_len_mean_not_separating': {
            'auroc': analysis['q_len_auroc'], 'max': Q_LEN_AUROC_MAX,
            'pass': bool(q_len_ok)},
        'n_cloud_constant': {'pass': n_cloud_ok},
        'bin_quotas_equal_across_sides': {
            'pass': True,
            'note': 'гарантировано assert-ами sample_length_matched_pools'},
        'failed_rows': analysis['failed_rows'],
    }
    status = 'VALID' if (smd_ok and q_len_ok and n_cloud_ok) else 'INVALID_LENGTH_CONTROL'
    if status == 'INVALID_LENGTH_CONTROL':
        progress('ВНИМАНИЕ: серия INVALID по length-контролю — метрики-результаты '
                 'не публикуются как headline')
    verdict = {
        'created': datetime.now(timezone.utc).isoformat(),
        'status': status,
        'checks': checks,
        'locked': lock,
        'min_alpha_power08_test': {
            metric.split('/', 1)[1]: value for metric, value in
            analysis['min_alpha'].items() if metric.startswith('test/')},
        'fpr': [
            {key: row.get(key) for key in (
                'metric', 'fpr_val', 'fpr_val_ci_low', 'fpr_val_ci_high',
                'fpr_test', 'fpr_test_ci_low', 'fpr_test_ci_high', 'degenerate')}
            for row in analysis['thresholds']],
        'headline_allowed': (
            'length-matched dose response on HaluEval candidate-answer '
            'representations (без claims о детекторе собственных галлюцинаций Qwen)'),
    }
    with open(os.path.join(RAW, 'verdict.json'), 'w', encoding='utf-8') as handle:
        json.dump(json_safe(verdict), handle, ensure_ascii=False, indent=2)
    progress(f'verdict: {status}')
    return verdict


@section('summary')
def section_summary(deps):
    analysis = deps.get('analysis')
    lock = deps.get('lock') or {}
    state = deps.get('state') or {}
    pilot = deps.get('pilot') or {}
    summary = {
        'created': datetime.now(timezone.utc).isoformat(),
        'sections': SECTION_STATUS,
        'locked': lock,
        'pilot_ladder': pilot.get('ladder_report'),
        'primary_endpoint': {
            'metric': PRIMARY,
            'thresholds_power': [
                {**{key: row.get(key) for key in (
                    'metric', 'direction', 'threshold', 'fpr_val',
                    'fpr_val_ci_low', 'fpr_val_ci_high', 'fpr_test',
                    'degenerate')},
                 'min_alpha_power08_val': analysis['min_alpha'].get(
                     f"val/{row.get('metric')}"),
                 'min_alpha_power08_test': analysis['min_alpha'].get(
                     f"test/{row.get('metric')}")}
                for row in analysis['thresholds']],
            'min_alpha_test': (
                {m.split('/', 1)[1]: v for m, v in analysis['min_alpha'].items()
                 if m.startswith('test/')} if analysis else None),
        },
        'invalidation': {
            'q_len_auroc': analysis['q_len_auroc'] if analysis else None,
            'smd_max_abs': analysis['smd_max_abs'] if analysis else None,
            'per_alpha_len': (
                analysis['per_alpha_len'].to_dict('records') if analysis else None),
        },
        'secondary_oof': {
            'detection': analysis['oof']['detection'] if analysis else None,
            'regression': analysis['oof']['regression'] if analysis else None,
            'single_feature': {
                metric: entry['auroc'] for metric, entry in
                analysis['oof']['single_feature'].items()} if analysis else None,
            'small_alpha': analysis['oof']['small_alpha'] if analysis else None,
        },
        'failed_rows': analysis['failed_rows'] if analysis else None,
        'split_hashes': state.get('split_hashes'),
        'cache_manifest_keys': {
            key: (state.get('manifest') or {}).get(key) for key in
            ('model', 'input_format', 'pooling', 'layer_indices')},
    }
    manifest = {
        'created': summary['created'],
        'protocol_path': PROTOCOL_PATH,
        'protocol_sha256': _sha256_file(PROTOCOL_PATH),
        'sections': SECTION_STATUS,
        'grids': {
            'alpha_grid': ALPHA_GRID, 'n_folds': N_FOLDS, 'repeats': REPEATS,
            'null_dev_per_fold': NULL_DEV_PER_FOLD, 'null_val': NULL_VAL,
            'null_test': NULL_TEST, 'n_cloud': lock.get('n_cloud'),
            'n_bins_locked': lock.get('n_bins'), 'layer': LAYER,
            'pca_dim': PCA_DIM, 'pca_fit': 'dev-correct, весь сплит',
            'rtd': 'исключён из батареи (отклонение, см. протокол)',
        },
        'split_hashes': state.get('split_hashes'),
        'runtime_seconds': time.perf_counter() - _STARTED,
    }
    with open(os.path.join(RAW, 'summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(json_safe(summary), handle, ensure_ascii=False, indent=2)
    with open(os.path.join(RAW, 'manifest.json'), 'w', encoding='utf-8') as handle:
        json.dump(json_safe(manifest), handle, ensure_ascii=False, indent=2)
    progress('summary/manifest записаны')


def main():
    deps = {}
    progress(f'L9M start: protocol_exists={os.path.exists(PROTOCOL_PATH)}')
    try:
        import numpy as np
        from tda_metrics.metrics import TopologyMetrics
        globals()['np'] = np
        deps['metrics_obj'] = TopologyMetrics(seed=0)
    except Exception:
        progress('HEAVY IMPORT FAIL:\n' + traceback.format_exc(limit=3))
        print('L9M DONE rc=1')
        return
    section_env(None)
    section_pytest(None)
    try:
        deps['state'] = load_state()
    except Exception:
        progress('LOAD FAIL:\n' + traceback.format_exc(limit=3))
        print('L9M DONE rc=1')
        return
    pilot = section_pilot(deps)
    deps['pilot'] = pilot
    if pilot is None:
        print('L9M DONE rc=1')
        return
    section_grid(deps)
    analysis = section_analysis(deps)
    deps['analysis'] = analysis
    if analysis is None:
        section_summary(deps)
        print('L9M DONE rc=1')
        return
    section_figures(deps)
    section_tables(deps)
    section_verdict(deps)
    section_summary(deps)
    ok = SECTION_STATUS.get('grid') == 'ok' and SECTION_STATUS.get('analysis') == 'ok'
    extras_ok = all(SECTION_STATUS.get(name) == 'ok' for name in
                    ('figures', 'tables', 'verdict', 'summary'))
    print('L9M DONE rc=' + ('0' if (ok and extras_ok) else '1'))


if __name__ == '__main__':
    os.makedirs('logs', exist_ok=True)
    os.makedirs(RAW, exist_ok=True)
    sys.stdout = Tee(LOG_PATH)
    main()
