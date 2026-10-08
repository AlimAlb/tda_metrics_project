"""Послойная LLM-серия 5.4 с нуля: L3/L3d/L3e по слоям + RTD/CKA layerwise.

Трекаемый перенос семантик прогонов wave3 (режим A, logs/orchestration_wave2/
wave3_llm_run.py) и wave5 (wave5_layerwise_run.py) в standalone-цепочку:
- l3_series: l3_direct / l3d_length_matched / l3e_shuffled_label на слоях
  {9, 18, 27, 36}, 5 повторов, n=500, PCA-16 (fit на dev-correct послойно),
  compute_all с rtd_trials=2; семейства l4a/l4b НЕ входят (это серии 5.3 и
  paired-сайдкар) — только три семейства послойной динамики 5.4;
- l5_layerwise: cross (correct vs hallucinated) и identity (same-law пол)
  RTD + 1-CKA, n=1000, 5 повторов, сиды подвыборок 200+k (cross),
  200+k/400+k (identity-пара) — как в историческом прогоне;
- сиды волн сохранены: прямая сопоставимость с results/raw/llm и
  results/raw/wave5; новый расчёт пишется в results/raw/layerwise_rerun и
  исторические таблицы не перезаписывает.

Resume: l3_series — через ResultStore (ключ experiment_id), l5_layerwise —
по experiment_id в parquet. SystemExit не используется.
"""
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

RAW = 'results/raw/layerwise_rerun'
STORE_L3 = os.path.join(RAW, 'l3_series')
L5_PARQUET = os.path.join(RAW, 'layerwise.parquet')
TABLES = 'results/tables'
LOG_PATH = 'logs/layerwise_rerun.log'
CACHE_DIR = 'embeddings/l3r_cache'
EXTRACT_SCRIPT = 'scripts/extract_llm_embeddings.py'

LAYERS = (9, 18, 27, 36)
REPEATS = 5
N_CLOUD_L3 = 500
N_CLOUD_L5 = 1000

METRIC_PARAMS = json.dumps({
    'nhood_sizes': [1, 3, 10], 'ntd_repeats': 5, 'rtd_batch': 500,
    'rtd_trials': 2, 'rtd_mode': 'random-coupling-exploratory', 'js_k': 5,
}, sort_keys=True, separators=(',', ':'))

L3_FAMILIES = ('l3_direct', 'l3d_length_matched', 'l3e_shuffled_label')

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


@section('env')
def section_env(_):
    import numpy
    versions = {'python': sys.version.split()[0], 'numpy': numpy.__version__}
    for name in ('pandas', 'scipy', 'torch', 'pyarrow'):
        try:
            module = __import__(name)
            versions[name] = getattr(module, '__version__', '?')
        except Exception:
            versions[name] = None
    progress(f'versions: {versions}')
    return {'versions': versions}


@section('extract')
def section_extract(_):
    marker = os.path.join(CACHE_DIR, 'vectors.npy')
    if os.path.exists(marker):
        progress(f'кэш {CACHE_DIR} существует — извлечение пропущено')
        return
    proc = subprocess.run(
        [sys.executable, EXTRACT_SCRIPT, '--out-dir', CACHE_DIR],
        capture_output=True, text=True,
    )
    print(proc.stdout[-2000:])
    if proc.returncode != 0 or not os.path.exists(marker):
        raise RuntimeError(f'extract rc={proc.returncode}')
    progress('извлечение эмбеддингов завершено')


@section('pytest_light')
def section_pytest(_):
    proc = subprocess.run(
        [sys.executable, '-m', 'pytest', '-q', '--tb=short',
         'tests/test_llm_experiments.py', 'tests/test_experiment_schema.py'],
        capture_output=True, text=True, timeout=1200,
    )
    print(proc.stdout[-1200:])
    if proc.returncode != 0:
        raise RuntimeError(f'pytest rc={proc.returncode}')
    progress('pytest-гейт пройден')


def load_state():
    from tda_metrics.embedding_cache import load_cache
    from tda_metrics.llm_experiments import layer_clouds, prompt_id_split, prompt_ids_hash
    from tda_metrics.reduction import pca_fit
    meta, vectors, manifest = load_cache(CACHE_DIR, expect={
        'model': 'Qwen/Qwen2.5-3B-Instruct',
        'pooling': 'mean_answer',
        'input_format': 'chat_knowledge_v1',
    })
    split = prompt_id_split(meta, seed=0)
    pca_models = {}
    pools = {}
    for layer in LAYERS:
        correct = layer_clouds(meta, vectors, split['dev'], 'correct', layer)
        hallucinated = layer_clouds(meta, vectors, split['dev'], 'hallucinated', layer)
        pca_models[layer] = pca_fit(correct, n_components=16)
        pools[(layer, 'correct')] = correct
        pools[(layer, 'hallucinated')] = hallucinated
        progress(f'слой {layer}: dev-correct {correct.shape}, dev-hall {hallucinated.shape}')
    return {
        'meta': meta, 'vectors': vectors, 'manifest': manifest, 'split': split,
        'dev_hash': prompt_ids_hash(split['dev']), 'pca_models': pca_models,
        'pools': pools,
    }


def _subsample_seed(matrix, seed, n):
    import numpy as np
    return matrix[np.random.default_rng(seed).choice(len(matrix), size=n, replace=False)]


def _projected(state, layer, subset):
    from tda_metrics.reduction import pca_transform
    return pca_transform(subset, model=state['pca_models'][layer])[0]


def l3_clouds(state, family, layer, k):
    import numpy as np
    from tda_metrics.llm_experiments import length_matched_pair, shuffled_label_pair
    pools = state['pools']
    if family == 'l3_direct':
        p = _projected(state, layer, _subsample_seed(pools[(layer, 'correct')], 200 + k, N_CLOUD_L3))
        q = _projected(state, layer, _subsample_seed(pools[(layer, 'hallucinated')], 200 + k, N_CLOUD_L3))
        return p, q
    if family == 'l3d_length_matched':
        c_m, h_m, _, _ = length_matched_pair(
            state['meta'], state['vectors'], state['split']['dev'], layer,
            n_max=N_CLOUD_L3)
        return (_projected(state, layer, c_m), _projected(state, layer, h_m))
    if family == 'l3e_shuffled_label':
        c_sub = _subsample_seed(pools[(layer, 'correct')], 200 + k, N_CLOUD_L3)
        h_sub = _subsample_seed(pools[(layer, 'hallucinated')], 200 + k, N_CLOUD_L3)
        joint = np.vstack([c_sub, h_sub])
        labels = np.array([0] * len(c_sub) + [1] * len(h_sub))
        p_raw, q_raw = shuffled_label_pair(joint, labels, seed=k)
        joint_proj = _projected(state, layer, np.vstack([p_raw, q_raw]))
        return joint_proj[:len(p_raw)], joint_proj[len(p_raw):]
    raise ValueError(f'семейство {family} не входит в серию 5.4')


@section('l3_series')
def section_l3(state):
    from dataclasses import replace
    from tda_metrics import TopologyMetrics
    from tda_metrics.experiment_schema import ResultStore, run_grid
    from tda_metrics.llm_experiments import build_l3_configs, prompt_ids_hash

    os.makedirs(STORE_L3, exist_ok=True)
    metrics = TopologyMetrics(seed=42, device='cuda')

    def runner(config):
        parsed_family = config.experiment_family
        layer = config.layer
        k = config.seed
        p, q = l3_clouds(state, parsed_family, layer, k)
        return metrics.compute_all(
            p, q, nhood_sizes=(1, 3, 10), js_k=5, rtd_trials=2, rtd_batch=500)

    store = ResultStore(STORE_L3)
    configs = [
        replace(c, split_id='dev',
                prompt_ids_hash=prompt_ids_hash(state['split']['dev']), pca_dim=16)
        for c in build_l3_configs(LAYERS, (0.1, 0.25, 0.5, 0.75, 1.0), REPEATS,
                                   n_cloud=N_CLOUD_L3)
        if c.experiment_family in L3_FAMILIES
    ]
    progress(f'l3_series: {len(configs)} конфигов ({len(store)} строк уже в store)')
    written = run_grid(store, configs, runner, metric_params=METRIC_PARAMS)
    frame = store.load_frame()
    ok = frame[frame['status'] == 'ok']
    failed = frame[frame['status'] == 'failed']
    progress(f'l3_series: ok={len(ok)}, failed={len(failed)}')
    if not failed.empty:
        for _, row in failed.iterrows():
            progress(f"  failed: {row['experiment_id']}: {row['error_message']}")
    return {'written': len(written), 'ok': len(ok), 'failed': len(failed)}


@section('l5_layerwise')
def section_l5(state):
    import numpy as np
    import pandas as pd
    from tda_metrics import TopologyMetrics
    from tda_metrics.llm_experiments import linear_cka

    os.makedirs(RAW, exist_ok=True)
    done = set()
    existing = pd.DataFrame()
    if os.path.exists(L5_PARQUET):
        try:
            existing = pd.read_parquet(L5_PARQUET)
            done = set(existing['experiment_id'])
        except Exception as exc:
            progress(f'resume не удался ({type(exc).__name__}: {exc}) — с нуля')
            existing = pd.DataFrame()
    metrics = TopologyMetrics(seed=42, device='cuda')
    pools = state['pools']
    new_rows = []
    total = len(LAYERS) * REPEATS * 2
    started = time.perf_counter()
    for layer in LAYERS:
        correct = pools[(layer, 'correct')]
        hallucinated = pools[(layer, 'hallucinated')]
        for k in range(REPEATS):
            for comparison in ('cross', 'identity'):
                eid = f'l5/{layer}/{comparison}/r{k}'
                if eid in done:
                    continue
                if comparison == 'cross':
                    p = _projected(state, layer,
                                   _subsample_seed(correct, 200 + k, N_CLOUD_L5))
                    q = _projected(state, layer,
                                   _subsample_seed(hallucinated, 200 + k, N_CLOUD_L5))
                else:
                    p = _projected(state, layer,
                                   _subsample_seed(correct, 200 + k, N_CLOUD_L5))
                    q = _projected(state, layer,
                                   _subsample_seed(correct, 400 + k, N_CLOUD_L5))
                rtd_value = metrics.rtd(p, q, trials=2, batch=500)
                cka_distance = 1.0 - linear_cka(p, q)
                new_rows.append({
                    'experiment_id': eid, 'layer': layer, 'repeat': k,
                    'comparison': comparison,
                    'rtd': float(rtd_value), 'cka_distance': float(cka_distance),
                })
                count = len(new_rows)
                if count % 5 == 0 or len(done) + count == total:
                    elapsed = time.perf_counter() - started
                    speed = count / max(elapsed, 1e-9)
                    progress(f'l5: {len(done) + count}/{total} | '
                             f'{speed * 60:.1f} строк/мин | ETA '
                             f'{(total - len(done) - count) / max(speed, 1e-9) / 60:.0f} мин')
    if new_rows:
        frame = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
        frame.to_parquet(L5_PARQUET, index=False)
    progress(f'l5_layerwise: всего {len(done) + len(new_rows)} строк')
    return {'total': len(done) + len(new_rows), 'new': len(new_rows)}


@section('analysis')
def section_analysis(state):
    import pandas as pd
    from tda_metrics.experiment_schema import ResultStore

    os.makedirs(TABLES, exist_ok=True)
    store = ResultStore(STORE_L3)
    frame = store.load_frame()
    ok = frame[frame['status'] == 'ok']
    metrics_cols = [c for c in (
        'mtd_PQ', 'mtd_QP', 'ntd_PQ', 'ntd_QP', 'rtd',
        'precision@3', 'recall@3', 'mmd', 'frechet', 'js') if c in ok.columns]
    rows = []
    for family in L3_FAMILIES:
        block = ok[ok['experiment_family'] == family]
        for layer in LAYERS:
            sub = block[block['layer'] == layer]
            if sub.empty:
                continue
            for metric in metrics_cols:
                vals = sub[metric].dropna()
                rows.append({
                    'family': family, 'layer': layer, 'metric': metric,
                    'mean': float(vals.mean()) if len(vals) else None,
                    'sd': float(vals.std(ddof=1)) if len(vals) > 1 else None,
                    'n_repeats': int(len(vals)),
                })
    l3_table = pd.DataFrame(rows)
    l3_table.to_csv(os.path.join(TABLES, 'layerwise_rerun_l3_by_layer.csv'), index=False)

    l5 = pd.read_parquet(L5_PARQUET)
    by_layer = {}
    for layer in LAYERS:
        by_layer[str(layer)] = {}
        for comparison in ('cross', 'identity'):
            block = l5[(l5['layer'] == layer) & (l5['comparison'] == comparison)]
            by_layer[str(layer)][comparison] = {
                'rtd': {'mean': float(block['rtd'].mean()),
                        'sd': float(block['rtd'].std(ddof=1)),
                        'n_repeats': int(len(block))},
                'cka_distance': {'mean': float(block['cka_distance'].mean()),
                                 'sd': float(block['cka_distance'].std(ddof=1)),
                                 'n_repeats': int(len(block))},
            }
    progress(f'analysis: {len(l3_table)} строк l3-таблицы; l5 слои {sorted(l5["layer"].unique())}')
    return {'l3_rows': len(l3_table), 'by_layer': by_layer}


@section('summary')
def section_summary(state):
    from tda_metrics.llm_experiments import prompt_ids_hash
    l3 = state.get('l3') or {}
    l5 = state.get('l5') or {}
    analysis = state.get('analysis') or {}
    summary = {
        'created': datetime.now(timezone.utc).isoformat(),
        'sections': SECTION_STATUS,
        'l3_series': l3,
        'l5_layerwise': l5,
        'l5_by_layer': analysis.get('by_layer'),
        'split': {
            'rule': 'prompt_id_split(meta, seed=0, 0.6/0.2/0.2)',
            'dev_prompt_ids_hash': state['dev_hash'],
        },
    }
    manifest = {
        'created': summary['created'],
        'sections': SECTION_STATUS,
        'provenance': {
            'l3_series': 'перенос wave3 режим A (logs/orchestration_wave2/wave3_llm_run.py): '
                         'сид подвыборок 200+k, l3d — детерминированный length_matched_pair '
                         '(n_max=500), l3e — shuffled_label_pair(seed=k); rtd_trials=2',
            'l5_layerwise': 'перенос wave5_layerwise_run.py: cross сиды 200+k оба пула, '
                            'identity 200+k/400+k по correct; RTD trials=2 batch=500; '
                            'cka_distance = 1 - linear_cka',
            'scope': 'семейства l4a/l4b исключены (серии 5.3/paired); серия 5.4 = '
                     'l3/l3d/l3e + l5 layerwise',
        },
        'embedding_cache': {
            'model': state['manifest'].get('model'),
            'input_format': state['manifest'].get('input_format'),
            'pooling': state['manifest'].get('pooling'),
            'layer_indices': state['manifest'].get('layer_indices'),
            'n_items': state['manifest'].get('n_items'),
        },
        'metric_params': METRIC_PARAMS,
        'grids': {
            'layers': list(LAYERS), 'repeats': REPEATS,
            'n_cloud_l3': N_CLOUD_L3, 'n_cloud_l5': N_CLOUD_L5,
        },
        'outputs': {
            'l3_store': STORE_L3, 'l5_parquet': L5_PARQUET,
        },
        'runtime_seconds': time.perf_counter() - _STARTED,
    }
    with open(os.path.join(RAW, 'summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(json_safe(summary), handle, ensure_ascii=False, indent=2)
    with open(os.path.join(RAW, 'manifest.json'), 'w', encoding='utf-8') as handle:
        json.dump(json_safe(manifest), handle, ensure_ascii=False, indent=2)
    progress('summary/manifest записаны')


def main():
    progress('LAYERWISE RERUN start')
    try:
        import numpy  # noqa: F401
        from tda_metrics import TopologyMetrics  # noqa: F401
    except Exception:
        progress('HEAVY IMPORT FAIL:\n' + traceback.format_exc(limit=3))
        print('LAYERWISE DONE rc=1')
        return
    section_env(None)
    section_extract(None)
    section_pytest(None)
    try:
        state = load_state()
    except Exception:
        progress('LOAD FAIL:\n' + traceback.format_exc(limit=3))
        print('LAYERWISE DONE rc=1')
        return
    l3 = section_l3(state)
    state['l3'] = l3 or {}
    l5 = section_l5(state)
    state['l5'] = l5 or {}
    analysis = section_analysis(state)
    state['analysis'] = analysis or {}
    section_summary(state)
    ok = SECTION_STATUS.get('l3_series') == 'ok' and SECTION_STATUS.get('l5_layerwise') == 'ok'
    print('LAYERWISE DONE rc=' + ('0' if ok else '1'))


if __name__ == '__main__':
    os.makedirs('logs', exist_ok=True)
    os.makedirs(RAW, exist_ok=True)
    sys.stdout = Tee(LOG_PATH)
    main()
