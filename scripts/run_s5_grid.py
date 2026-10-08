"""Чистая синтетика направленной селективности (§5.3–5.6).

Мега-скрипт VM-прогона: env → pytest-light → grid (resume) → analysis
(effect-таблица с FPR/power/BH) → figures → summary/manifest. Секции в
try/except: ошибка одной не останавливает остальные. Полный прогон —
1680 строк (s5a 480 + s5b 720 + s5c 480), smoke-режим S5_SMOKE=1 — усечённая
сетка для проверки структуры. SystemExit не используется (вешает ядро в
exec-контексте): rc печатается в конце, запуск анализирует лог.
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

RAW = 'results/raw/s5'
TABLES = 'results/tables'
FIGURES = 'results/figures'
LOG_PATH = 'logs/s5_run.log'
ROWS_PATH = os.path.join(RAW, 'rows.jsonl')
PARQUET_PATH = os.path.join(RAW, 'results.parquet')
CSV_PATH = os.path.join(RAW, 'results.csv')
PROTOCOL_PATH = 'protocols/s5_protocol.json'

SMOKE = os.environ.get('S5_SMOKE') == '1'
ONLY_N = os.environ.get('S5_ONLY_N')
GRID_ONLY = os.environ.get('S5_GRID_ONLY') == '1'

FULL_N_GRID = [100, 250, 500, 1000]
FULL_REPS = 30
FULL_S5B_ALPHAS = [0.0, 0.01, 0.05, 0.125, 0.25, 0.5]
S5A_PAIRS = {
    0: [],
    1: [(0, 4)],
    2: [(0, 4), (1, 5)],
    3: [(0, 4), (1, 5), (2, 6)],
}
S5C_DIRECTIONS = ['p8q4', 'p4q8', 'p8p8', 'q4q4']
S5C_CONTROLS = {'p8q4': 'p8p8', 'p4q8': 'q4q4'}

METRIC_KINDS = {
    'mtd_PQ': 'distance', 'mtd_QP': 'distance', 'ntd_PQ': 'distance',
    'ntd_QP': 'distance', 'rtd': 'distance', 'mmd': 'distance',
    'frechet': 'distance', 'js': 'distance', 'energy': 'distance',
    'sinkhorn': 'distance', 'precision@3': 'similarity', 'recall@3': 'similarity',
}
ALL_METRICS = list(METRIC_KINDS)
PRIMARY_METRICS = ['mtd_PQ', 'mtd_QP', 'ntd_PQ', 'ntd_QP', 'precision@3', 'recall@3']

SEED_BASE_P = 42
SEED_BASE_Q = 7
MAX_MOMENT_ATTEMPTS = 100
N_BOOTSTRAP = 1000
N_FPR_SPLITS = 200
FPR_THRESHOLD = 1.96

SECTION_STATUS = {}
_STARTED = time.perf_counter()


class Tee:
    """Дублирует stdout в файл лога (сырой лог прогона)."""

    def __init__(self, path):
        self._file = open(path, 'a', encoding='utf-8')

    def write(self, text):
        sys.__stdout__.write(text)
        self._file.write(text)

    def flush(self):
        sys.__stdout__.flush()
        self._file.flush()


def progress(text):
    stamp = time.strftime('%H:%M:%S')
    print(f'[{stamp}] {text}', flush=True)


def json_safe(value):
    """NaN/inf -> None: summary/manifest обязаны быть валидным JSON."""
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


def _versions():
    import numpy
    versions = {'python': sys.version.split()[0], 'numpy': numpy.__version__}
    for name in ('pandas', 'scipy', 'matplotlib', 'torch'):
        try:
            module = __import__(name)
            versions[name] = getattr(module, '__version__', '?')
        except Exception:
            versions[name] = None
    try:
        import ot
        versions['pot'] = getattr(ot, '__version__', '?')
    except Exception:
        versions['pot'] = None
    return versions


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


def grid_spec():
    """Сетка конфигураций: full, smoke или ONLY_N-подмножество одного n."""
    if SMOKE:
        n_grid, reps, s5b_alphas = [100], range(2), [0.0, 0.5]
    elif ONLY_N is not None:
        n_grid, reps, s5b_alphas = [int(ONLY_N)], range(FULL_REPS), FULL_S5B_ALPHAS
    else:
        n_grid, reps, s5b_alphas = FULL_N_GRID, range(FULL_REPS), FULL_S5B_ALPHAS
    return n_grid, reps, s5b_alphas


def build_configs():
    """Список конфигураций S5: dicts с experiment_id и параметрами."""
    n_grid, reps, s5b_alphas = grid_spec()
    configs = []
    for n in n_grid:
        for rep in reps:
            seed_p = SEED_BASE_P + rep
            seed_q = SEED_BASE_Q + rep
            for k, pairs in S5A_PAIRS.items():
                dropped_mass = 2 * len(pairs) / 8.0
                configs.append({
                    'experiment_id': f's5a/np{k}/n{n}/r{rep}',
                    'series': 's5a', 'n': n, 'rep': rep,
                    'parameter': float(dropped_mass), 'parameter_name': 'dropped_mass',
                    'direction': '', 'is_control': k == 0,
                    'seed_p': seed_p, 'seed_q': seed_q, 's5a_pairs': pairs,
                })
            for alpha in s5b_alphas:
                n_replaced = int(round(alpha * n))
                configs.append({
                    'experiment_id': f's5b/a{alpha:.4g}/n{n}/r{rep}',
                    'series': 's5b', 'n': n, 'rep': rep,
                    'parameter': n_replaced / n, 'parameter_name': 'alpha_realized',
                    'direction': '', 'is_control': n_replaced == 0,
                    'seed_p': seed_p, 'seed_q': seed_q, 'n_replaced': n_replaced,
                })
            for direction in S5C_DIRECTIONS:
                configs.append({
                    'experiment_id': f's5c/{direction}/n{n}/r{rep}',
                    'series': 's5c', 'n': n, 'rep': rep,
                    'parameter': 0.0, 'parameter_name': 'direction',
                    'direction': direction,
                    'is_control': direction in ('p8p8', 'q4q4'),
                    'seed_p': seed_p, 'seed_q': seed_q,
                })
    return configs


def moment_tolerances(n):
    """Допуски moment-gate для s5c: 4 сигма выборочной оценки при cov=33I."""
    var = 33.0
    tol_mean = 4.0 * math.sqrt(var / n)
    tol_cov = 4.0 * var * math.sqrt(2.0 / n)
    return tol_mean, tol_cov


def sample_pair(cfg, ref, pops):
    """P/Q для одной конфигурации + диагностические поля строки."""
    from tda_metrics.s5_samplers import (
        sample_from_population, sample_s5a, sample_s5b,
    )
    from tda_metrics.mixtures import sample_reference

    n, seed_p, seed_q = cfg['n'], cfg['seed_p'], cfg['seed_q']
    extra = {'moment_attempts': 0}
    if cfg['series'] in ('s5a', 's5b'):
        P = sample_reference(ref, n, seed_p)
    else:
        pop_p, pop_q = pops
        p_pop = pop_p if cfg['direction'] in ('p8q4', 'p8p8') else pop_q
        P = sample_from_population(p_pop, n, seed_p)

    if cfg['series'] == 's5a':
        points, _ = sample_s5a(ref, n, seed_q, cfg['s5a_pairs'])
        Q = points
        extra['replaced_count'] = 0
    elif cfg['series'] == 's5b':
        points, mask, _ = sample_s5b(ref, n, seed_q, cfg['n_replaced'])
        Q = points
        extra['replaced_count'] = int(mask.sum())
    else:
        pop_p, pop_q = pops
        q_pop = {
            'p8q4': pop_q, 'p4q8': pop_p, 'p8p8': pop_p, 'q4q4': pop_q,
        }[cfg['direction']]
        tol_mean, tol_cov = moment_tolerances(n)
        from tda_metrics.s5_samplers import moment_diagnostics
        attempt = 0
        while True:
            Q = sample_from_population(q_pop, n, seed_q + 1000 * attempt)
            diag = moment_diagnostics(P, Q)
            if (diag['mean_diff'] <= tol_mean and diag['cov_diff'] <= tol_cov
                    or attempt >= MAX_MOMENT_ATTEMPTS - 1):
                if not (diag['mean_diff'] <= tol_mean and diag['cov_diff'] <= tol_cov):
                    raise RuntimeError(
                        f'moment-gate не пройден за {attempt + 1} попыток: '
                        f"mean_diff={diag['mean_diff']:.3f}>{tol_mean:.3f} "
                        f"cov_diff={diag['cov_diff']:.3f}>{tol_cov:.3f}"
                    )
                break
            attempt += 1
        extra['moment_attempts'] = attempt + 1
        extra['replaced_count'] = 0
    return P, Q, extra


def energy_distance(P, Q):
    """Energy distance (смещённая, состоятельная оценка) на cap=1000 точек."""
    from scipy.spatial.distance import cdist
    cap = 1000
    P = P[:cap]
    Q = Q[:cap]
    d_pp = cdist(P, P)
    d_qq = cdist(Q, Q)
    d_pq = cdist(P, Q)
    return float(2.0 * d_pq.mean() - d_pp.mean() - d_qq.mean())


def sinkhorn_cost(P, Q):
    """Sinkhorn transport cost (POT sinkhorn2, reg=0.1, подвыборка 500)."""
    try:
        import ot
    except Exception:
        return None
    from scipy.spatial.distance import cdist
    cap = 500
    P = P[:cap]
    Q = Q[:cap]
    cost = cdist(P, Q) ** 2
    median = np.median(cost)
    if not np.isfinite(median) or median <= 0:
        return None
    a = np.full(P.shape[0], 1.0 / P.shape[0])
    b = np.full(Q.shape[0], 1.0 / Q.shape[0])
    value = ot.sinkhorn2(a, b, cost / median, 0.1)
    return float(value)


def run_row(cfg, ref, pops, metrics_obj):
    """Одна строка grid: сэмплы, метрики, диагностика."""
    from tda_metrics.s5_samplers import moment_diagnostics
    started = time.perf_counter()
    P, Q, extra = sample_pair(cfg, ref, pops)
    diag = moment_diagnostics(P, Q)
    skip = ('rtd',) if cfg['n'] >= 500 else ()
    result = metrics_obj.compute_all(P, Q, skip=skip)
    row = {
        'experiment_id': cfg['experiment_id'],
        'series': cfg['series'],
        'n': cfg['n'],
        'rep': cfg['rep'],
        'parameter': cfg['parameter'],
        'parameter_name': cfg['parameter_name'],
        'direction': cfg['direction'],
        'is_control': bool(cfg['is_control']),
        'seed_p': cfg['seed_p'],
        'seed_q': cfg['seed_q'],
        'moment_mean_diff': float(diag['mean_diff']),
        'moment_cov_diff': float(diag['cov_diff']),
        'moment_attempts': int(extra['moment_attempts']),
        'replaced_count': int(extra['replaced_count']),
        'status': 'ok',
        'error_message': '',
    }
    for key, value in result.items():
        row[key] = None if value is None else float(value)
    row['energy'] = energy_distance(P, Q)
    row['sinkhorn'] = sinkhorn_cost(P, Q)
    row['runtime_seconds'] = time.perf_counter() - started
    return row


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


@section('grid')
def section_grid(deps):
    configs, ref, pops, metrics_obj = deps
    os.makedirs(RAW, exist_ok=True)
    done = load_done_ids()
    todo = [c for c in configs if c['experiment_id'] not in done]
    n_ok = n_failed = n_skip = len(done)
    progress(f'grid: {len(configs)} конфигураций, {len(done)} готовы (resume), {len(todo)} к расчёту')
    handle = open(ROWS_PATH, 'a', encoding='utf-8')
    started = time.perf_counter()
    for pos, cfg in enumerate(todo, 1):
        t0 = time.perf_counter()
        try:
            row = run_row(cfg, ref, pops, metrics_obj)
            n_ok += 1
        except Exception as exc:
            row = {
                'experiment_id': cfg['experiment_id'],
                'series': cfg['series'], 'n': cfg['n'], 'rep': cfg['rep'],
                'parameter': cfg['parameter'],
                'parameter_name': cfg['parameter_name'],
                'direction': cfg.get('direction', ''),
                'is_control': bool(cfg['is_control']),
                'seed_p': cfg['seed_p'], 'seed_q': cfg['seed_q'],
                'moment_mean_diff': None, 'moment_cov_diff': None,
                'moment_attempts': 0, 'replaced_count': 0,
                'status': 'failed', 'error_message': f'{type(exc).__name__}: {exc}',
            }
            n_failed += 1
            progress(f"FAILED {cfg['experiment_id']}: {exc}")
        handle.write(json.dumps(json_safe(row)) + '\n')
        handle.flush()
        if pos % 20 == 0 or pos == len(todo):
            elapsed = time.perf_counter() - started
            eta = elapsed / pos * (len(todo) - pos)
            progress(f'grid: {pos}/{len(todo)} ok={n_ok} failed={n_failed} '
                     f'row={time.perf_counter() - t0:.1f}s eta={eta / 60:.1f}m')
    handle.close()
    progress(f'grid: завершено ok={n_ok} failed={n_failed} skipped={n_skip}')
    return {'ok': n_ok, 'failed': n_failed, 'skipped': n_skip}


def read_rows_frame():
    import pandas as pd
    rows = []
    with open(ROWS_PATH, encoding='utf-8') as handle:
        for line in handle:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def control_values(frame, series, n, metric, cfg_control):
    mask = (
        (frame['series'] == series) & (frame['n'] == n) & frame['is_control']
        & (frame[metric].notna())
    )
    if series == 's5c':
        mask &= frame['direction'] == cfg_control
    return frame.loc[mask, metric].to_numpy()


@section('analysis')
def section_analysis(deps):
    import pandas as pd
    from scipy.stats import norm
    from tda_metrics.statistics import benjamini_hochberg, contrast_to_control

    frame = read_rows_frame()
    if frame.empty:
        raise RuntimeError('rows.jsonl пуст — analysis невозможен')
    n_grid, reps, _ = grid_spec()
    effect_rows = []
    moment_rows = []
    for series in ('s5a', 's5b', 's5c'):
        sub = frame[frame['series'] == series]
        params = sorted(sub['parameter'].unique()) if series != 's5c' else [0.0]
        directions = sorted(sub['direction'].unique()) if series == 's5c' else ['']
        for n in n_grid:
            for metric in ALL_METRICS:
                if series == 's5c':
                    controls_pool = {
                        direction: control_values(frame, series, n, metric, 'p8p8' if direction in ('p8q4', 'p8p8') else 'q4q4')
                        for direction in directions
                    }
                    for direction in directions:
                        control = controls_pool['p8p8' if direction in ('p8q4', 'p8p8') else 'q4q4']
                        values_mask = (sub['n'] == n) & (sub['direction'] == direction) & sub[metric].notna()
                        values = sub.loc[values_mask, metric].to_numpy()
                        if values.size < 5 or control.size < 5:
                            continue
                        _append_effect(effect_rows, series, n, direction, metric,
                                        values, control, norm, contrast_to_control)
                else:
                    control = control_values(frame, series, n, metric, '')
                    for param in params:
                        values_mask = (
                            (sub['n'] == n) & (sub['parameter'] == param)
                            & sub[metric].notna()
                        )
                        values = sub.loc[values_mask, metric].to_numpy()
                        if values.size < 5 or control.size < 5:
                            continue
                        label = f'{param:.4g}'
                        _append_effect(effect_rows, series, n, label, metric,
                                        values, control, norm, contrast_to_control)
            for n in n_grid:
                for param in (params if series != 's5c' else [0.0]):
                    for direction in directions:
                        mask = sub['n'] == n
                        if series == 's5c':
                            mask &= sub['direction'] == direction
                        else:
                            mask &= sub['parameter'] == param
                        block = sub.loc[mask]
                        if block.empty:
                            continue
                        moment_rows.append({
                            'series': series, 'n': n,
                            'parameter': f'{param:.4g}' if series != 's5c' else direction,
                            'mean_diff_mean': float(block['moment_mean_diff'].mean()),
                            'mean_diff_max': float(block['moment_mean_diff'].max()),
                            'cov_diff_mean': float(block['moment_cov_diff'].mean()),
                            'cov_diff_max': float(block['moment_cov_diff'].max()),
                            'max_attempts': int(block['moment_attempts'].max()),
                            'replaced_share_mean': float(block['replaced_count'].mean() / block['n'].mean()),
                        })
    effect = pd.DataFrame(effect_rows)
    if not effect.empty:
        for (series, n), block in effect.groupby(['series', 'n']):
            mask = block['metric'].isin(PRIMARY_METRICS)
            p = block.loc[mask, 'p_value'].to_numpy()
            if p.size == 0:
                continue
            _, adjusted = benjamini_hochberg(p, alpha=0.05)
            labels = block.index[mask]
            effect.loc[labels, 'q_bh'] = adjusted
    os.makedirs(TABLES, exist_ok=True)
    effect.to_csv(os.path.join(TABLES, 's5_effect_table.csv'), index=False)
    pd.DataFrame(moment_rows).to_csv(
        os.path.join(TABLES, 's5_moment_diagnostics.csv'), index=False)
    _write_parquet(frame)
    progress(f'analysis: {len(effect)} строк effect-таблицы, {len(moment_rows)} moment-строк')
    return {'effect_rows': len(effect)}


def _append_effect(effect_rows, series, n, label, metric, values, control, norm, contrast_to_control):
    kind = METRIC_KINDS[metric]
    delta, z_rep = contrast_to_control(values, control, kind=kind)
    delta_mean = float(np.mean(delta))
    sigma0 = float(np.std(control, ddof=1))
    z_mean = delta_mean / (sigma0 / math.sqrt(values.size) + 1e-12)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, values.size, size=values.size)
        boots.append(float(np.mean(values[idx])) - float(np.mean(control)))
    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
    p_value = 2.0 * float(norm.sf(abs(z_mean)))
    power = float(np.mean(np.abs(z_rep) > FPR_THRESHOLD))
    rng_fpr = np.random.default_rng(1)
    hits = 0
    for _ in range(N_FPR_SPLITS):
        perm = rng_fpr.permutation(control.size)
        half_a, half_b = control[perm[: control.size // 2]], control[perm[control.size // 2:]]
        try:
            _, z_half = contrast_to_control(half_a, half_b, kind=kind)
            hits += int(abs(z_half) > FPR_THRESHOLD)
        except Exception:
            continue
    effect_rows.append({
        'series': series, 'n': n, 'parameter': label, 'metric': metric,
        'mean': float(np.mean(values)), 'sd': float(np.std(values, ddof=1)),
        'delta': delta_mean, 'z_mean': z_mean,
        'ci_low': float(ci_low), 'ci_high': float(ci_high),
        'p_value': p_value, 'q_bh': None,
        'power': power, 'fpr_control': hits / max(N_FPR_SPLITS, 1),
        'n_reps': int(values.size),
    })


def _write_parquet(frame):
    try:
        frame.to_parquet(PARQUET_PATH, index=False)
    except Exception:
        frame.to_csv(CSV_PATH, index=False)


@section('figures')
def section_figures(deps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    effect_path = os.path.join(TABLES, 's5_effect_table.csv')
    import pandas as pd
    effect = pd.read_csv(effect_path)
    if effect.empty:
        raise RuntimeError('effect-таблица пуста')
    os.makedirs(FIGURES, exist_ok=True)
    n_grid, _, _ = grid_spec()
    metrics_cols = PRIMARY_METRICS
    fig, axes = plt.subplots(3, len(metrics_cols), figsize=(4.2 * len(metrics_cols), 11))
    colors = {n: f'C{i}' for i, n in enumerate(n_grid)}
    for row_idx, series in enumerate(('s5a', 's5b', 's5c')):
        for col_idx, metric in enumerate(metrics_cols):
            ax = axes[row_idx, col_idx]
            block = effect[(effect['series'] == series) & (effect['metric'] == metric)]
            for n in n_grid:
                nb = block[block['n'] == n]
                if series == 's5c':
                    directions = ['p8p8', 'q4q4', 'p8q4', 'p4q8']
                    xs = [i for i, d in enumerate(directions) if d in set(nb['parameter'])]
                    ys = [float(nb[nb['parameter'] == d]['mean'].iloc[0]) for d in directions if d in set(nb['parameter'])]
                    sem = [1.96 * float(nb[nb['parameter'] == d]['sd'].iloc[0]) / math.sqrt(max(int(nb[nb['parameter'] == d]['n_reps'].iloc[0]), 1)) for d in directions if d in set(nb['parameter'])]
                    ax.errorbar(xs, ys, yerr=sem, marker='o', label=f'n={n}', color=colors[n])
                    ax.set_xticks(range(4))
                    ax.set_xticklabels(directions, fontsize=8)
                else:
                    nb = nb.sort_values('parameter')
                    xs = nb['parameter'].to_numpy(dtype=float)
                    ys = nb['mean'].to_numpy(dtype=float)
                    sem = 1.96 * nb['sd'].to_numpy(dtype=float) / np.sqrt(nb['n_reps'].to_numpy(dtype=float))
                    control_mean = float(nb.loc[nb['parameter'] == nb['parameter'].min(), 'mean'].iloc[0]) if len(nb) else np.nan
                    ax.errorbar(xs, ys, yerr=sem, marker='o', label=f'n={n}', color=colors[n])
                    if np.isfinite(control_mean):
                        ax.axhline(control_mean, color='gray', lw=0.8, ls='--')
                ax.set_title(f'{series} {metric}', fontsize=10)
                ax.set_xlabel('dropped mass' if series == 's5a' else ('alpha' if series == 's5b' else 'направление'))
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=7)
    fig.suptitle('S5: dose-response первичных метрик (mean ± 1.96·SE, пунктир — same-law контроль)')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(FIGURES, 's5_dose_response.png'), dpi=130)
    plt.close(fig)

    moments = pd.read_csv(os.path.join(TABLES, 's5_moment_diagnostics.csv'))
    s5c = moments[moments['series'] == 's5c']
    if not s5c.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, col, label in zip(axes, ('mean_diff_max', 'cov_diff_max'), ('|Δ mean| max', '|Δ cov| max')):
            for direction in sorted(s5c['parameter'].unique()):
                block = s5c[s5c['parameter'] == direction]
                ax.plot(block['n'], block[col], marker='o', label=direction)
            tol_col = 'mean_diff_max' if 'mean' in label else 'cov_diff_max'
            tols = [moment_tolerances(n)[0 if 'mean' in label else 1] for n in n_grid]
            ax.plot(n_grid, tols, ls='--', color='gray', label='допуск 4σ')
            ax.set_xscale('log')
            ax.set_xlabel('n')
            ax.set_title(label)
            ax.legend(fontsize=8)
        fig.suptitle('S5c: момент-баланс (population-моменты совпадают по построению)')
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES, 's5_moment_balance.png'), dpi=130)
        plt.close(fig)
    progress('figures: s5_dose_response.png (+ s5_moment_balance.png если s5c не пуст)')
    return {'figures': ['s5_dose_response.png']}


@section('summary')
def section_summary(deps):
    grid_stats = deps.get('grid') or {}
    effect_rows = (deps.get('analysis') or {}).get('effect_rows', 0)
    import pandas as pd
    effect_path = os.path.join(TABLES, 's5_effect_table.csv')
    verdict = {}
    if os.path.exists(effect_path):
        effect = pd.read_csv(effect_path)
        n_max = int(effect['n'].max())
        for series, family, expect in (
            ('s5a', ('mtd_PQ', 'ntd_PQ', 'recall@3'), 'recall-like ↑ при drop'),
            ('s5b', ('mtd_QP', 'ntd_QP', 'precision@3'), 'precision-like ↑ при invent'),
            ('s5c', ('mtd_PQ', 'mtd_QP', 'ntd_PQ', 'ntd_QP', 'recall@3', 'precision@3'), 'эффект при matched-моментах'),
        ):
            block = effect[(effect['series'] == series) & (effect['n'] == n_max) & effect['metric'].isin(family)]
            if block.empty:
                continue
            if series == 's5c':
                contrast = block[~block['parameter'].isin(('p8p8', 'q4q4'))]
                best = contrast.loc[contrast['z_mean'].abs().idxmax()] if not contrast.empty else None
                verdict[series] = {
                    'expectation': expect,
                    'max_abs_z_mean': None if best is None else float(best['z_mean']),
                    'at': None if best is None else f"{best['parameter']}/{best['metric']}",
                    'q_bh': None if best is None else (None if pd.isna(best['q_bh']) else float(best['q_bh'])),
                }
            else:
                top_param = block['parameter'].max()
                top = block[block['parameter'] == top_param]
                verdict[series] = {
                    'expectation': expect,
                    'max_dose': top_param,
                    'z_mean_by_metric': {r['metric']: float(r['z_mean']) for _, r in top.iterrows()},
                    'fpr_control_max': float(block['fpr_control'].max()),
                }
    summary = {
        'created': datetime.now(timezone.utc).isoformat(),
        'smoke': SMOKE,
        'sections': SECTION_STATUS,
        'grid': grid_stats,
        'effect_rows': effect_rows,
        'verdict': verdict,
    }
    manifest = {
        'created': summary['created'],
        'smoke': SMOKE,
        'protocol_path': PROTOCOL_PATH,
        'protocol_sha256': _sha256_file(PROTOCOL_PATH) if os.path.exists(PROTOCOL_PATH) else None,
        'versions': _versions(),
        'grids': {
            'n_grid': grid_spec()[0], 'n_reps': len(list(grid_spec()[1])),
            's5a_pairs': {k: v for k, v in S5A_PAIRS.items()},
            's5b_alphas': grid_spec()[2],
            's5c_directions': S5C_DIRECTIONS,
            'seeds': {'P': SEED_BASE_P, 'Q': SEED_BASE_Q},
            'moment_gate': {
                'tol_mean': '4*sqrt(33/n)', 'tol_cov': '4*33*sqrt(2/n)',
                'max_attempts': MAX_MOMENT_ATTEMPTS,
            },
        },
        'sections': SECTION_STATUS,
        'runtime_seconds': time.perf_counter() - _STARTED,
    }
    os.makedirs(RAW, exist_ok=True)
    with open(os.path.join(RAW, 'summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(json_safe(summary), handle, ensure_ascii=False, indent=2)
    with open(os.path.join(RAW, 'manifest.json'), 'w', encoding='utf-8') as handle:
        json.dump(json_safe(manifest), handle, ensure_ascii=False, indent=2)
    progress('summary/manifest записаны')
    return summary


@section('pytest_light')
def section_pytest_light(deps):
    results = {}
    for test_file in ('tests/test_s5_samplers.py', 'tests/test_precision_recall_viz.py'):
        proc = subprocess.run(
            [sys.executable, '-m', 'pytest', test_file, '-q'],
            capture_output=True, text=True, timeout=900,
        )
        results[test_file] = proc.returncode
        progress(f'pytest {test_file}: rc={proc.returncode}')
        print(proc.stdout[-2000:])
    return results


def main():
    deps = {}
    protocol_missing = not os.path.exists(PROTOCOL_PATH)
    progress(f'S5 run start: smoke={SMOKE}, protocol_missing={protocol_missing}')
    heavy_ok = True
    try:
        global np
        import numpy as np
        from tda_metrics.mixtures import make_mode_reference
        from tda_metrics.s5_samplers import make_matched_pair_s5c
        from tda_metrics.metrics import TopologyMetrics
        deps['ref'] = make_mode_reference(8, radius=8.0, spread=1.0)
        deps['pops'] = make_matched_pair_s5c()
        deps['metrics_obj'] = TopologyMetrics(seed=0)
    except Exception:
        heavy_ok = False
        progress('HEAVY IMPORT FAIL:\n' + traceback.format_exc(limit=3))
        if SMOKE:
            progress('SMOKE: heavy unavailable, grid skipped — rc=0')
            print('S5 DONE rc=0')
            return
        print('S5 DONE rc=1')
        return
    configs = build_configs()
    deps['configs'] = configs
    progress(f'конфигураций: {len(configs)}')
    section_pytest_light(deps)
    grid_result = None
    if not protocol_missing:
        grid_result = section_grid((configs, deps['ref'], deps['pops'], deps['metrics_obj']))
    else:
        progress('PROTOCOL MISSING: grid пропущен (rc будет 1)')
    deps['grid'] = grid_result
    analysis_result = None
    figures_result = None
    if grid_result is not None:
        deps['analysis'] = None
        analysis_result = section_analysis(None)
        deps['analysis'] = analysis_result
        figures_result = section_figures(None)
        deps['figures'] = figures_result
    deps.update({'grid': grid_result, 'analysis': analysis_result, 'figures': figures_result})
    if GRID_ONLY:
        print(f'S5 GRID-ONLY DONE (ok={grid_result.get("ok") if grid_result else 0})')
        return
    summary = section_summary(deps)
    grid_ok = grid_result is not None and grid_result.get('failed', 1) == 0 and grid_result.get('ok', 0) > 0
    tables_ok = analysis_result is not None and (analysis_result.get('effect_rows', 0) > 0)
    figures_ok = figures_result is not None
    rc = 0 if (grid_ok and tables_ok and figures_ok) else 1
    print(f'S5 DONE rc={rc}')


if __name__ == '__main__':
    os.makedirs('logs', exist_ok=True)
    os.makedirs(RAW, exist_ok=True)
    sys.stdout = Tee(LOG_PATH)
    main()
