"""Сборка таблиц и фигур фазы синтетики S0/S1/S3/S4 из raw-паркета (v2, после review).

Правки v2 (review B-1/M-1/M-2/M-3/M-5/M-6/M-8):
- единая знаковая конвенция §5.1: similarity-метрики (precision/recall) —
  разворот знака («хуже» = положительный контраст), в CSV хранятся знаковые
  delta/z;
- S4: Welch-CI разности двух средних (дисперсия контроля учтена), обнаружение =
  весь знаковый CI выше нуля (для similarity — весь CI ниже нуля исходной
  шкалы); в таблице — размер эффекта (delta, z) в t* и фактическое число
  мод/пар (гранулярность сетки, no-op уровни помечены);
- S3 sensitivity: sign-контраст по уровням ДО max для направленных метрик;
  для двусторонних (NTD, RTD) — max|Δ|, конвенция — в подписи фигуры.

Читает ТОЛЬКО results.parquet хранилища. Выход: /content/results/tables/*.csv,
/content/results/figures/*.png. SystemExit не поднимается; Tee в
/content/wave1b_tables_log.txt.
"""
import sys
import traceback

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys_stdout = sys.stdout


class Tee:
    def __init__(self, path):
        self.file = open(path, 'w', encoding='utf-8')

    def write(self, text):
        sys_stdout.write(text)
        self.file.write(text)

    def flush(self):
        sys_stdout.flush()
        self.file.flush()


sys.stdout = Tee('/content/wave1b_tables_log.txt')

TABLES = '/content/results/tables'
FIGURES = '/content/results/figures'

METRICS = [
    'mtd_PQ', 'mtd_QP', 'ntd_PQ', 'ntd_QP', 'rtd',
    'precision@3', 'recall@3', 'precision@10', 'recall@10',
    'mmd', 'frechet', 'js',
]
SIMILARITY = {name for name in METRICS if 'precision' in name or 'recall' in name}
TWOSIDED = {'ntd_PQ', 'ntd_QP', 'rtd'}
N_MODES = 8


def metric_kind(metric):
    """Конвенция знака: +1 — рост значения = «хуже» (distance); -1 — падение = «хуже» (similarity)."""
    return -1 if metric in SIMILARITY else 1


def actual_modes(level, axis):
    """Фактическое число удалённых мод / добавленных пар на уровне (гранулярность сетки)."""
    if axis == 'drop':
        return int(round(level * N_MODES))
    return int(round(level * N_MODES / 2.0))


results = {}

try:
    import os
    os.makedirs(TABLES, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    frame = pd.read_parquet('/content/results/raw/synthetic/results.parquet')
    print('raw строк:', len(frame))
    ok = frame[frame['status'] == 'ok'].copy()
    print('ok строк:', len(ok), '; failed:', len(frame) - len(ok))
    metric_cols = [c for c in METRICS if c in ok.columns]
    results['load'] = 'OK'
except Exception:
    results['load'] = 'FAIL'
    traceback.print_exc(file=sys.stdout)

# ============ S0: полы vs n ============
try:
    print('\n=== S0: полы (same-law) и копия ===')
    s0 = ok[ok['experiment_id'].str.startswith('s0_')].copy()
    parts = s0['experiment_id'].str.split('_', expand=True)
    s0['distribution'] = parts[1]
    s0['mode'] = parts[2]
    s0['n'] = parts[3].str[1:].astype(int)
    floors = (
        s0[s0['mode'] == 'samelaw']
        .groupby(['distribution', 'n'])[metric_cols]
        .agg(['mean', 'std'])
    )
    floors.to_csv(f'{TABLES}/s0_floors.csv')
    copies = (
        s0[s0['mode'] == 'copy']
        .groupby(['distribution', 'n'])[metric_cols]
        .agg(['mean', 'std'])
    )
    copies.to_csv(f'{TABLES}/s0_copy.csv')
    for name in ('gaussian', 'ring', 'mixture'):
        sub = s0[(s0['mode'] == 'samelaw') & (s0['distribution'] == name)]
        print(f'{name} same-law: n=1000 полы — '
              + ', '.join(
                  f"{c}={sub[sub['n'] == 1000][c].mean():.3f}"
                  for c in ('mtd_PQ', 'rtd', 'ntd_PQ', 'mmd') if c in sub.columns
              ))
        n250 = s0[(s0['mode'] == 'samelaw') & (s0['distribution'] == name) & (s0['n'] == 250)]
        if len(n250):
            print(f'{name} same-law: n=250 ntd_PQ={n250["ntd_PQ"].mean():.3f} (подвыборки пересекаются: 3m/n>1)')
        copy_sub = s0[(s0['mode'] == 'copy') & (s0['distribution'] == name)]
        print(f'{name} copy: n=1000 — '
              + ', '.join(
                  f"{c}={copy_sub[copy_sub['n'] == 1000][c].mean():.4g}"
                  for c in ('mmd', 'precision@3', 'ntd_PQ') if c in copy_sub.columns
              ))

    fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharex=True)
    for row, name in enumerate(('gaussian', 'ring', 'mixture')):
        for col, metric in enumerate(('mtd_PQ', 'rtd', 'mmd', 'ntd_PQ')):
            ax = axes[row, col]
            sub = s0[(s0['mode'] == 'samelaw') & (s0['distribution'] == name)]
            for n, group in sub.groupby('n'):
                ax.errorbar(n, group[metric].mean(), yerr=group[metric].std(),
                            fmt='o', color='tab:blue', capsize=3)
            copy = s0[(s0['mode'] == 'copy') & (s0['distribution'] == name)]
            for n, group in copy.groupby('n'):
                ax.errorbar(n, group[metric].mean(), yerr=group[metric].std(),
                            fmt='s', color='tab:red', capsize=3)
            ax.set_title(f'{name}: {metric}', fontsize=9)
            ax.set_xscale('log')
    fig.suptitle('S0: шумовые полы (same-law, синие) и точная копия (красные) vs n')
    fig.tight_layout()
    fig.savefig(f'{FIGURES}/s0_floors.png', dpi=150)
    plt.close(fig)
    results['s0'] = 'OK'
except Exception:
    results['s0'] = 'FAIL'
    traceback.print_exc(file=sys.stdout)

# ============ S1: карта инвариантов (z со знаком «хуже» = +) ============
try:
    print('\n=== S1: знаковые z-контрасты относительно same-law ===')
    s1 = ok[ok['experiment_id'].str.startswith('s1_')].copy()
    parts = s1['experiment_id'].str.split('_', expand=True)
    s1['transformation'] = parts[1]
    s1['level'] = parts[2].str[1:]
    s1['level'] = np.where(s1['transformation'] == 'reflection',
                           s1['level'].astype(float).astype(int),
                           s1['level'].astype(float))
    s1sl = ok[ok['experiment_id'].str.startswith('s1samelaw_')].copy()
    s1sl['transformation'] = s1sl['experiment_id'].str.split('_', expand=True)[1]

    z_rows = []
    for transformation, sub in s1.groupby('transformation'):
        control = s1sl[s1sl['transformation'] == transformation][metric_cols]
        for metric in metric_cols:
            kind = metric_kind(metric)
            mu0, sd0 = control[metric].mean(), control[metric].std(ddof=1)
            for level, group in sub.groupby('level'):
                delta = kind * (group[metric].mean() - mu0)
                z = delta / (sd0 + 1e-12)
                z_rows.append({
                    'transformation': transformation, 'level': level,
                    'metric': metric, 'z_signed': z, 'delta_signed': delta,
                    'raw': group[metric].mean(), 'mu0': mu0, 'sd0': sd0,
                })
    z_table = pd.DataFrame(z_rows)
    z_table.to_csv(f'{TABLES}/s1_z.csv', index=False)

    max_levels = s1.groupby('transformation')['level'].max()
    z_max = pd.DataFrame([
        {
            'transformation': transformation,
            **{
                metric: float(
                    z_table[(z_table['transformation'] == transformation)
                            & (z_table['level'] == max_levels[transformation])
                            & (z_table['metric'] == metric)]['z_signed'].iloc[0]
                )
                for metric in metric_cols
            },
        }
        for transformation in max_levels.index
    ]).set_index('transformation')
    z_max.to_csv(f'{TABLES}/s1_z_max_level.csv')
    print('знаковые z на максимальном уровне (хуже = +, similarity развёрнуты):\n',
          z_max.round(2).to_string())

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(z_max.values, cmap='RdBu_r', vmin=-10, vmax=10, aspect='auto')
    ax.set_xticks(range(len(z_max.columns)), z_max.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(z_max.index)), z_max.index)
    for i in range(z_max.shape[0]):
        for j in range(z_max.shape[1]):
            ax.text(j, i, f'{z_max.values[i, j]:.1f}', ha='center', va='center', fontsize=7)
    fig.colorbar(im, ax=ax, label='знаковый z-контраст (хуже контроля = +)')
    ax.set_title('S1: стандартизованные эффекты трансформаций (максимальный уровень; similarity со знаком «хуже»)')
    fig.tight_layout()
    fig.savefig(f'{FIGURES}/s1_invariant_map.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False)
    for ax, (transformation, sub) in zip(axes.flat, s1.groupby('transformation')):
        control = s1sl[s1sl['transformation'] == transformation][metric_cols]
        for metric in ('rtd', 'mmd', 'mtd_PQ'):
            kind = metric_kind(metric)
            mu0, sd0 = control[metric].mean(), control[metric].std(ddof=1)
            means = sub.groupby('level')[metric].mean()
            ax.plot(means.index, kind * (means - mu0) / (sd0 + 1e-12), marker='o', label=metric)
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_title(transformation)
        ax.legend(fontsize=7)
    fig.suptitle('S1: знаковые z-траектории по уровням (rtd — положительный контроль инвариантности)')
    fig.tight_layout()
    fig.savefig(f'{FIGURES}/s1_trajectories.png', dpi=150)
    plt.close(fig)
    results['s1'] = 'OK'
except Exception:
    results['s1'] = 'FAIL'
    traceback.print_exc(file=sys.stdout)

# ============ S3: dropping × invention ============
try:
    print('\n=== S3: сетка dropping × invention ===')
    s3 = ok[ok['experiment_id'].str.startswith('s3_')].copy()
    parts = s3['experiment_id'].str.split('_', expand=True)
    s3['d'] = parts[1].str[1:].astype(float)
    s3['i'] = parts[2].str[1:].astype(float)
    grid = s3.groupby(['d', 'i'])[metric_cols].agg(['mean', 'std', 'count'])
    grid.to_csv(f'{TABLES}/s3_grid.csv')

    base = s3[(s3['d'] == 0.0) & (s3['i'] == 0.0)][metric_cols]
    ds = sorted(s3['d'].unique())
    is_ = sorted(s3['i'].unique())
    heat_metrics = ('mtd_PQ', 'mtd_QP', 'ntd_PQ', 'ntd_QP', 'rtd',
                    'precision@3', 'recall@3', 'mmd', 'frechet')
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    for ax, metric in zip(axes.flat, heat_metrics):
        kind = metric_kind(metric)
        twosided = metric in TWOSIDED
        matrix = np.zeros((len(is_), len(ds)))
        cell_text = np.empty((len(is_), len(ds)), dtype=object)
        for a, d in enumerate(ds):
            for b, i in enumerate(is_):
                delta = s3[(s3['d'] == d) & (s3['i'] == i)][metric].mean() - base[metric].mean()
                shown = delta if twosided else kind * delta
                matrix[b, a] = shown
                cell_text[b, a] = f'{shown:.2g}'
        im = ax.imshow(matrix, cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(ds)), [str(d) for d in ds])
        ax.set_yticks(range(len(is_)), [str(i) for i in is_])
        ax.set_xlabel('drop')
        ax.set_ylabel('invent')
        subtitle = 'Δ от (0,0), двусторонняя' if twosided else 'знаковый Δ (хуже = +)'
        ax.set_title(f'{metric} [{subtitle}]', fontsize=9)
        for b in range(len(is_)):
            for a in range(len(ds)):
                ax.text(a, b, cell_text[b, a], ha='center', va='center', fontsize=6)
        fig.colorbar(im, ax=ax)
    fig.suptitle('S3: контрасты относительно (0,0) same-law (PR с разворотом знака; NTD/RTD — сырой Δ)')
    fig.tight_layout()
    fig.savefig(f'{FIGURES}/s3_heatmaps.png', dpi=150)
    plt.close(fig)

    sens = []
    for metric in metric_cols:
        kind = metric_kind(metric)
        twosided = metric in TWOSIDED
        values = {}
        for axis_name, col in (('d', 'drop'), ('i', 'invent')):
            fixed = s3[(s3['i'] == 0.0) if axis_name == 'd' else (s3['d'] == 0.0)]
            line = fixed[fixed[axis_name] > 0].groupby(axis_name)[metric].mean() - base[metric].mean()
            if twosided:
                values[col] = float(np.abs(line).max())
            else:
                values[col] = float((kind * line).max())
        sens.append({
            'metric': metric,
            'kind': 'twosided' if twosided else 'signed',
            'drop_sensitivity': values['drop'],
            'invent_sensitivity': values['invent'],
        })
    sens_table = pd.DataFrame(sens)
    sens_table.to_csv(f'{TABLES}/s3_sensitivity.csv', index=False)
    print('sensitivity plane (signed: хуже = +; twosided: max|Δ|):\n',
          sens_table.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 7))
    for row in sens_table.itertuples():
        x, y = row.drop_sensitivity, row.invent_sensitivity
        ax.scatter(x, y)
        ax.annotate(row.metric, (x, y), fontsize=8)
        ax.plot([0, x], [0, y], lw=0.5, alpha=0.5)
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel('чувствительность к dropping (хуже = +; NTD/RTD: |Δ|)')
    ax.set_ylabel('чувствительность к invention (хуже = +; NTD/RTD: |Δ|)')
    ax.set_title('S3: плоскость чувствительностей (sign-контраст до max; двусторонние — max|Δ|)')
    fig.tight_layout()
    fig.savefig(f'{FIGURES}/s3_sensitivity_plane.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (axis, col) in zip(axes, (('d', 'drop'), ('i', 'invent'))):
        for metric in ('mtd_PQ', 'mtd_QP', 'ntd_PQ', 'ntd_QP', 'precision@3', 'recall@3'):
            line = s3[(s3['i'] == 0.0) if axis == 'd' else (s3['d'] == 0.0)].groupby(axis)[metric].mean()
            ax.plot(line.index, line.values, marker='o', label=metric)
        ax.set_title(f'срез по {col} (противоположная ось = 0)')
        ax.set_xlabel(col + ' fraction')
        ax.legend(fontsize=7)
    fig.suptitle('S3: одномерные срезы (зеркальные режимы из exp3а/3б)')
    fig.tight_layout()
    fig.savefig(f'{FIGURES}/s3_slices.png', dpi=150)
    plt.close(fig)
    results['s3'] = 'OK'
except Exception:
    results['s3'] = 'FAIL'
    traceback.print_exc(file=sys.stdout)

# ============ S4: первое обнаружение (v2: Welch-CI, правильный хвост, эффект-сайз, моды) ============
try:
    print('\n=== S4: первый обнаруженный шаг (v2) ===')
    s4 = ok[ok['experiment_id'].str.startswith('s4_')].copy()
    parts = s4['experiment_id'].str.split('_', expand=True)
    s4['axis'] = parts[1]
    s4['level'] = parts[2].str[1:].astype(float)

    base = s3[(s3['d'] == 0.0) & (s3['i'] == 0.0)]
    control = {}
    for metric in metric_cols:
        control[metric] = base[metric]
    rows = []
    for axis, sub in s4.groupby('axis'):
        for metric in metric_cols:
            kind = metric_kind(metric)
            ctrl = control[metric]
            mu0 = ctrl.mean()
            sd0 = ctrl.std(ddof=1)
            n_ctrl = len(ctrl)
            first = None
            share = None
            first_delta = None
            first_z = None
            for level in sorted(sub['level'].unique()):
                group = sub[sub['level'] == level][metric]
                n_g = len(group)
                delta = kind * (group.mean() - mu0)
                sem_diff = np.sqrt(group.var(ddof=1) / n_g + ctrl.var(ddof=1) / n_ctrl)
                ci_low = delta - 1.96 * sem_diff
                ci_high = delta + 1.96 * sem_diff
                if min(ci_low, ci_high) > 0:
                    first = level
                    first_delta = float(delta)
                    first_z = float(delta / (sd0 + 1e-12))
                    share = float(np.mean(kind * (group - mu0) > 0))
                    break
            rows.append({
                'axis': axis, 'metric': metric, 'first_detected': first,
                'delta_signed': first_delta, 'z_signed': first_z,
                'direction_share': share,
                'actual_changes': (actual_modes(first, axis) if first is not None else None),
                'noop_level': (first is not None and actual_modes(first, axis) == 0),
            })
    s4_table = pd.DataFrame(rows)
    s4_table.to_csv(f'{TABLES}/s4_first_detected.csv', index=False)
    print(s4_table.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot = s4_table.pivot(index='metric', columns='axis', values='first_detected')
    pivot = pivot.loc[[m for m in metric_cols if m in pivot.index]]
    ax.imshow(pivot.notna().astype(float).values, cmap='Greys', alpha=0.15, aspect='auto')
    for j, axis in enumerate(pivot.columns):
        for row_idx, metric in enumerate(pivot.index):
            value = pivot.loc[metric, axis]
            ax.text(j, row_idx, '—' if pd.isna(value) else f'{value:g}', ha='center', va='center', fontsize=9)
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title('S4: первый обнаруженный уровень (Welch 95% CI, весь CI «хуже нуля»; без BH — exploratory)')
    fig.tight_layout()
    fig.savefig(f'{FIGURES}/s4_first_detected.png', dpi=150)
    plt.close(fig)
    results['s4'] = 'OK'
except Exception:
    results['s4'] = 'FAIL'
    traceback.print_exc(file=sys.stdout)

print('\n=== ИТОГ СБОРКИ ТАБЛИЦ/ФИГУР (v2) ===')
for key, value in results.items():
    print(f'  {key}: {value}')
sys.stdout.flush()