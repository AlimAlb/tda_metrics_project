"""Переагрегация power-таблиц wave2 с direction-aware альтернативами (v2, R-PWR-1).

Дефект v1: aggregate_power вызывался с единым alternative='greater' для всех
метрик, поэтому similarity-метрики (precision@k, recall@k), у которых эффект
направлен ВНИЗ, получали power~0 даже в своих направлениях (пример: precision@3
на invent0.5 при n=1000: 0.972 -> 0.653 при пороге 0.982, power 0.0).
Исправление: aggregate_power(kind_map=default_alternative_map(metrics)).
Raw-значения power_values.parquet корректны и НЕ пересчитываются —
перегоняется только агрегация (сводка, min_n, заметки, фигуры).
Запуск из корня репо: python scripts/rebuild_power_tables.py
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POWER_MODULE_PATH = os.path.join(REPO_ROOT, 'src', 'tda_metrics', 'power.py')

ALPHA_GRID = (0.05, 0.10, 0.20)
TARGET_POWER = 0.8
FIGURE_METRICS = ('ntd_PQ', 'ntd_QP', 'recall@3', 'precision@3', 'rtd', 'mmd', 'frechet')


def _load_power_module():
    spec = importlib.util.spec_from_file_location('power_v2', POWER_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(65536), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--values', default='results/raw/power/power_values.parquet')
    parser.add_argument('--manifest', default='results/raw/power/manifest.json')
    parser.add_argument('--tables', default='results/tables')
    parser.add_argument('--figures', default='results/figures')
    args = parser.parse_args()

    import pandas as pd

    power = _load_power_module()
    aggregate_power = power.aggregate_power
    select_min_n = power.select_min_n
    default_alternative_map = power.default_alternative_map

    values = pd.read_parquet(args.values)
    with open(args.manifest, encoding='utf-8') as handle:
        manifest = json.load(handle)
    os.makedirs(args.tables, exist_ok=True)
    print(f'values: {len(values)} строк; эффекты: {sorted(values["effect"].unique())}')

    kind_map = default_alternative_map(values['metric'].unique())
    print('alternative map:', {m: a for m, a in sorted(kind_map.items())
                                if a == 'less'})

    summary_parts = []
    min_n_rows = []
    for effect, group in values.groupby('effect'):
        summary = aggregate_power(group, alpha_grid=ALPHA_GRID, kind_map=kind_map)
        summary['effect'] = effect
        summary_parts.append(summary)
        for alpha in ALPHA_GRID:
            min_n = select_min_n(summary[summary['alpha'] == alpha], target_power=TARGET_POWER)
            for metric, n_value in min_n.items():
                min_n_rows.append({'effect': effect, 'alpha': alpha,
                                   'metric': metric, 'min_n': n_value})
    summary_frame = pd.concat(summary_parts, ignore_index=True)
    summary_frame.to_csv(os.path.join(args.tables, 'power_summary.csv'), index=False)
    min_n_frame = pd.DataFrame(min_n_rows)
    min_n_frame.to_csv(os.path.join(args.tables, 'power_min_n.csv'), index=False)

    notes = {
        'n_control': manifest['params']['n_control'],
        'n_effect': manifest['params']['n_effect'],
        'effective_alpha_note': (
            'пороги эмпирические: greater — quantile(control, 1-alpha, method="higher"), '
            'less — quantile(control, alpha, method="lower"); фактический уровень '
            '1/(n_control+1) при вырождении; alpha=0.05 и 0.10 различимы при n_control>=20'),
        'effects_discrete': {'drop0.1': '1/8 = 12.5% мод', 'drop0.25': '2/8 = 25%',
                             'invent0.15': '1 пара = 2/8 = 25%',
                             'invent0.5': '2 пары = 4/8 = 50%'},
        'v2_reaggregation': {
            'rebuilt_at': datetime.now(timezone.utc).isoformat(),
            'reason': (
                'v1 применял alternative="greater" ко всем метрикам — similarity-метрики '
                '(precision@k/recall@k) с эффектом вниз получали power~0 даже в своих '
                'направлениях; v2 агрегирует их с alternative="less" через '
                'aggregate_power(kind_map=default_alternative_map)'),
            'input_values_sha256': _sha256_file(args.values),
            'raw_values_unchanged': True,
        },
    }
    with open(os.path.join(args.tables, 'power_notes.json'), 'w', encoding='utf-8') as handle:
        json.dump(notes, handle, ensure_ascii=False, indent=2)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(args.figures, exist_ok=True)
    for effect, group in summary_frame.groupby('effect'):
        fig, ax = plt.subplots(figsize=(9, 5))
        for metric in FIGURE_METRICS:
            sub = group[(group['metric'] == metric) & (group['alpha'] == 0.05)]
            if sub.empty:
                continue
            ax.plot(sub['n'], sub['power'], marker='o', label=metric)
        ax.axhline(TARGET_POWER, color='gray', ls='--', lw=1, label='target 0.8')
        ax.set_title(f'Power(n) при alpha=0.05 — эффект {effect} (v2, direction-aware)')
        ax.set_xlabel('n (размер облака)')
        ax.set_ylabel('мощность')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.figures, f"power_{str(effect).replace('.', '_')}.png"),
                    dpi=130)
        plt.close(fig)
        print('фигура power v2:', effect)

    print('\n=== select_min_n (alpha=0.05, target 0.8) — v2 ===')
    for effect in sorted(min_n_frame['effect'].unique()):
        block = min_n_frame[(min_n_frame['effect'] == effect) & (min_n_frame['alpha'] == 0.05)]
        pairs = {row['metric']: (None if pd.isna(row['min_n']) else int(row['min_n']))
                  for _, row in block.iterrows()}
        print(effect, '->', {m: n for m, n in sorted(pairs.items())})
    print('REBUILD DONE')


if __name__ == '__main__':
    sys.exit(main())
