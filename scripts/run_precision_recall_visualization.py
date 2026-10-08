#!/usr/bin/env python3
"""CPU-раннер визуализации improved precision/recall@k на 2D-смесях.

Строит две фигуры: механику покрытия (семантика precision/recall@k=3 на
сценариях same_law / mode_drop / mode_invent) и sweep по k (1/3/10) для
mode invention. Числовое ядро — tda_metrics.precision_recall_viz (numpy,
без TensorFlow); сценарии собраны поверх tda_metrics.mixtures методом
общих случайных чисел. Все артефакты пишутся атомарно: сначала в
out_dir/.tmp_run_<pid>/, затем пофайлово os.replace в out_dir и fig_dir.
Раннер отказывается перезаписывать manifest.json с чужим config_sha256
(exit 2). Печать краткого отчёта в stdout разрешена только здесь.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tda_metrics.mixtures import make_mode_reference, sample_reference
from tda_metrics.precision_recall_viz import (
    plot_precision_recall_panel,
    precision_recall_membership,
    sample_mode_drop,
    sample_mode_invent,
)

FIGURE_SEMANTICS = 'prv_precision_recall_semantics.png'
FIGURE_K_SWEEP = 'prv_precision_recall_k_sweep.png'
MEMBERSHIP_CSV = 'memberships.csv'
SUMMARY_JSON = 'summary.json'
MANIFEST_JSON = 'manifest.json'
SCENARIOS = ('same_law', 'mode_drop', 'mode_invent')


def build_config():
    """Словарь параметров прогона; его sha256 идентифицирует запуск."""
    return {
        'n_points': 480,
        'k_semantics': 3,
        'k_sweep': [1, 3, 10],
        'seeds': {'P': 42, 'Q': 7},
        'ref': {'n_modes': 8, 'radius': 8.0, 'spread': 1.0},
        'mode_drop': {'dropped_pair': [0, 4]},
        'mode_invent': {'n_replaced': 60, 'invent_offset': 14.0},
        'highlight': {'precision_center_index': 17, 'recall_center_index': 23},
        'axes_limit': 18.5,
        'atol': 1e-12,
    }


def config_digest(config):
    """sha256 от каноничного json-представления конфига."""
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def ensure_fresh_run(out_dir, digest):
    """Защита от перезаписи чужого запуска: несовпадающий digest — exit 2."""
    manifest_path = out_dir / MANIFEST_JSON
    if not manifest_path.exists():
        return
    try:
        existing = json.loads(manifest_path.read_text(encoding='utf-8'))
        existing_digest = existing.get('config_sha256')
    except (OSError, ValueError):
        existing_digest = None
    if existing_digest != digest:
        print(
            f'ОШИБКА: {manifest_path} принадлежит другому запуску '
            f'(config_sha256={existing_digest!r}, текущий={digest!r}); '
            'перезапись запрещена, укажите новый --out-dir'
        )
        sys.exit(2)


def run_scenarios(config):
    """Облака и membership@k_semantics для каждого сценария; ошибка одного не роняет остальные."""
    ref = make_mode_reference(
        n_modes=config['ref']['n_modes'],
        radius=config['ref']['radius'],
        spread=config['ref']['spread'],
    )
    n = config['n_points']
    P = sample_reference(ref, n, seed=config['seeds']['P'])
    seeds_q = config['seeds']['Q']

    def same_law():
        return sample_reference(ref, n, seed=seeds_q), None

    def mode_drop():
        return (
            sample_mode_drop(
                ref,
                n,
                seed=seeds_q,
                dropped_pair=tuple(config['mode_drop']['dropped_pair']),
            ),
            None,
        )

    def mode_invent():
        return sample_mode_invent(
            ref,
            n,
            seed=seeds_q,
            n_replaced=config['mode_invent']['n_replaced'],
            invent_offset=config['mode_invent']['invent_offset'],
        )

    builders = {'same_law': same_law, 'mode_drop': mode_drop, 'mode_invent': mode_invent}
    data = {}
    statuses = {}
    for name in SCENARIOS:
        try:
            Q, replaced_mask = builders[name]()
            membership = precision_recall_membership(
                P, Q, k=config['k_semantics'], atol=config['atol']
            )
            data[name] = {
                'P': P,
                'Q': Q,
                'replaced_mask': replaced_mask,
                'membership': membership,
            }
            statuses[name] = 'ok'
        except Exception as exc:
            statuses[name] = f'failed: {exc!r}'
    return data, statuses


def run_k_sweep(config, data):
    """Membership для mode_invent на каждом k из k_sweep (радиусы зависят от k)."""
    sweep = {}
    if 'mode_invent' not in data:
        return sweep
    entry = data['mode_invent']
    for k in config['k_sweep']:
        sweep[k] = precision_recall_membership(
            entry['P'], entry['Q'], k=k, atol=config['atol']
        )
    return sweep


def format_float(value):
    """Числа в CSV с достаточной точностью для восстановления масок."""
    return f'{value:.10g}'


def collect_membership_rows(config, data, sweep):
    """Строки memberships.csv: панели основной фигуры (k=3, 3 сценария, оба направления)
    и k-sweep (mode_invent, k из k_sweep, оба направления); панель mode_invent@k=3
    общая для обеих фигур и пишется один раз.
    """
    panels = {}
    k_main = config['k_semantics']
    for name in SCENARIOS:
        if name not in data:
            continue
        entry = data[name]
        membership = entry['membership']
        panels[(name, k_main, 'precision')] = (
            entry['P'], membership['radii_P'], entry['Q'], membership['q_covered'],
        )
        panels[(name, k_main, 'recall')] = (
            entry['Q'], membership['radii_Q'], entry['P'], membership['p_covered'],
        )
    if 'mode_invent' in data:
        entry = data['mode_invent']
        for k, membership in sweep.items():
            panels.setdefault(
                ('mode_invent', k, 'precision'),
                (entry['P'], membership['radii_P'], entry['Q'], membership['q_covered']),
            )
            panels.setdefault(
                ('mode_invent', k, 'recall'),
                (entry['Q'], membership['radii_Q'], entry['P'], membership['p_covered']),
            )
    rows = []
    for (scenario, k, direction), (centers, radii, query, covered) in panels.items():
        for i in range(centers.shape[0]):
            rows.append(
                (
                    scenario, direction, k, 'center', i,
                    format_float(centers[i, 0]), format_float(centers[i, 1]),
                    '', format_float(radii[i]),
                )
            )
        for j in range(query.shape[0]):
            rows.append(
                (
                    scenario, direction, k, 'query', j,
                    format_float(query[j, 0]), format_float(query[j, 1]),
                    str(bool(covered[j])), '',
                )
            )
    return rows


def write_memberships_csv(path, rows):
    """memberships.csv: scenario, direction, k, point_role, point_index, x, y, covered, radius."""
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ['scenario', 'direction', 'k', 'point_role', 'point_index', 'x', 'y', 'covered', 'radius']
        )
        writer.writerows(rows)


def build_semantics_figure(plt, config, data, path):
    """Фигура 1: 3 сценария × 2 направления (precision, recall) при k = k_semantics."""
    k = config['k_semantics']
    limit = config['axes_limit']
    fig, axes = plt.subplots(nrows=len(SCENARIOS), ncols=2, figsize=(14, 19))
    for row, name in enumerate(SCENARIOS):
        entry = data[name]
        membership = entry['membership']
        P, Q = entry['P'], entry['Q']
        plot_precision_recall_panel(
            axes[row, 0],
            P, membership['radii_P'], Q, membership['q_covered'], k,
            config['highlight']['precision_center_index'],
            'precision', membership['precision'], limit,
        )
        plot_precision_recall_panel(
            axes[row, 1],
            Q, membership['radii_Q'], P, membership['p_covered'], k,
            config['highlight']['recall_center_index'],
            'recall', membership['recall'], limit,
        )
    fig.suptitle(
        f'Improved precision/recall@k={k}: механика покрытия (2D-иллюстрация)',
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_k_sweep_figure(plt, config, entry, sweep, path):
    """Фигура 2: 2 направления × 3 значения k, та же пара облаков mode_invent."""
    limit = config['axes_limit']
    k_values = list(config['k_sweep'])
    fig, axes = plt.subplots(nrows=2, ncols=len(k_values), figsize=(18, 11))
    for col, k in enumerate(k_values):
        membership = sweep[k]
        plot_precision_recall_panel(
            axes[0, col],
            entry['P'], membership['radii_P'], entry['Q'], membership['q_covered'], k,
            config['highlight']['precision_center_index'],
            'precision', membership['precision'], limit,
        )
        plot_precision_recall_panel(
            axes[1, col],
            entry['Q'], membership['radii_Q'], entry['P'], membership['p_covered'], k,
            config['highlight']['recall_center_index'],
            'recall', membership['recall'], limit,
        )
    alpha = config['mode_invent']['n_replaced'] / config['n_points']
    fig.suptitle(f'Влияние k: mode invention (alpha={alpha:g})', fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Визуализация improved precision/recall@k на 2D-смесях (CPU, numpy).'
    )
    parser.add_argument('--out-dir', default='results/raw/prv', help='каталог артефактов прогона')
    parser.add_argument('--fig-dir', default='results/figures', help='каталог PNG-фигур')
    args = parser.parse_args(argv)

    config = build_config()
    digest = config_digest(config)
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    ensure_fresh_run(out_dir, digest)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    tmp_dir = out_dir / f'.tmp_run_{os.getpid()}'
    os.makedirs(tmp_dir, exist_ok=True)

    versions = {'python': platform.python_version(), 'numpy': np.__version__, 'matplotlib': None}
    plt = None
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt_module
        plt = plt_module
        versions['matplotlib'] = matplotlib.__version__
    except Exception as exc:
        print(f'matplotlib недоступен, фигуры не будут построены: {exc!r}')

    data, scenario_status = run_scenarios(config)
    sweep = run_k_sweep(config, data)

    rows = collect_membership_rows(config, data, sweep)
    write_memberships_csv(tmp_dir / MEMBERSHIP_CSV, rows)

    figure_status = {}
    figure_builders = (
        (FIGURE_SEMANTICS, lambda path: build_semantics_figure(plt, config, data, path)),
        (
            FIGURE_K_SWEEP,
            lambda path: build_k_sweep_figure(plt, config, data['mode_invent'], sweep, path),
        ),
    )
    for figure_name, builder in figure_builders:
        if plt is None:
            figure_status[figure_name] = 'failed: matplotlib недоступен'
            continue
        try:
            builder(tmp_dir / figure_name)
            figure_status[figure_name] = 'ok'
        except Exception as exc:
            figure_status[figure_name] = f'failed: {exc!r}'

    summary = {}
    for name in SCENARIOS:
        if name not in data:
            continue
        membership = data[name]['membership']
        summary[name] = {
            'precision': membership['precision'],
            'recall': membership['recall'],
            'n': config['n_points'],
            'k': config['k_semantics'],
        }
    summary['k_sweep'] = {
        str(k): {'precision': membership['precision'], 'recall': membership['recall']}
        for k, membership in sweep.items()
    }
    write_json(tmp_dir / SUMMARY_JSON, summary)

    move_errors = []
    transfers = [(tmp_dir / MEMBERSHIP_CSV, out_dir / MEMBERSHIP_CSV)]
    transfers.append((tmp_dir / SUMMARY_JSON, out_dir / SUMMARY_JSON))
    for figure_name in (FIGURE_SEMANTICS, FIGURE_K_SWEEP):
        transfers.append((tmp_dir / figure_name, fig_dir / figure_name))
    for source, target in transfers:
        if not source.exists():
            continue
        try:
            os.replace(source, target)
        except OSError as exc:
            move_errors.append(f'{source} -> {target}: {exc!r}')

    figures_ok = all(figure_status[name] == 'ok' for name in (FIGURE_SEMANTICS, FIGURE_K_SWEEP))
    status = 'ok' if figures_ok and not move_errors else 'failed'

    replaced_fraction = None
    if 'mode_invent' in data and data['mode_invent']['replaced_mask'] is not None:
        replaced_fraction = float(np.mean(data['mode_invent']['replaced_mask']))
    manifest = {
        'created': datetime.now(timezone.utc).isoformat(),
        'config': config,
        'config_sha256': digest,
        'versions': versions,
        'status': status,
        'scenarios': scenario_status,
        'figures': figure_status,
        'replaced_fraction_mode_invent': replaced_fraction,
    }
    write_json(tmp_dir / MANIFEST_JSON, manifest)
    try:
        os.replace(tmp_dir / MANIFEST_JSON, out_dir / MANIFEST_JSON)
    except OSError as exc:
        move_errors.append(f'{tmp_dir / MANIFEST_JSON} -> {out_dir / MANIFEST_JSON}: {exc!r}')
        status = 'failed'

    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f'out_dir: {out_dir}')
    print(f'fig_dir: {fig_dir}')
    print(f'config_sha256: {digest}')
    for name in SCENARIOS:
        if name in summary and scenario_status[name] == 'ok':
            print(
                f"scenario {name}: ok "
                f"(precision={summary[name]['precision']:.4f}, recall={summary[name]['recall']:.4f})"
            )
        else:
            print(f'scenario {name}: {scenario_status[name]}')
    for k, entry in summary.get('k_sweep', {}).items():
        print(f"k_sweep mode_invent k={k}: precision={entry['precision']:.4f}, recall={entry['recall']:.4f}")
    for figure_name, figure_state in figure_status.items():
        print(f'figure {figure_name}: {figure_state}')
    for error in move_errors:
        print(f'ошибка переноса: {error}')
    if replaced_fraction is not None:
        print(f'replaced_fraction_mode_invent: {replaced_fraction}')
    print(f'status: {status}')
    return 0 if status == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
