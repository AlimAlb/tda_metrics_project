"""Перегенерация фигур C2 из points.parquet: 4 отдельных графика по сравнениям.

v2 по фидбеку: вместо одной сетки 4×4 — четыре отдельные фигуры (cc, arc,
zoom, arc360; в каждой 2×2 пространства, как C4); точки P темнее
(#4d4d4d вместо lightgray) — различимы на белом фоне. Данные те же:
results/raw/c2_clouds/points.parquet + panel_metrics.csv; метрики в
заголовках идентичны артефакту a547f7d.

Запуск (нужны pandas+matplotlib): python scripts/rebuild_c2_cloud_figures.py
"""
import argparse
import os

import numpy as np
import pandas as pd

COMPS = ('cc', 'arc', 'zoom', 'arc360')
COMP_TITLES = {
    'cc': 'контроль: круг vs круг',
    'arc': 'дуга 0–180°',
    'zoom': 'масштабный свип 0.6–1.4',
    'arc360': 'дуга 180–360°',
}
SPACES = ('pixels_pca16', 'clip_pca16', 'dino_pca16', 'vae_latent16')
P_COLOR = '#4d4d4d'
P_ALPHA = 0.55
Q_ALPHA = 0.85


def render(comp, points, panels, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    is_zoom = comp == 'zoom'
    cmap = 'viridis' if is_zoom else 'hsv'
    vmin, vmax = (0.6, 1.4) if is_zoom else (0.0, 360.0)
    cbar_label = 'масштаб' if is_zoom else 'угол, градусы'

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.0))
    for ax, space in zip(axes.flat, SPACES):
        p = points[(points['space'] == space) & (points['comp'] == 'P')]
        q = points[(points['space'] == space) & (points['comp'] == comp)]
        ax.scatter(p['pc1'], p['pc2'], s=10, c=P_COLOR, alpha=P_ALPHA,
                   linewidths=0, label='P: полный круг')
        sc = ax.scatter(q['pc1'], q['pc2'], c=q['value'], cmap=cmap, s=11,
                        alpha=Q_ALPHA, linewidths=0, vmin=vmin, vmax=vmax)
        fig.colorbar(sc, ax=ax, label=cbar_label)
        panel = panels[(panels['space'] == space) & (panels['comp'] == comp)].iloc[0]
        ax.set_title(
            f"{space}\nrtd={panel['rtd']:.2f} ntd_PQ={panel['ntd_PQ']:.3f} "
            f"p@3={panel['precision@3']:.3f}", fontsize=10)
        ax.legend(fontsize=8, loc='lower left')
    fig.suptitle(
        f'C2: {COMP_TITLES[comp]}\n'
        'PCA-2 (fit на P+Q); метрики в заголовках посчитаны по протоколу C2 '
        '(PCA-16 fit на P, rtd_trials=2) на этих же облаках',
        y=0.99, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = os.path.join(out_dir, f'c2_cloud_{comp}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('фигура:', out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', default='results/raw/c2_clouds')
    parser.add_argument('--fig-dir', default='results/figures')
    args = parser.parse_args()

    points = pd.read_parquet(os.path.join(args.raw, 'points.parquet'))
    panels = pd.read_csv(os.path.join(args.raw, 'panel_metrics.csv'))
    os.makedirs(args.fig_dir, exist_ok=True)
    for comp in COMPS:
        render(comp, points, panels, args.fig_dir)
    print('REBUILD C2 FIGURES DONE')


if __name__ == '__main__':
    main()
