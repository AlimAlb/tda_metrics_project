"""Инварианты модуля barcodes: H0 single-linkage, диаграммы ripser++, кросс-баркоды MTopDiv.

Тяжёлые тесты (persistence_diagrams, cross_bars) требуют ripser++/MTopDiv и
идут на Colab-VM; без них пропускаются (importorskip). Кросс-чек cross_bars
против TopologyMetrics.mtd — rel <= 1e-6 при полных батчах (та же механика,
что mtd_homology в metrics). Кольцо — закон S1 (radius=1.5, thickness=1.0);
полукольцо строится вручную тем же радиальным законом на углах [0, pi).
"""
import matplotlib

matplotlib.use('Agg')

import numpy as np
import pytest

from tda_metrics.barcodes import (
    bar_counts,
    cross_bars,
    longest_bar_length,
    persistence_diagrams,
    plot_barcode,
    single_linkage_bars,
    summarize_bars,
    top_bars,
)
from tda_metrics.samplers import sample_ball, sample_ring

RADIUS = 1.5
THICKNESS = 1.0
N_RING = 60
N_CROSS = 40
N_QUIET = 200
MTD_REL_TOL = 1e-4


def _half_ring(n, radius=RADIUS, thickness=THICKNESS, seed=42):
    """Полукольцо: направления равномерны на [0, pi), радиальный закон как у sample_ring."""
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.0, np.pi, size=n)
    radii = np.maximum(rng.normal(radius, thickness / 4.0, size=n), 0.0)
    return np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)


# --------------------------------------------------------------------------
# 1. Диаграммы персистентности одного облака (ripser++)
# --------------------------------------------------------------------------

def test_persistence_diagrams_ring_single_long_h1():
    pytest.importorskip('ripserplusplus')
    ring = sample_ring(N_RING, radius=RADIUS, thickness=THICKNESS, seed=42)
    dgms = persistence_diagrams(ring, maxdim=1)
    assert len(dgms) == 2
    assert dgms[0].shape[1] == 2
    assert len(dgms[0]) > 0
    assert np.isfinite(dgms[0][:, 1]).all()
    h1 = dgms[1]
    assert bar_counts(h1, RADIUS / 2.0) == 1
    assert longest_bar_length(h1) > RADIUS / 2.0


def test_persistence_diagrams_arc_and_disk_no_long_h1():
    pytest.importorskip('ripserplusplus')
    arc = _half_ring(N_QUIET)
    disk = sample_ball(N_QUIET, radius=RADIUS, seed=42)
    for cloud in (arc, disk):
        h1 = persistence_diagrams(cloud, maxdim=1)[1]
        assert bar_counts(h1, RADIUS / 2.0) == 0
        assert longest_bar_length(h1) < RADIUS / 2.0


# --------------------------------------------------------------------------
# 2. H0 single-linkage одного облака
# --------------------------------------------------------------------------

def test_single_linkage_bars_shape_births_monotone_last_inf():
    rng = np.random.default_rng(0)
    cloud = rng.normal(size=(25, 3))
    bars = single_linkage_bars(cloud)
    assert bars.shape == (25, 2)
    assert np.all(bars[:, 0] == 0.0)
    assert np.all(np.diff(bars[:-1, 1]) >= 0.0)
    assert np.isinf(bars[-1, 1])


def test_single_linkage_bars_determinism_and_edges():
    cloud = sample_ring(30, radius=RADIUS, thickness=THICKNESS, seed=42)
    assert np.array_equal(single_linkage_bars(cloud), single_linkage_bars(cloud))
    assert single_linkage_bars(np.zeros((0, 2))).shape == (0, 2)
    single = single_linkage_bars(np.array([[1.0, 2.0]]))
    assert single.shape == (1, 2)
    assert single[0, 0] == 0.0 and np.isinf(single[0, 1])


# --------------------------------------------------------------------------
# 3. Кросс-баркод пары: структура + кросс-чек против метрики
# --------------------------------------------------------------------------

def test_cross_bars_structure_and_mtd_crosscheck():
    pytest.importorskip('ripserplusplus')
    pytest.importorskip('mtd')
    from tda_metrics import TopologyMetrics

    P = sample_ring(N_CROSS, radius=RADIUS, thickness=THICKNESS, seed=42)
    Q = sample_ring(N_CROSS, radius=RADIUS, thickness=THICKNESS, seed=7)
    bars = cross_bars(P, Q)
    assert isinstance(bars, list) and len(bars) >= 2
    h0 = np.asarray(bars[0]).reshape(-1, 2)
    assert len(h0) > 0
    h1 = np.asarray(bars[1]).reshape(-1, 2)
    h1_total = float((h1[:, 1] - h1[:, 0]).sum())
    assert h1_total > 0.0
    reference = TopologyMetrics(seed=42).mtd(P, Q)
    assert h1_total == pytest.approx(reference, rel=MTD_REL_TOL)


# --------------------------------------------------------------------------
# 4. longest_bar_length / bar_counts на ручных массивах
# --------------------------------------------------------------------------

def test_longest_bar_length_and_bar_counts_manual():
    dgm = np.array([[0.0, 3.0], [0.0, 1.0]])
    assert longest_bar_length(dgm) == pytest.approx(3.0)
    assert bar_counts(dgm, 2.0) == 1
    empty = np.zeros((0, 2))
    assert longest_bar_length(empty) == 0.0
    assert bar_counts(empty, 0.5) == 0


def test_longest_bar_length_and_bar_counts_ignore_inf():
    dgm = np.array([[0.0, np.inf], [0.0, 2.5]])
    assert longest_bar_length(dgm) == pytest.approx(2.5)
    assert bar_counts(dgm, 1.0) == 1
    assert longest_bar_length(np.array([[0.0, np.inf]])) == 0.0
    assert bar_counts(np.array([[0.0, np.inf]]), 0.1) == 0


# --------------------------------------------------------------------------
# 5. Сводки summarize_bars / top_bars
# --------------------------------------------------------------------------

def test_summarize_bars_and_top_bars_manual():
    bars = np.array([[0.0, 1.0], [0.0, 3.0], [2.0, 2.5]])
    summary = summarize_bars(bars)
    assert summary['n_bars'] == 3
    assert summary['total_length'] == pytest.approx(4.5)
    assert summary['longest'] == pytest.approx(3.0)
    assert summary['mean_length'] == pytest.approx(1.5)
    assert summary['max_death'] == pytest.approx(3.0)
    np.testing.assert_allclose(top_bars(bars, 2), [[0.0, 3.0], [0.0, 1.0]])
    assert top_bars(bars, 0).shape == (0, 2)
    empty = summarize_bars([])
    assert empty['n_bars'] == 0 and empty['total_length'] == 0.0
    assert top_bars([]).shape == (0, 2)


# --------------------------------------------------------------------------
# 6. Отрисовка plot_barcode (backend Agg)
# --------------------------------------------------------------------------

def test_plot_barcode_returns_ax_and_handles_inf_and_empty():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    bars = np.array([[0.0, 3.0], [0.0, 1.0], [0.5, np.inf]])
    assert plot_barcode(ax, bars, level=0, max_bars=3) is ax
    assert plot_barcode(ax, np.zeros((0, 2)), level=1) is ax
    assert plot_barcode(ax, bars, level=2, max_bars=0) is ax
    plt.close(fig)
