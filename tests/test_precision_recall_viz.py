"""Инварианты прозрачной 2D-геометрии improved precision/recall@k и CPU-раннера визуализации."""
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tda_metrics.mixtures import make_mode_reference
from tda_metrics.precision_recall_viz import (
    coverage_mask,
    knn_radii,
    precision_recall_membership,
    sample_mode_drop,
    sample_mode_invent,
)

SQUARE = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
REPO_ROOT = Path(__file__).resolve().parents[1]

requires_matplotlib = pytest.mark.skipif(
    importlib.util.find_spec('matplotlib') is None,
    reason='matplotlib не установлен',
)
requires_precision_recall = pytest.mark.skipif(
    importlib.util.find_spec('precision_recall') is None,
    reason='precision_recall не установлен',
)


def test_knn_radii_excludes_self_and_validates():
    """k-й ближайший ДРУГОЙ: у угла квадрата два соседа на расстоянии 1, диагональ — только при k=3."""
    np.testing.assert_allclose(knn_radii(SQUARE, 1), np.ones(4), rtol=1e-12)
    np.testing.assert_allclose(knn_radii(SQUARE, 2), np.ones(4), rtol=1e-12)
    np.testing.assert_allclose(knn_radii(SQUARE, 3), np.full(4, np.sqrt(2.0)), rtol=1e-12)
    with pytest.raises(ValueError):
        knn_radii(SQUARE, 0)
    with pytest.raises(ValueError):
        knn_radii(SQUARE, len(SQUARE))
    with pytest.raises(ValueError):
        knn_radii(SQUARE, -1)
    with pytest.raises(ValueError):
        knn_radii(np.zeros(4), 1)
    with pytest.raises(ValueError):
        knn_radii(np.zeros((2, 2, 2)), 1)


def test_coverage_mask_hit_miss_boundary():
    centers = [[0.0, 0.0], [5.0, 0.0]]
    radii = [1.0, 2.0]
    query = [[0.5, 0.0], [2.0, 0.0], [3.0, 0.0], [5.0 - 2.0, 0.0], [0.0, 2.0]]
    np.testing.assert_array_equal(
        coverage_mask(centers, radii, query),
        [True, False, True, True, False],
    )


def test_membership_handcrafted():
    P = SQUARE
    Q = np.array([[0.2, 0.2], [0.8, 0.8], [5.0, 5.0]])
    result = precision_recall_membership(P, Q, 1)

    dist_pp = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    np.fill_diagonal(dist_pp, np.inf)
    radii_p_expected = np.sort(dist_pp, axis=1)[:, 0]
    dist_qq = np.linalg.norm(Q[:, None, :] - Q[None, :, :], axis=2)
    np.fill_diagonal(dist_qq, np.inf)
    radii_q_expected = np.sort(dist_qq, axis=1)[:, 0]
    np.testing.assert_allclose(radii_p_expected, np.ones(4), rtol=1e-12)
    np.testing.assert_allclose(radii_q_expected[0], np.sqrt(0.72), rtol=1e-12)
    np.testing.assert_allclose(radii_q_expected[1], np.sqrt(0.72), rtol=1e-12)
    np.testing.assert_allclose(radii_q_expected[2], np.sqrt(4.2 ** 2 + 4.2 ** 2), rtol=1e-12)

    dist_qp = np.linalg.norm(Q[:, None, :] - P[None, :, :], axis=2)
    q_covered_expected = np.any(dist_qp <= radii_p_expected[None, :], axis=1)
    dist_pq = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=2)
    p_covered_expected = np.any(dist_pq <= radii_q_expected[None, :], axis=1)
    np.testing.assert_array_equal(q_covered_expected, [True, True, False])
    np.testing.assert_array_equal(p_covered_expected, [True, True, True, True])

    assert result['k'] == 1
    np.testing.assert_array_equal(result['q_covered'], q_covered_expected)
    np.testing.assert_array_equal(result['p_covered'], p_covered_expected)
    assert result['precision'] == float(np.mean(q_covered_expected))
    assert result['recall'] == float(np.mean(p_covered_expected))
    assert result['precision'] == float(np.mean(result['q_covered']))
    assert result['recall'] == float(np.mean(result['p_covered']))


def test_mirror_symmetry():
    P = np.random.default_rng(0).normal(size=(30, 2))
    Q = np.random.default_rng(1).normal(size=(25, 2))
    forward = precision_recall_membership(P, Q, 3)
    backward = precision_recall_membership(Q, P, 3)
    assert forward['precision'] == backward['recall']
    assert forward['recall'] == backward['precision']


@requires_precision_recall
def test_parity_with_library():
    P = np.random.default_rng(0).normal(size=(120, 2))
    Q = np.random.default_rng(1).normal(size=(110, 2))
    try:
        import tensorflow.compat.v1 as tf
    except ImportError:
        import tensorflow as tf
    from precision_recall import knn_precision_recall_features

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config):
        state = knn_precision_recall_features(
            P,
            Q,
            nhood_sizes=[1, 3, 10],
            row_batch_size=10000,
            col_batch_size=50000,
            num_gpus=1,
        )
    tf.reset_default_graph()
    for i, k in enumerate([1, 3, 10]):
        ours = precision_recall_membership(P, Q, k)
        assert ours['precision'] == pytest.approx(float(state['precision'][i]), abs=1e-6)
        assert ours['recall'] == pytest.approx(float(state['recall'][i]), abs=1e-6)


@requires_matplotlib
def test_runner_smoke(tmp_path):
    raw_dir = tmp_path / 'raw'
    figs_dir = tmp_path / 'figs'
    result = subprocess.run(
        [
            sys.executable,
            'scripts/run_precision_recall_visualization.py',
            '--out-dir',
            str(raw_dir),
            '--fig-dir',
            str(figs_dir),
        ],
        cwd=REPO_ROOT,
        timeout=300,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f'stdout:\n{result.stdout}\nstderr:\n{result.stderr}'

    fig_semantics = figs_dir / 'prv_precision_recall_semantics.png'
    fig_sweep = figs_dir / 'prv_precision_recall_k_sweep.png'
    assert fig_semantics.stat().st_size > 10_000
    assert fig_sweep.stat().st_size > 10_000

    manifest = json.loads((raw_dir / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status'] == 'ok'
    assert manifest['config_sha256']
    assert manifest['versions']['python']
    assert manifest['versions']['numpy']
    assert manifest['versions']['matplotlib']

    with (raw_dir / 'memberships.csv').open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert {'same_law', 'mode_drop', 'mode_invent'} <= {row['scenario'] for row in rows}
    assert {'center', 'query'} <= {row['point_role'] for row in rows}
    k_values = {row['k'] for row in rows}
    assert '3' in k_values
    assert {'1', '10'} <= k_values

    summary = json.loads((raw_dir / 'summary.json').read_text(encoding='utf-8'))
    for scenario in ('same_law', 'mode_drop', 'mode_invent'):
        assert 0.0 <= summary[scenario]['precision'] <= 1.0
        assert 0.0 <= summary[scenario]['recall'] <= 1.0
    for entry in summary['k_sweep'].values():
        assert 0.0 <= entry['precision'] <= 1.0
        assert 0.0 <= entry['recall'] <= 1.0

    covered = [
        row['covered'] == 'True'
        for row in rows
        if row['scenario'] == 'mode_drop'
        and row['direction'] == 'precision'
        and row['k'] == '3'
        and row['point_role'] == 'query'
    ]
    assert covered
    assert summary['mode_drop']['precision'] == pytest.approx(float(np.mean(covered)), rel=1e-9)


def test_samplers():
    ref = make_mode_reference(8, radius=8.0, spread=1.0)
    dropped_centers = ref.centers[[0, 4]]

    q_drop = sample_mode_drop(ref, 480, 7)
    assert q_drop.shape == (480, 2)
    assert np.allclose(np.mean(q_drop, axis=0), 0.0, atol=0.8)
    dist_dropped = np.linalg.norm(q_drop[:, None, :] - dropped_centers[None, :, :], axis=2)
    assert dist_dropped.min() > 3.0

    q_invent, replaced = sample_mode_invent(ref, 480, 7, 60)
    assert q_invent.shape == (480, 2)
    assert int(replaced.sum()) == 60
    external = np.array([[14.0, 0.0], [-14.0, 0.0]])
    dist_external = np.linalg.norm(q_invent[replaced][:, None, :] - external[None, :, :], axis=2)
    assert dist_external.min(axis=1).max() < 5.0
    dist_reference = np.linalg.norm(q_invent[~replaced][:, None, :] - ref.centers[None, :, :], axis=2)
    assert dist_reference.min(axis=1).max() < 5.0

    np.testing.assert_array_equal(q_drop, sample_mode_drop(ref, 480, 7))
    q_invent_repeat, replaced_repeat = sample_mode_invent(ref, 480, 7, 60)
    np.testing.assert_array_equal(q_invent, q_invent_repeat)
    np.testing.assert_array_equal(replaced, replaced_repeat)
    assert not np.array_equal(q_drop, sample_mode_drop(ref, 480, 8))
    assert not np.array_equal(q_invent, sample_mode_invent(ref, 480, 8, 60)[0])

    with pytest.raises(ValueError):
        sample_mode_invent(ref, 480, 7, -1)
    with pytest.raises(ValueError):
        sample_mode_invent(ref, 480, 7, 481)
    with pytest.raises(ValueError):
        sample_mode_invent(ref, 480, 7, 60, invent_offset=8.0)
