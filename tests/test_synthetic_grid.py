"""Инварианты билдеров синтетических экспериментов S0/S1/S3/S4."""
import numpy as np
import pytest

from tda_metrics.experiment_schema import ExperimentConfig
from tda_metrics.synthetic_grid import (
    METRIC_PARAMS,
    build_clouds,
    build_s0_configs,
    build_s1_configs,
    build_s1_samelaw_configs,
    build_s3_configs,
    build_s4_configs,
    make_runner,
    parse_experiment_id,
)


def _unique_keys(configs):
    keys = [c.key() for c in configs]
    assert len(keys) == len(set(keys))
    return keys


def test_s0_config_count_and_fields():
    configs = build_s0_configs()
    assert len(configs) == 3 * 3 * 2 * 5
    _unique_keys(configs)
    for config in configs:
        assert config.metric_params == METRIC_PARAMS
        assert config.experiment_family == 'synthetic'
        assert config.n_P == config.n_Q
        assert config.alpha is None and config.drop_fraction is None


def test_s0_copy_clouds_identical_samelaw_different():
    copy_config = [c for c in build_s0_configs() if '_copy_' in c.experiment_id][0]
    p, q = build_clouds(copy_config)
    np.testing.assert_array_equal(p, q)
    samelaw_config = [
        c for c in build_s0_configs()
        if '_samelaw_' in c.experiment_id and c.experiment_id.startswith('s0_gaussian_samelaw_n250')
    ][0]
    p, q = build_clouds(samelaw_config)
    assert not np.array_equal(p, q)
    assert p.shape == q.shape == (250, 2)


def test_s1_config_count():
    configs = build_s1_configs()
    assert len(configs) == (7 + 7 + 2 + 7 + 7 + 7) * 5
    _unique_keys(configs)


def test_s1_samelaw_count():
    assert len(build_s1_samelaw_configs()) == 6 * 5


def test_s1_zero_level_matches_samelaw_clouds():
    zero_shift = ExperimentConfig(
        experiment_id='s1_shift_l0.0_k0', experiment_family='synthetic',
        dataset='ring', n_P=1000, n_Q=1000, seed=0,
    )
    samelaw_shift = ExperimentConfig(
        experiment_id='s1samelaw_shift_k0', experiment_family='synthetic',
        dataset='ring', n_P=1000, n_Q=1000, seed=0,
    )
    p1, q1 = build_clouds(zero_shift)
    p2, q2 = build_clouds(samelaw_shift)
    np.testing.assert_array_equal(p1, p2)
    np.testing.assert_array_equal(q1, q2)


def test_s1_thickness_changes_cloud():
    base = ExperimentConfig(
        experiment_id='s1_thickness_l1.0_k0', experiment_family='synthetic',
        dataset='ring', n_P=1000, n_Q=1000, seed=0,
    )
    thick = ExperimentConfig(
        experiment_id='s1_thickness_l1.75_k0', experiment_family='synthetic',
        dataset='ring', n_P=1000, n_Q=1000, seed=0,
    )
    _, q_base = build_clouds(base)
    _, q_thick = build_clouds(thick)
    radii_base = np.linalg.norm(q_base, axis=1)
    radii_thick = np.linalg.norm(q_thick, axis=1)
    assert radii_thick.std() > radii_base.std()


def test_s3_config_count_and_grid_fields():
    configs = build_s3_configs()
    assert len(configs) == 4 * 4 * 10
    _unique_keys(configs)
    by_eid = {c.experiment_id: c for c in configs}
    corner = by_eid['s3_d0.75_i0.75_k3']
    assert corner.drop_fraction == 0.75 and corner.invent_fraction == 0.75
    center = by_eid['s3_d0.0_i0.0_k7']
    assert center.drop_fraction == 0.0 and center.invent_fraction == 0.0


def test_s3_clouds_constant_size():
    config = [c for c in build_s3_configs() if c.experiment_id == 's3_d0.75_i0.75_k0'][0]
    p, q = build_clouds(config)
    assert p.shape == q.shape == (1000, 2)


def test_s4_config_count_and_axes():
    configs = build_s4_configs()
    assert len(configs) == 8 * 2 * 10
    _unique_keys(configs)
    drop = [c for c in configs if c.experiment_id == 's4_drop_l0.15_k0'][0]
    assert drop.drop_fraction == 0.15 and drop.invent_fraction == 0.0
    invent = [c for c in configs if c.experiment_id == 's4_invent_l0.15_k0'][0]
    assert invent.drop_fraction == 0.0 and invent.invent_fraction == 0.15


def test_all_configs_share_metric_params_and_unique_keys():
    combined = (
        build_s0_configs() + build_s1_configs()
        + build_s1_samelaw_configs() + build_s3_configs() + build_s4_configs()
    )
    assert all(c.metric_params == METRIC_PARAMS for c in combined)
    keys = [c.key() for c in combined]
    assert len(keys) == len(set(keys))


def test_build_clouds_deterministic():
    config = [c for c in build_s3_configs() if c.experiment_id == 's3_d0.25_i0.5_k2'][0]
    p1, q1 = build_clouds(config)
    p2, q2 = build_clouds(config)
    np.testing.assert_array_equal(p1, p2)
    np.testing.assert_array_equal(q1, q2)


def test_parse_experiment_id_round_trip():
    parsed = parse_experiment_id('s0_gaussian_copy_n250_k3')
    assert parsed == {
        'experiment': 's0', 'distribution': 'gaussian', 'mode': 'copy',
        'n': 250, 'repeat': 3,
    }
    parsed = parse_experiment_id('s1_rotation_l15.0_k4')
    assert parsed['transformation'] == 'rotation'
    assert parsed['level'] == 15.0 and parsed['repeat'] == 4
    parsed = parse_experiment_id('s1_reflection_l1_k0')
    assert parsed['level'] == 1 and isinstance(parsed['level'], int)
    parsed = parse_experiment_id('s1samelaw_anisotropy_k1')
    assert parsed['transformation'] == 'anisotropy' and parsed['repeat'] == 1
    parsed = parse_experiment_id('s3_d0.25_i0.75_k9')
    assert parsed['drop'] == 0.25 and parsed['invent'] == 0.75 and parsed['repeat'] == 9
    parsed = parse_experiment_id('s4_invent_l0.05_k7')
    assert parsed['axis'] == 'invent' and parsed['level'] == 0.05


def test_parse_rejects_unknown():
    with pytest.raises(ValueError):
        parse_experiment_id('s9_unknown_k0')


def test_make_runner_with_stub_metrics():
    config = [c for c in build_s0_configs() if c.experiment_id == 's0_ring_samelaw_n500_k1'][0]
    calls = []

    def metrics_fn(p, q):
        calls.append((p.shape, q.shape))
        return {'mmd': 0.0}

    runner = make_runner(metrics_fn)
    assert runner(config) == {'mmd': 0.0}
    assert runner(config) == {'mmd': 0.0}
    assert len(calls) == 2
    assert calls[0] == ((500, 2), (500, 2))


def test_build_clouds_seed_convention_p_never_equals_q_seed_source():
    samelaw = ExperimentConfig(
        experiment_id='s0_gaussian_samelaw_n250_k2', experiment_family='synthetic',
        dataset='gaussian', n_P=250, n_Q=250, seed=2,
    )
    p, q = build_clouds(samelaw)
    assert not np.array_equal(p, q)