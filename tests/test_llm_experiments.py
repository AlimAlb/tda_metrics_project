"""Инварианты llm_experiments: сплиты, облака, смеси, контроли, CKA, конфиги.

Мок кэша: meta со строкой на (пункт, слой) в порядке пункт-мажор/слой-минор
(как extract_embeddings), vectors (n_items, n_layers, hidden); длины n_pooled
заданы явно для length-matched тестов. Всё на numpy/pandas, без GPU/сети.
"""
import numpy as np
import pandas as pd
import pytest

from tda_metrics.experiment_schema import ExperimentConfig
from tda_metrics.llm_experiments import (
    build_l3_configs,
    independent_mixture,
    layer_clouds,
    linear_cka,
    paired_clouds,
    pca_reduce,
    parse_experiment_id,
    prompt_id_split,
    prompt_ids_hash,
    shuffled_label_pair,
    length_matched_pair,
)

LAYERS = (9, 18, 27)
LAYER_FRACTIONS = (0.25, 0.5, 0.75)
HIDDEN = 8


def make_cache(n_pairs=30, length_offset_correct=2.0, seed=0):
    """Синтетический кэш: n_pairs prompt_id x 2 answer_kind x len(LAYERS) строк."""
    rng = np.random.default_rng(seed)
    rows = []
    vectors = np.zeros((n_pairs * 2, len(LAYERS), HIDDEN))
    item = 0
    for pair in range(n_pairs):
        prompt_id = f'p{pair:03d}'
        for kind in ('correct', 'hallucinated'):
            base = rng.normal(size=HIDDEN)
            for layer_pos, layer in enumerate(LAYERS):
                vectors[item, layer_pos] = base + 0.1 * layer_pos
                rows.append({
                    'prompt_id': prompt_id,
                    'answer_kind': kind,
                    'layer_index': layer,
                    'layer_fraction': LAYER_FRACTIONS[layer_pos],
                    'pooling': 'mean_answer',
                    'input_len': 50,
                    'n_pooled': int(base[0] * 10 + 50 + (length_offset_correct if kind == 'correct' else 20)),
                    'truncated': False,
                })
            item += 1
    meta = pd.DataFrame(rows)
    assert len(meta) == item * len(LAYERS)
    return meta, vectors


# ---------- сплиты ----------

def test_prompt_id_split_group_integrity_and_determinism():
    meta, _ = make_cache(n_pairs=30)
    split = prompt_id_split(meta, seed=0)
    assert set(split) == {'dev', 'val', 'test'}
    all_ids = [pid for part in split.values() for pid in part]
    assert sorted(all_ids) == sorted(pd.unique(meta['prompt_id']).tolist())
    assert not (set(split['dev']) & set(split['val']))
    assert not (set(split['dev']) & set(split['test']))
    assert not (set(split['val']) & set(split['test']))
    assert prompt_id_split(meta, seed=0) == split
    assert prompt_id_split(meta, seed=1) != split


def test_prompt_id_split_pairs_stay_together():
    meta, _ = make_cache(n_pairs=30)
    split = prompt_id_split(meta, seed=0)
    for part in split.values():
        part_set = set(part)
        counts = meta[meta['prompt_id'].isin(part_set)]['prompt_id'].value_counts()
        assert set(counts) == {len(LAYERS) * 2}


def test_prompt_id_split_invalid_fractions():
    meta, _ = make_cache(n_pairs=10)
    with pytest.raises(ValueError):
        prompt_id_split(meta, fractions=(0.5, 0.2, 0.1))
    with pytest.raises(ValueError):
        prompt_id_split(meta, fractions=(0.98, 0.01, 0.01))


def test_prompt_ids_hash_stable_and_sensitive():
    assert prompt_ids_hash(['b', 'a']) == prompt_ids_hash(['a', 'b'])
    assert prompt_ids_hash(['a', 'b']) != prompt_ids_hash(['a', 'c'])
    assert len(prompt_ids_hash(['a'])) == 64


# ---------- облака ----------

def test_layer_clouds_filters_and_shape():
    meta, vectors = make_cache(n_pairs=12)
    split = prompt_id_split(meta, seed=0)
    cloud = layer_clouds(meta, vectors, split['dev'], 'correct', 18)
    expected = len(split['dev'])
    assert cloud.shape == (expected, HIDDEN)
    with pytest.raises(ValueError):
        layer_clouds(meta, vectors, ['nonexistent'], 'correct', 18)


def test_layer_clouds_rows_correspond_to_meta():
    meta, vectors = make_cache(n_pairs=6, seed=3)
    all_ids = sorted(pd.unique(meta['prompt_id']).tolist())
    cloud = layer_clouds(meta, vectors, all_ids, 'hallucinated', LAYERS[-1])
    mask = (meta['answer_kind'] == 'hallucinated') & (meta['layer_index'] == LAYERS[-1])
    expected_rows = np.flatnonzero(mask.to_numpy())
    expected = vectors[expected_rows // len(LAYERS), expected_rows % len(LAYERS)]
    assert np.array_equal(cloud, expected)


def test_pca_reduce_reference_invariance():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(500, 32))
    target_a = rng.normal(size=(100, 32))
    target_b = rng.normal(size=(70, 32))
    proj_a = pca_reduce(reference, target_a, n_components=5)
    proj_b1 = pca_reduce(reference, target_b, n_components=5)
    proj_b2 = pca_reduce(reference, target_b, n_components=5)
    assert proj_a.shape == (100, 5)
    assert np.array_equal(proj_b1, proj_b2)
    changed = np.concatenate([target_b, rng.normal(size=(10, 32))])
    proj_changed = pca_reduce(reference, changed, n_components=5)[:70]
    assert np.allclose(proj_b1, proj_changed, rtol=1e-9)


# ---------- смеси ----------

def test_independent_mixture_nested_and_extremes():
    rng = np.random.default_rng(0)
    correct = rng.normal(size=(60, 4))
    hallucinated = rng.normal(size=(60, 4)) + 10.0
    permutation = rng.permutation(60)
    zero = independent_mixture(correct, hallucinated, 0.0, permutation)
    one = independent_mixture(correct, hallucinated, 1.0, permutation)
    small = independent_mixture(correct, hallucinated, 0.2, permutation)
    big = independent_mixture(correct, hallucinated, 0.5, permutation)
    assert np.array_equal(zero, correct)
    assert np.array_equal(one, hallucinated)
    replaced_small = np.any(np.isclose(small, 10.0, atol=2).all(axis=1), axis=0)
    assert replaced_small
    diff_small = (np.abs(small - correct) > 1e-6).any(axis=1).sum()
    diff_big = (np.abs(big - correct) > 1e-6).any(axis=1).sum()
    assert diff_big >= diff_small
    assert diff_small == 12


def test_paired_clouds_requires_equal_shapes():
    correct = np.zeros((10, 4))
    hallucinated = np.zeros((12, 4))
    with pytest.raises(ValueError):
        paired_clouds(correct, hallucinated, 0.5, np.arange(12))


def test_paired_clouds_matches_independent_mixture():
    rng = np.random.default_rng(1)
    correct = rng.normal(size=(40, 3))
    hallucinated = rng.normal(size=(40, 3)) + 5.0
    permutation = rng.permutation(40)
    assert np.array_equal(
        paired_clouds(correct, hallucinated, 0.25, permutation),
        independent_mixture(correct, hallucinated, 0.25, permutation),
    )


# ---------- length-matched ----------

def test_length_matched_pair_aligns_length_distributions():
    meta, vectors = make_cache(n_pairs=40, length_offset_correct=2.0, seed=5)
    all_ids = sorted(pd.unique(meta['prompt_id']).tolist())
    correct_m, hallucinated_m, meta_c, meta_h = length_matched_pair(
        meta, vectors, all_ids, LAYERS[1])
    assert correct_m.shape == hallucinated_m.shape
    assert len(meta_c) == len(correct_m)
    lengths_c = meta_c['n_pooled'].to_numpy(dtype=float)
    lengths_h = meta_h['n_pooled'].to_numpy(dtype=float)
    quantiles = (0.1, 0.25, 0.5, 0.75, 0.9)
    assert np.allclose(np.quantile(lengths_c, quantiles),
                       np.quantile(lengths_h, quantiles), rtol=0.15)
    assert len(correct_m) == len(meta_c) < 40


def test_length_matched_pair_deterministic_and_capped():
    meta, vectors = make_cache(n_pairs=30, seed=2)
    all_ids = sorted(pd.unique(meta['prompt_id']).tolist())
    result = length_matched_pair(meta, vectors, all_ids, LAYERS[0])
    result_again = length_matched_pair(meta, vectors, all_ids, LAYERS[0])
    assert np.array_equal(result[0], result_again[0])
    capped = length_matched_pair(meta, vectors, all_ids, LAYERS[0], n_max=7)
    assert capped[0].shape[0] == 7 and capped[1].shape[0] == 7


# ---------- shuffled-label ----------

def test_shuffled_label_pair_composition_and_determinism():
    rng = np.random.default_rng(0)
    joint = rng.normal(size=(50, 3))
    labels = np.array([0] * 30 + [1] * 20)
    p, q = shuffled_label_pair(joint, labels, seed=0)
    assert p.shape == (30, 3) and q.shape == (20, 3)
    assert np.array_equal(np.sort(np.vstack([p, q]), axis=0),
                          np.sort(joint, axis=0))
    p2, q2 = shuffled_label_pair(joint, labels, seed=0)
    assert np.array_equal(p, p2) and np.array_equal(q, q2)
    with pytest.raises(ValueError):
        shuffled_label_pair(joint, labels[:10])
    with pytest.raises(ValueError):
        shuffled_label_pair(joint, np.array([0, 2] * 25))


# ---------- CKA ----------

def test_linear_cka_identity_and_sensitivity():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 6))
    assert linear_cka(X, X) == pytest.approx(1.0)
    Y = rng.normal(size=(100, 6))
    assert linear_cka(X, Y) < 1.0
    assert linear_cka(X, X + 5.0) == pytest.approx(linear_cka(X, X))
    assert linear_cka(X, Y) == pytest.approx(linear_cka(Y, X))
    with pytest.raises(ValueError):
        linear_cka(X, Y[:50])
    with pytest.raises(ValueError):
        linear_cka(np.zeros((10, 3)), np.zeros((10, 3)))


# ---------- конфиги ----------

def test_build_l3_configs_counts_fields_uniqueness():
    layers = (9, 18)
    alphas = (0.25, 0.5)
    repeats = 2
    configs = build_l3_configs(layers, alphas, repeats, n_cloud=500)
    per_layer = 3 * repeats + 2 * len(alphas) * repeats
    assert len(configs) == len(layers) * per_layer
    assert len({c.key() for c in configs}) == len(configs)
    direct = [c for c in configs if c.experiment_family == 'l3_direct'][0]
    assert direct.dataset == 'halueval_qa'
    assert direct.layer in layers
    assert direct.pooling == 'mean_answer'
    assert direct.pca_dim is None
    assert direct.alpha is None
    assert direct.split_id == 'dev'
    assert direct.metric_params is None
    mixture = [c for c in configs if c.experiment_family == 'l4b_paired'][0]
    assert mixture.alpha in alphas
    assert mixture.n_P == mixture.n_Q == 500


def test_build_l3_configs_parse_round_trip():
    layers = (9, 27)
    alphas = (0.1, 0.75, 1.0)
    repeats = 2
    configs = build_l3_configs(layers, alphas, repeats)
    for config in configs:
        parsed = parse_experiment_id(config.experiment_id)
        assert parsed['layer'] in layers
        if parsed['family'] in ('l4a', 'l4b'):
            assert parsed['alpha'] in alphas
            assert config.alpha == parsed['alpha']
        assert parsed['repeat'] == config.seed


@pytest.mark.parametrize('bad', ['l4/9/a0.5/r0', 'l3/9/direct/x', 'l3/9/invert/r0',
                                 'l4a/9/x0.5/r0', 'l4a/9/a0.5/x', 'l3d/9/r', ''])
def test_parse_experiment_id_rejects_garbage(bad):
    with pytest.raises(ValueError):
        parse_experiment_id(bad)


def test_configs_are_experiment_schema_instances():
    configs = build_l3_configs((9,), (0.5,), 1)
    assert all(isinstance(c, ExperimentConfig) for c in configs)
