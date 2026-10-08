"""Инварианты sample_length_matched_pair: стратификация, независимость повторений, квоты.

Синтетический датасет одного слоя: 200 промптов x (correct, hallucinated),
длины — lognormal(mean(log)=3.8, sigma=0.35) со сдвигом hallucinated на +3%,
vectors (400, 8) выровнены по строкам meta. Требует pandas/scipy — прогон на VM.
"""
import numpy as np
import pandas as pd
import pytest

from tda_metrics.llm_experiments import sample_length_matched_pair

N_PROMPTS = 200
HIDDEN = 8
N_CLOUD = 60


@pytest.fixture(scope='module')
def dataset():
    rng = np.random.default_rng(0)
    prompt_ids = [f'p{i:03d}' for i in range(N_PROMPTS)]
    base = rng.lognormal(mean=3.8, sigma=0.35, size=(N_PROMPTS, 2))
    vectors = rng.normal(size=(2 * N_PROMPTS, HIDDEN))
    rows = []
    for i, prompt_id in enumerate(prompt_ids):
        for j, kind in enumerate(('correct', 'hallucinated')):
            length = base[i, 0] if kind == 'correct' else base[i, 1] * 1.03
            rows.append({
                'prompt_id': prompt_id,
                'answer_kind': kind,
                'layer_index': 0,
                'n_pooled': float(length),
                'answer_id': f'a{2 * i + j}',
            })
    meta = pd.DataFrame(rows)
    return meta, vectors, prompt_ids


def call(dataset, seed, n_cloud, bins=10, prompt_ids=None):
    meta, vectors, all_ids = dataset
    ids = all_ids if prompt_ids is None else prompt_ids
    return sample_length_matched_pair(meta, vectors, ids, 0, seed, n_cloud, bins=bins)


def test_determinism_same_seed(dataset):
    c1, h1, mc1, mh1, info1 = call(dataset, seed=0, n_cloud=N_CLOUD)
    c2, h2, mc2, mh2, info2 = call(dataset, seed=0, n_cloud=N_CLOUD)
    assert np.array_equal(c1, c2)
    assert np.array_equal(h1, h2)
    assert mc1.equals(mc2)
    assert mh1.equals(mh2)
    assert np.array_equal(info1['selected_correct_prompt_ids'],
                          info2['selected_correct_prompt_ids'])
    assert np.array_equal(info1['selected_hallucinated_prompt_ids'],
                          info2['selected_hallucinated_prompt_ids'])


def test_different_seeds_differ(dataset):
    _, _, mc0, mh0, info0 = call(dataset, seed=0, n_cloud=N_CLOUD)
    _, _, mc1, mh1, info1 = call(dataset, seed=1, n_cloud=N_CLOUD)
    sides_differ = (
        set(mc0['answer_id']) != set(mc1['answer_id'])
        or set(mh0['answer_id']) != set(mh1['answer_id'])
    )
    assert sides_differ
    assert info0['length_diagnostics']['ks_p'] > 0.05
    assert info1['length_diagnostics']['ks_p'] > 0.05


def test_no_replacement_within_side(dataset):
    _, _, mc, mh, info = call(dataset, seed=0, n_cloud=N_CLOUD)
    assert not mc['answer_id'].duplicated().any()
    assert not mh['answer_id'].duplicated().any()
    assert info['n_selected'] == len(mc) == len(mh)


def test_equal_bin_counts(dataset):
    _, _, mc, mh, info = call(dataset, seed=0, n_cloud=N_CLOUD)
    edges = np.asarray(info['bin_edges'], dtype=float)
    n_bins = len(edges) - 1

    def assign(frame):
        raw = np.digitize(np.asarray(frame['n_pooled'], dtype=float), edges) - 1
        return np.clip(raw, 0, n_bins - 1)

    counts_c = np.bincount(assign(mc), minlength=n_bins)
    counts_h = np.bincount(assign(mh), minlength=n_bins)
    assert np.array_equal(counts_c, counts_h)
    assert np.array_equal(counts_c, np.asarray(info['quotas']))
    assert counts_c.sum() == info['n_selected']
    assert info['n_selected'] <= N_CLOUD


def test_length_balance(dataset):
    meta, _, _ = dataset
    pooled_sd = float(np.std(np.asarray(meta['n_pooled'], dtype=float)))
    _, _, mc, mh, info = call(dataset, seed=0, n_cloud=N_CLOUD)
    diag = info['length_diagnostics']
    assert abs(diag['mean_correct'] - diag['mean_hallucinated']) < 0.15 * pooled_sd
    assert abs(diag['median_correct'] - diag['median_hallucinated']) < 0.2 * pooled_sd
    assert diag['ks_p'] > 0.05
    assert diag['mean_correct'] == pytest.approx(float(np.mean(np.asarray(mc['n_pooled'], dtype=float))))
    assert diag['mean_hallucinated'] == pytest.approx(float(np.mean(np.asarray(mh['n_pooled'], dtype=float))))


def test_group_split_preserved(dataset):
    meta, vectors, all_ids = dataset
    included = all_ids[:150]
    excluded = set(all_ids[150:])
    _, _, mc, mh, info = call(dataset, seed=0, n_cloud=N_CLOUD, prompt_ids=included)
    assert set(mc['prompt_id']) <= set(included)
    assert set(mh['prompt_id']) <= set(included)
    assert not (set(mc['prompt_id']) & excluded)
    assert not (set(mh['prompt_id']) & excluded)
    assert set(info['selected_correct_prompt_ids']) <= set(included)
    assert set(info['selected_hallucinated_prompt_ids']) <= set(included)


def test_quota_shrink_deterministic(dataset):
    meta, vectors, all_ids = dataset
    _, _, mc, mh, info = call(dataset, seed=3, n_cloud=10, bins=4)
    lengths_c = np.asarray(meta.loc[meta['answer_kind'] == 'correct', 'n_pooled'], dtype=float)
    lengths_h = np.asarray(meta.loc[meta['answer_kind'] == 'hallucinated', 'n_pooled'], dtype=float)
    edges = np.quantile(np.concatenate([lengths_c, lengths_h]), np.linspace(0.0, 1.0, 5))
    edges[-1] += 1e-9
    assert np.allclose(np.asarray(info['bin_edges']), edges)
    bins_c = np.clip(np.digitize(lengths_c, edges) - 1, 0, 3)
    bins_h = np.clip(np.digitize(lengths_h, edges) - 1, 0, 3)
    available = sum(int(min((bins_c == b).sum(), (bins_h == b).sum())) for b in range(4))
    assert sum(info['quotas']) == info['n_selected'] == min(10, available)
    assert len(mc) == len(mh) == info['n_selected']
    for b, q in enumerate(info['quotas']):
        assert q <= min((bins_c == b).sum(), (bins_h == b).sum())
    again = call(dataset, seed=3, n_cloud=10, bins=4)
    assert info['quotas'] == again[4]['quotas']
    assert info['n_selected'] == again[4]['n_selected']


def test_validation(dataset):
    meta, vectors, all_ids = dataset
    with pytest.raises(ValueError):
        call(dataset, seed=0, n_cloud=N_CLOUD, bins=0)
    with pytest.raises(ValueError):
        call(dataset, seed=0, n_cloud=1)
    with pytest.raises(ValueError):
        sample_length_matched_pair(meta, vectors, [], 0, 0, N_CLOUD)
    with pytest.raises(ValueError):
        sample_length_matched_pair(meta, vectors, all_ids, 0, 'x', N_CLOUD)


def test_independence_across_calls(dataset):
    outputs = [call(dataset, seed=s, n_cloud=N_CLOUD) for s in (0, 1, 0, 1)]
    c0, h0, mc0, mh0, _ = outputs[0]
    c1, h1, mc1, mh1, _ = outputs[1]
    c0b, h0b, mc0b, mh0b, _ = outputs[2]
    c1b, h1b, mc1b, mh1b, _ = outputs[3]
    assert np.array_equal(c0, c0b) and np.array_equal(h0, h0b)
    assert np.array_equal(c1, c1b) and np.array_equal(h1, h1b)
    assert mc0.equals(mc0b) and mh0.equals(mh0b)
    assert mc1.equals(mc1b) and mh1.equals(mh1b)
    assert not np.array_equal(c0, c1)
    assert not np.array_equal(h0, h1)
