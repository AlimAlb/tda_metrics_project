"""Инварианты embedding_cache: round-trip, CacheMismatchError (expect-поля
и согласованность manifest с массивами), атомарность, prompt_ids_hash.
Тесты чисто механические (pandas + pyarrow + numpy), heavy-стек
(torch/transformers) не нужен."""
import os

import numpy as np
import pandas as pd
import pytest

from tda_metrics.embedding_cache import (
    CacheMismatchError,
    META_NAME,
    MANIFEST_NAME,
    VECTORS_NAME,
    load_cache,
    prompt_ids_hash,
    save_cache,
)


def make_meta(n=4, n_layers=4):
    """meta в формате ``extract_embeddings``: строка на пару (пункт, слой),
    порядок пунктов x слоёв (``n`` пунктов -> ``n * n_layers`` строк)."""
    rows = [
        {
            'prompt_id': f'p{i}',
            'answer_kind': 'correct' if i % 2 == 0 else 'hallucinated',
            'layer_index': 9 * (j + 1),
            'layer_fraction': (j + 1) / n_layers,
            'pooling': 'mean_answer',
            'input_len': 10 + i,
            'n_pooled': 3,
            'truncated': False,
        }
        for i in range(n)
        for j in range(n_layers)
    ]
    return pd.DataFrame(rows)


def make_manifest(**overrides):
    base = dict(
        model='Qwen/Qwen2.5-3B-Instruct',
        model_revision='abc123',
        input_format='chat_knowledge_v1',
        template_notes='chat template, knowledge+question в user',
        dataset_sha256='deadbeef',
        n_records=2,
        n_items=4,
        answer_kinds={'correct': 2, 'hallucinated': 2},
        layer_fractions=[0.25, 0.5, 0.75, 1.0],
        layer_indices=[9, 18, 27, 36],
        pooling='mean_answer',
        dtype='float16',
        max_length=1024,
        truncation_share=0.25,
        prompt_ids_hash=prompt_ids_hash(make_meta()),
        lens={'prompt': [10, 20], 'answer': [3, 4], 'pooled': [3, 3]},
    )
    base.update(overrides)
    return base


def make_vectors(n=4, n_layers=4, hidden=8):
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, n_layers, hidden)).astype(np.float32)


def test_round_trip_meta_vectors_manifest(tmp_path):
    directory = tmp_path / 'cache'
    meta = make_meta()
    vectors = make_vectors()
    manifest = make_manifest()
    save_cache(directory, meta, vectors, manifest)

    loaded_meta, loaded_vectors, loaded_manifest = load_cache(directory)

    pd.testing.assert_frame_equal(loaded_meta, meta)
    assert np.array_equal(loaded_vectors, vectors)
    assert loaded_vectors.dtype == vectors.dtype
    assert loaded_manifest == manifest


def test_round_trip_files_present(tmp_path):
    directory = tmp_path / 'cache'
    save_cache(directory, make_meta(), make_vectors(), make_manifest())
    names = set(os.listdir(directory))
    assert names == {META_NAME, VECTORS_NAME, MANIFEST_NAME}


def test_load_cache_missing_file_raises(tmp_path):
    directory = tmp_path / 'cache'
    save_cache(directory, make_meta(), make_vectors(), make_manifest())
    os.remove(directory / VECTORS_NAME)
    with pytest.raises(FileNotFoundError):
        load_cache(directory)


@pytest.mark.parametrize('key,bad_value', [
    ('model', 'Qwen/Qwen2.5-7B-Instruct'),
    ('model_revision', 'other-revision'),
    ('input_format', 'plain_v0'),
    ('template_notes', 'другое описание шаблона'),
    ('dataset_sha256', 'feedface'),
    ('n_records', 999),
    ('n_items', 999),
    ('answer_kinds', {'correct': 1, 'hallucinated': 1}),
    ('pooling', 'last_answer_token'),
    ('dtype', 'float32'),
    ('max_length', 2048),
    ('layer_fractions', [0.5, 1.0]),
    ('layer_indices', [18, 36]),
    ('prompt_ids_hash', '0' * 64),
    ('truncation_share', 0.9),
    ('lens', {'prompt': [1], 'answer': [1], 'pooled': [1]}),
])
def test_load_cache_mismatch_per_key(tmp_path, key, bad_value):
    directory = tmp_path / 'cache'
    manifest = make_manifest()
    save_cache(directory, make_meta(), make_vectors(), manifest)

    expect = dict(manifest)
    expect[key] = bad_value

    with pytest.raises(CacheMismatchError) as excinfo:
        load_cache(directory, expect=expect)
    assert key in str(excinfo.value)


def test_load_cache_mismatch_missing_key(tmp_path):
    directory = tmp_path / 'cache'
    manifest = make_manifest()
    save_cache(directory, make_meta(), make_vectors(), manifest)

    with pytest.raises(CacheMismatchError) as excinfo:
        load_cache(directory, expect={'nonexistent_field': 'x'})
    assert 'nonexistent_field' in str(excinfo.value)


def test_load_cache_expect_matching_subset_ok(tmp_path):
    directory = tmp_path / 'cache'
    manifest = make_manifest()
    save_cache(directory, make_meta(), make_vectors(), manifest)

    loaded_meta, loaded_vectors, loaded_manifest = load_cache(
        directory,
        expect={'model': manifest['model'], 'pooling': manifest['pooling']},
    )
    assert loaded_manifest == manifest


def test_load_cache_vectors_desync_raises(tmp_path):
    directory = tmp_path / 'cache'
    save_cache(directory, make_meta(), make_vectors(), make_manifest())
    np.save(directory / VECTORS_NAME, make_vectors(n=2))

    with pytest.raises(CacheMismatchError) as excinfo:
        load_cache(directory)
    message = str(excinfo.value)
    assert 'n_items=4' in message
    assert 'vectors.shape[0]=2' in message


def test_load_cache_layer_axis_desync_raises(tmp_path):
    directory = tmp_path / 'cache'
    save_cache(directory, make_meta(), make_vectors(), make_manifest())
    np.save(directory / VECTORS_NAME, make_vectors(n_layers=2))

    with pytest.raises(CacheMismatchError) as excinfo:
        load_cache(directory)
    message = str(excinfo.value)
    assert 'слоёв' in message
    assert 'vectors.shape[1]=2' in message


def test_load_cache_meta_rows_desync_raises(tmp_path):
    directory = tmp_path / 'cache'
    save_cache(directory, make_meta(), make_vectors(), make_manifest())
    make_meta(n=2).to_parquet(directory / META_NAME)

    with pytest.raises(CacheMismatchError) as excinfo:
        load_cache(directory)
    message = str(excinfo.value)
    assert 'строк meta=8' in message
    assert 'ожидалось 16' in message


def test_load_cache_manifest_without_n_items_raises(tmp_path):
    directory = tmp_path / 'cache'
    manifest = make_manifest()
    del manifest['n_items']
    save_cache(directory, make_meta(), make_vectors(), manifest)

    with pytest.raises(CacheMismatchError) as excinfo:
        load_cache(directory)
    assert 'n_items' in str(excinfo.value)


def test_save_cache_atomic_no_leftover_tmp_files(tmp_path):
    directory = tmp_path / 'cache'
    save_cache(directory, make_meta(), make_vectors(), make_manifest())
    names = os.listdir(directory)
    assert all(not name.endswith('.tmp') for name in names)
    assert all('.tmp.' not in name for name in names)


def test_save_cache_overwrite_leaves_no_garbage(tmp_path):
    directory = tmp_path / 'cache'
    save_cache(directory, make_meta(n=2), make_vectors(n=2), make_manifest(n_items=2))
    save_cache(directory, make_meta(n=4), make_vectors(n=4), make_manifest(n_items=4))
    names = set(os.listdir(directory))
    assert names == {META_NAME, VECTORS_NAME, MANIFEST_NAME}
    _, vectors, manifest = load_cache(directory)
    assert vectors.shape[0] == 4
    assert manifest['n_items'] == 4


def test_prompt_ids_hash_stable_dataframe_vs_list():
    meta = make_meta()
    assert prompt_ids_hash(meta) == prompt_ids_hash(meta['prompt_id'].tolist())


def test_prompt_ids_hash_order_independent():
    meta = make_meta()
    shuffled = meta.iloc[::-1].reset_index(drop=True)
    assert prompt_ids_hash(meta) == prompt_ids_hash(shuffled)


def test_prompt_ids_hash_duplicate_invariant():
    meta = make_meta()
    duplicated = pd.concat([meta, meta], ignore_index=True)
    assert prompt_ids_hash(meta) == prompt_ids_hash(duplicated)


def test_prompt_ids_hash_sensitive_to_change():
    meta = make_meta()
    changed = meta.copy()
    changed.loc[0, 'prompt_id'] = 'different_prompt_id'
    assert prompt_ids_hash(meta) != prompt_ids_hash(changed)


def test_prompt_ids_hash_sensitive_to_added_id():
    meta = make_meta()
    extended = pd.concat(
        [meta, pd.DataFrame({'prompt_id': ['extra_prompt']})],
        ignore_index=True,
    )
    assert prompt_ids_hash(meta) != prompt_ids_hash(extended)


def test_prompt_ids_hash_is_hex_sha256():
    digest = prompt_ids_hash(make_meta())
    assert len(digest) == 64
    int(digest, 16)
