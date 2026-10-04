"""Инварианты experiment_schema: round-trip, resume, дубликаты ключей,
изоляция ошибок, manifest mismatch. Тесты чисто механические: heavy-стек
(mtd/rtd/PR) не нужен, достаточно pandas + pyarrow."""
import json
from dataclasses import fields as dataclass_fields

import pandas as pd
import pytest

from tda_metrics.experiment_schema import (
    DuplicateKeyError,
    ExperimentConfig,
    ManifestMismatchError,
    ResultRow,
    ResultStore,
    ResultStoreError,
    ROW_COLUMNS,
    failed_row,
    row_from_compute_all,
    run_grid,
)


def make_config(**overrides):
    base = dict(
        experiment_id='sweep_alpha',
        experiment_family='synthetic',
        dataset='ring',
        model='toy',
        representation='raw',
        n_P=900,
        n_Q=900,
        seed=42,
    )
    base.update(overrides)
    return ExperimentConfig(**base)


def compute_all_stub(config):
    return {
        'mtd_PQ': 1.0, 'mtd_QP': 2.0, 'mmd': 0.1, 'frechet': 0.2, 'js': 0.3,
        'rtd': 35.0, 'ntd_PQ': 1.1, 'ntd_QP': 1.2,
        'precision@1': 0.9, 'recall@1': 0.8,
        'precision@3': 0.7, 'recall@3': 0.6,
        'precision@10': 0.5, 'recall@10': 0.4,
    }


def assert_rows_equal(left, right):
    for f in dataclass_fields(ResultRow):
        va, vb = getattr(left, f.name), getattr(right, f.name)
        if isinstance(va, float) or isinstance(vb, float):
            assert va == pytest.approx(vb), f.name
        else:
            assert va == vb, f.name


@pytest.mark.parametrize('fmt', ['parquet', 'csv'])
def test_round_trip(tmp_path, fmt):
    store = ResultStore(tmp_path / 'store', format=fmt)
    config = make_config(alpha=0.25, split_id='dev', layer=6, pca_dim=16)
    row = row_from_compute_all(config, compute_all_stub(config), runtime_seconds=1.5)
    key = store.append(row)
    assert key == config.key()

    reopened = ResultStore(tmp_path / 'store')
    assert len(reopened) == 1
    frame = reopened.load_frame()
    assert set(ROW_COLUMNS) <= set(frame.columns)
    assert_rows_equal(reopened.rows()[0], row)


def test_round_trip_preserves_failed_row(tmp_path):
    store = ResultStore(tmp_path / 'store')
    config = make_config(alpha=0.5, corruption='gaussian_noise', severity=3)
    store.append(row_from_compute_all(config, compute_all_stub(config)))
    store.append(row_from_compute_all(make_config(alpha=0.75), {'mmd': 0.4, 'rtd': 30.0}))
    failed_config = make_config(alpha=0.9, corruption='gaussian_noise', severity=3)
    store.append(failed_row(failed_config, 'RuntimeError: boom', runtime_seconds=0.25))

    reopened = ResultStore(tmp_path / 'store')
    rows = reopened.rows()
    assert len(rows) == 3
    statuses = sorted(r.status for r in rows)
    assert statuses == ['failed', 'ok', 'ok']
    failed = [r for r in rows if r.status == 'failed'][0]
    assert failed.error_message == 'RuntimeError: boom'
    assert failed.mmd is None
    assert failed.corruption == 'gaussian_noise'
    assert failed.severity == 3


def test_row_from_compute_all_mapping_and_unknown_keys():
    config = make_config()
    row = row_from_compute_all(config, compute_all_stub(config), runtime_seconds=0.5)
    assert row.precision_3 == pytest.approx(0.7)
    assert row.recall_10 == pytest.approx(0.4)
    assert row.ntd_QP == pytest.approx(1.2)
    assert row.status == 'ok'
    assert row.error_message is None
    assert row.n_P == config.n_P and row.seed == config.seed

    partial = {'mmd': 0.1, 'precision@1': 0.9}
    row2 = row_from_compute_all(config, partial)
    assert row2.mmd == pytest.approx(0.1)
    assert row2.precision_1 == pytest.approx(0.9)
    assert row2.rtd is None and row2.ntd_PQ is None and row2.recall_3 is None

    with pytest.raises(ValueError):
        row_from_compute_all(config, {'cka': 0.5})


def test_config_key_stable_and_sensitive():
    c1 = make_config(alpha=0.1)
    assert c1.key() == make_config(alpha=0.1).key()
    assert c1.key() != make_config(alpha=0.2).key()
    assert c1.key() != make_config(seed=7).key()
    assert c1.key() != make_config(n_P=1000).key()
    assert c1.key() != make_config(experiment_id='other').key()
    row = row_from_compute_all(c1, {'mmd': 0.0})
    assert row.config_key() == c1.key()
    assert row.config() == c1


def test_duplicate_key_detection(tmp_path):
    store = ResultStore(tmp_path / 'store')
    config = make_config(alpha=0.3)
    store.append(row_from_compute_all(config, {'mmd': 0.1}))
    with pytest.raises(DuplicateKeyError):
        store.append(row_from_compute_all(config, {'mmd': 0.2}))
    assert len(store) == 1
    assert store.contains(config)
    assert not store.contains(make_config(alpha=0.4))


def test_resume_after_interrupted_grid(tmp_path):
    store = ResultStore(tmp_path / 'store')
    configs = [make_config(alpha=i / 5) for i in range(5)]
    calls = []

    def runner(config):
        calls.append(config.alpha)
        return {'mmd': float(config.alpha)}

    run_grid(store, configs[:2], runner)
    assert len(store) == 2
    written = run_grid(store, configs, runner)
    assert len(written) == 3
    assert len(calls) == 5
    assert len(store) == 5

    rerun = run_grid(store, configs, runner)
    assert rerun == []
    assert len(calls) == 5
    assert len(store) == 5
    alphas = sorted(r.alpha for r in store.rows())
    assert alphas == pytest.approx(sorted(i / 5 for i in range(5)))


def test_failed_config_isolation(tmp_path):
    store = ResultStore(tmp_path / 'store')
    configs = [make_config(alpha=i / 5) for i in range(5)]

    def runner(config):
        if config.alpha == 0.4:
            raise RuntimeError('искусственный сбой')
        return {'mmd': float(config.alpha), 'rtd': 30.0}

    written = run_grid(store, configs, runner)
    assert len(written) == 5
    assert len(store) == 5
    failed = [r for r in written if r.status == 'failed']
    assert len(failed) == 1
    assert failed[0].alpha == pytest.approx(0.4)
    assert 'RuntimeError' in failed[0].error_message
    assert failed[0].mmd is None and failed[0].runtime_seconds is not None
    assert len(store.failed_keys()) == 1
    assert all(r.status == 'ok' and r.rtd == pytest.approx(30.0)
               for r in written if r.alpha != 0.4)
    frame = store.load_frame()
    assert set(frame['status']) == {'ok', 'failed'}


def test_retry_failed_configs(tmp_path):
    store = ResultStore(tmp_path / 'store')
    configs = [make_config(alpha=0.1), make_config(alpha=0.4)]

    def failing(config):
        if config.alpha == 0.4:
            raise RuntimeError('сбой')
        return {'mmd': 1.0}

    def healed(config):
        return {'mmd': 2.0}

    run_grid(store, configs, failing)
    assert len(store.failed_keys()) == 1
    written = run_grid(store, configs, healed, retry_failed=True)
    assert len(written) == 1
    assert written[0].status == 'ok' and written[0].mmd == pytest.approx(2.0)
    assert len(store) == 2
    assert store.failed_keys() == []
    statuses = {r.config_key(): r.status for r in store.rows()}
    assert set(statuses.values()) == {'ok'}


def test_manifest_mismatch_detection(tmp_path):
    directory = tmp_path / 'store'
    store = ResultStore(directory)
    configs = [make_config(alpha=0.1), make_config(alpha=0.2)]
    run_grid(store, configs, lambda c: {'mmd': 0.5})
    manifest_path = directory / 'manifest.json'
    data_path = directory / 'results.parquet'

    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['rows'] = {}
    manifest['n_rows'] = 0
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ManifestMismatchError):
        ResultStore(directory)
    ResultStore.rebuild_manifest(directory)
    assert len(ResultStore(directory)) == 2

    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['n_rows'] = 99
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ManifestMismatchError):
        ResultStore(directory)
    ResultStore.rebuild_manifest(directory)
    assert len(ResultStore(directory)) == 2

    frame = pd.read_parquet(data_path)
    frame.iloc[1:].to_parquet(data_path, index=False)
    with pytest.raises(ManifestMismatchError):
        ResultStore(directory)
    ResultStore.rebuild_manifest(directory)
    assert len(ResultStore(directory)) == 1

    manifest_path.unlink()
    with pytest.raises(ManifestMismatchError):
        ResultStore(directory)


def test_rebuild_manifest_recovers(tmp_path):
    directory = tmp_path / 'store'
    store = ResultStore(directory)
    configs = [make_config(alpha=0.1), make_config(alpha=0.2)]
    run_grid(store, configs, lambda c: {'mmd': 0.5})

    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    first_key = configs[0].key()
    manifest['rows'] = {k: v for k, v in manifest['rows'].items() if k != first_key}
    manifest['n_rows'] = len(manifest['rows'])
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ManifestMismatchError):
        ResultStore(directory)

    store = ResultStore.rebuild_manifest(directory)
    assert len(store) == 2
    assert store.row_status(first_key) == 'ok'
    assert [r.alpha for r in store.rows()] == pytest.approx([0.1, 0.2])


def test_store_rejects_wrong_format_request(tmp_path):
    store = ResultStore(tmp_path / 'store', format='csv')
    store.append(row_from_compute_all(make_config(alpha=0.1), {'mmd': 0.5}))
    with pytest.raises(ResultStoreError):
        ResultStore(tmp_path / 'store', format='parquet')
    assert len(ResultStore(tmp_path / 'store')) == 1


def test_row_validation_requires_identity_fields():
    with pytest.raises(ValueError):
        ResultRow.from_dict({'experiment_family': 'f', 'dataset': 'd', 'n_P': 1, 'n_Q': 1, 'status': 'ok'})
    with pytest.raises(ValueError):
        ResultRow.from_dict({
            'experiment_id': 'e', 'experiment_family': 'f', 'dataset': 'd',
            'n_P': None, 'n_Q': 1, 'status': 'ok',
        })
    row = ResultRow.from_dict({
        'experiment_id': 'e', 'experiment_family': 'f', 'dataset': 'd',
        'n_P': 900, 'n_Q': 900, 'status': 'ok', 'layer': 6.0, 'seed': 42,
        'model': None, 'representation': None,
    })
    assert row.layer == 6 and row.seed == 42
    assert row.model == '' and row.representation == ''


def test_external_store_change_detected(tmp_path):
    store_first = ResultStore(tmp_path / 'store')
    store_second = ResultStore(tmp_path / 'store')
    config_a = make_config(alpha=0.1)
    config_b = make_config(alpha=0.2)
    store_second.append(row_from_compute_all(config_b, {'mmd': 0.2}))
    with pytest.raises(ResultStoreError, match='извне'):
        store_first.append(row_from_compute_all(config_a, {'mmd': 0.1}))
    reopened = ResultStore(tmp_path / 'store')
    assert len(reopened) == 1
    assert reopened.rows()[0].alpha == pytest.approx(0.2)
    store_second.append(row_from_compute_all(config_a, {'mmd': 0.1}))
    assert len(ResultStore(tmp_path / 'store')) == 2


def test_metric_params_in_key_and_round_trip(tmp_path):
    params = '{"js_k":5,"rtd_trials":5}'
    plain = make_config(alpha=0.1)
    tuned = make_config(alpha=0.1, metric_params=params)
    assert tuned.key() != plain.key()
    assert tuned.key() == make_config(alpha=0.1, metric_params=params).key()

    store = ResultStore(tmp_path / 'store')
    key = store.append(row_from_compute_all(tuned, {'mmd': 0.3}))
    reopened = ResultStore(tmp_path / 'store')
    row = reopened.rows()[0]
    assert row.metric_params == params
    assert row.config_key() == key == tuned.key()
    assert reopened.contains(tuned) and not reopened.contains(plain)


def test_run_grid_metric_params_passthrough(tmp_path):
    store = ResultStore(tmp_path / 'store')
    configs = [make_config(alpha=0.1), make_config(alpha=0.2, metric_params='{"js_k":3}')]
    seen = []

    def runner(config):
        seen.append(config.metric_params)
        return {'mmd': 1.0}

    run_grid(store, configs[:1], runner, metric_params='{"js_k":5}')
    assert seen == ['{"js_k":5}']
    assert store.rows()[0].metric_params == '{"js_k":5}'
    assert store.contains(make_config(alpha=0.1, metric_params='{"js_k":5}'))

    with pytest.raises(ValueError, match='metric_params'):
        run_grid(store, configs[1:], runner, metric_params='{"js_k":5}')
    assert len(store) == 1


def test_on_error_failure_does_not_cancel_failed_row(tmp_path):
    store = ResultStore(tmp_path / 'store')
    config = make_config(alpha=0.7)

    def runner(config):
        raise RuntimeError('сбой')

    def broken_on_error(config, exc):
        raise ValueError('наблюдатель сломан')

    with pytest.warns(UserWarning, match='on_error'):
        written = run_grid(store, [config], runner, on_error=broken_on_error)
    assert len(written) == 1 and written[0].status == 'failed'
    reopened = ResultStore(tmp_path / 'store')
    assert len(reopened) == 1
    assert reopened.failed_keys() == [config.key()]


def test_corrupt_manifest_raises_result_store_error(tmp_path):
    directory = tmp_path / 'store'
    store = ResultStore(directory)
    store.append(row_from_compute_all(make_config(alpha=0.1), {'mmd': 0.5}))
    (directory / 'manifest.json').write_text('{"rows": ', encoding='utf-8')
    with pytest.raises(ResultStoreError):
        ResultStore(directory)


def test_config_key_rejects_nan():
    config = make_config(alpha=float('nan'))
    with pytest.raises(ValueError):
        config.key()
