"""Единая строка результата эксперимента и persistency-слой (мастер-план §6.2, §8 Фаза 2).

Schema общая для всех семейств экспериментов: поля, необязательные для
конкретного семейства, остаются None, но колонки присутствуют всегда.
Ключ конфигурации — стабильный хэш канонического JSON config-полей; он же —
основа запрета дубликатов и resume. Числовое ядро (metrics.py) не используется
и не изменяется: модуль принимает только плоские словари формата compute_all.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass, fields as dataclass_fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import pandas as pd

SCHEMA_VERSION = 1
MANIFEST_NAME = 'manifest.json'
DATA_PARQUET = 'results.parquet'
DATA_CSV = 'results.csv'
KEY_LENGTH = 16
STORE_FORMATS = ('parquet', 'csv')


class ResultStoreError(Exception):
    """Базовая ошибка хранилища результатов."""


class DuplicateKeyError(ResultStoreError):
    """Конфигурация с таким ключом уже записана."""


class ManifestMismatchError(ResultStoreError):
    """Manifest не соответствует фактическому содержимому хранилища."""


@dataclass
class ExperimentConfig:
    """Экспериментальная конфигурация; key() — уникальный ключ строки результата.

    Все поля, идентифицирующие конфигурацию (включая seed и experiment_id),
    входят в ключ: повторный прогон той же конфигурации — дубликат,
    намеренный пере-ран нужно оформлять новым experiment_id.
    metric_params — параметры метрик прогона (rtd_trials, js_k, ...)
    канонической JSON-строкой; входят в ключ (мастер-план §9.6 «все параметры
    метрик записаны»).
    """

    experiment_id: str
    experiment_family: str
    dataset: str
    model: str = ''
    model_revision: Optional[str] = None
    representation: str = ''
    layer: Optional[int] = None
    pooling: Optional[str] = None
    pca_dim: Optional[int] = None
    normalization: Optional[str] = None
    n_P: int = 0
    n_Q: int = 0
    alpha: Optional[float] = None
    drop_fraction: Optional[float] = None
    invent_fraction: Optional[float] = None
    corruption: Optional[str] = None
    severity: Optional[float] = None
    seed: Optional[int] = None
    split_id: Optional[str] = None
    prompt_ids_hash: Optional[str] = None
    metric_params: Optional[str] = None

    def key(self) -> str:
        """Стабильный ключ конфигурации (sha256 канонического JSON, 16 hex-символов)."""
        return stable_hash({name: getattr(self, name) for name in CONFIG_COLUMNS})


_FIELD_TO_COLUMN = {
    'precision_1': 'precision@1',
    'recall_1': 'recall@1',
    'precision_3': 'precision@3',
    'recall_3': 'recall@3',
    'precision_10': 'precision@10',
    'recall_10': 'recall@10',
}

CONFIG_COLUMNS = tuple(f.name for f in dataclass_fields(ExperimentConfig))
METRIC_COLUMNS = (
    'mtd_PQ', 'mtd_QP', 'ntd_PQ', 'ntd_QP', 'rtd',
    'precision@1', 'recall@1', 'precision@3', 'recall@3', 'precision@10', 'recall@10',
    'mmd', 'frechet', 'js',
)
ROW_COLUMNS = CONFIG_COLUMNS + METRIC_COLUMNS + ('runtime_seconds', 'status', 'error_message')
FRAME_COLUMNS = ROW_COLUMNS + ('config_key',)

_INT_ROW_FIELDS = ('n_P', 'n_Q', 'layer', 'pca_dim', 'seed')
_METRIC_ROW_FIELDS = (
    'mtd_PQ', 'mtd_QP', 'ntd_PQ', 'ntd_QP', 'rtd',
    'precision_1', 'recall_1', 'precision_3', 'recall_3', 'precision_10', 'recall_10',
    'mmd', 'frechet', 'js',
)
_FLOAT_ROW_FIELDS = ('alpha', 'drop_fraction', 'invent_fraction', 'severity') + _METRIC_ROW_FIELDS + ('runtime_seconds',)
NUMERIC_COLUMNS = tuple(
    _FIELD_TO_COLUMN.get(name, name) for name in _INT_ROW_FIELDS + _FLOAT_ROW_FIELDS
)
_REQUIRED_STR_ROW_FIELDS = ('experiment_id', 'experiment_family', 'dataset', 'status')
_STR_DEFAULT_ROW_FIELDS = {'model': '', 'representation': ''}

_METRIC_FROM_COMPUTE_ALL = {
    'mtd_PQ': 'mtd_PQ',
    'mtd_QP': 'mtd_QP',
    'ntd_PQ': 'ntd_PQ',
    'ntd_QP': 'ntd_QP',
    'rtd': 'rtd',
    'mmd': 'mmd',
    'frechet': 'frechet',
    'js': 'js',
    'precision@1': 'precision_1',
    'recall@1': 'recall_1',
    'precision@3': 'precision_3',
    'recall@3': 'recall_3',
    'precision@10': 'precision_10',
    'recall@10': 'recall_10',
}


def stable_hash(payload: dict) -> str:
    """Стабильный короткий хэш словаря: канонический JSON (sort_keys) -> sha256.

    allow_nan=False: NaN/Infinity — мусорная конфигурация — дают громкий
    ValueError, а не «валидный» ключ с нестандартными токенами.
    """
    canonical = json.dumps(
        payload, sort_keys=True, ensure_ascii=True, separators=(',', ':'),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:KEY_LENGTH]


@dataclass
class ResultRow:
    """Единая строка результата (мастер-план §6.2); имена полей precision_k
    соответствуют колонкам precision@k."""

    experiment_id: str
    experiment_family: str
    dataset: str
    model: str = ''
    model_revision: Optional[str] = None
    representation: str = ''
    layer: Optional[int] = None
    pooling: Optional[str] = None
    pca_dim: Optional[int] = None
    normalization: Optional[str] = None
    n_P: int = 0
    n_Q: int = 0
    alpha: Optional[float] = None
    drop_fraction: Optional[float] = None
    invent_fraction: Optional[float] = None
    corruption: Optional[str] = None
    severity: Optional[float] = None
    seed: Optional[int] = None
    split_id: Optional[str] = None
    prompt_ids_hash: Optional[str] = None
    metric_params: Optional[str] = None
    mtd_PQ: Optional[float] = None
    mtd_QP: Optional[float] = None
    ntd_PQ: Optional[float] = None
    ntd_QP: Optional[float] = None
    rtd: Optional[float] = None
    precision_1: Optional[float] = None
    recall_1: Optional[float] = None
    precision_3: Optional[float] = None
    recall_3: Optional[float] = None
    precision_10: Optional[float] = None
    recall_10: Optional[float] = None
    mmd: Optional[float] = None
    frechet: Optional[float] = None
    js: Optional[float] = None
    runtime_seconds: Optional[float] = None
    status: str = 'ok'
    error_message: Optional[str] = None

    def config(self) -> ExperimentConfig:
        """Конфигурационная часть строки (поля §6.2 до метрик)."""
        return ExperimentConfig(**{name: getattr(self, name) for name in CONFIG_COLUMNS})

    def config_key(self) -> str:
        """Ключ конфигурации строки; совпадает с key() её ExperimentConfig."""
        return self.config().key()

    def to_dict(self) -> dict:
        """Словарь с колонками §6.2 (precision@k и т.д.) в порядке ROW_COLUMNS."""
        return {
            _FIELD_TO_COLUMN.get(name, name): value
            for name, value in asdict(self).items()
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ResultRow':
        """Строка из словаря колонок §6.2; NaN/None -> None, целочисленные поля
        приводятся к int, обязательные поля (experiment_id, experiment_family,
        dataset, n_P, n_Q, status) при отсутствии дают ValueError."""
        kwargs = {}
        for f in dataclass_fields(cls):
            column = _FIELD_TO_COLUMN.get(f.name, f.name)
            value = data.get(column)
            missing = value is None or (isinstance(value, float) and value != value)
            if f.name in _REQUIRED_STR_ROW_FIELDS:
                if missing:
                    raise ValueError(f'обязательное поле {column} отсутствует или NaN')
                kwargs[f.name] = str(value)
            elif f.name in ('n_P', 'n_Q'):
                if missing:
                    raise ValueError(f'обязательное поле {column} отсутствует или NaN')
                kwargs[f.name] = int(value)
            elif f.name in _STR_DEFAULT_ROW_FIELDS:
                kwargs[f.name] = '' if missing else str(value)
            elif f.name in _INT_ROW_FIELDS:
                kwargs[f.name] = None if missing else int(value)
            elif f.name in _FLOAT_ROW_FIELDS:
                kwargs[f.name] = None if missing else float(value)
            else:
                kwargs[f.name] = None if missing else value
        return cls(**kwargs)


def row_from_compute_all(
    config: ExperimentConfig,
    metrics: dict,
    runtime_seconds: Optional[float] = None,
) -> ResultRow:
    """Строка schema из плоского словаря compute_all (формат metrics.py:248-260).

    Отсутствующие необязательные метрики (rtd/ntd выключены через skip,
    не все nhood_sizes) -> None. Неизвестные ключи -> ValueError: молча
    терять посчитанные метрики нельзя.
    """
    unknown = sorted(set(metrics) - set(_METRIC_FROM_COMPUTE_ALL))
    if unknown:
        raise ValueError(f'неизвестные ключи метрик: {unknown}')
    metric_kwargs = {
        field: float(metrics[key])
        for key, field in _METRIC_FROM_COMPUTE_ALL.items()
        if key in metrics
    }
    return ResultRow(
        **asdict(config),
        **metric_kwargs,
        runtime_seconds=runtime_seconds,
        status='ok',
    )


def failed_row(
    config: ExperimentConfig,
    error_message: Any,
    runtime_seconds: Optional[float] = None,
) -> ResultRow:
    """Строка со status='failed' для упавшей конфигурации (изоляция ошибки)."""
    return ResultRow(
        **asdict(config),
        status='failed',
        error_message=str(error_message),
        runtime_seconds=runtime_seconds,
    )


def _pyarrow_available() -> bool:
    try:
        import pyarrow
    except ImportError:
        return False
    return True


def _serialize_frame(fmt: str, frame: pd.DataFrame) -> bytes:
    if fmt == 'parquet':
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.Table.from_pandas(frame, preserve_index=False)
        sink = pa.BufferOutputStream()
        pq.write_table(table, sink)
        return sink.getvalue().to_pybytes()
    return frame.to_csv(index=False).encode('utf-8')


def _deserialize_frame(fmt: str, data: bytes) -> pd.DataFrame:
    if fmt == 'parquet':
        return pd.read_parquet(io.BytesIO(data), engine='pyarrow')
    return pd.read_csv(io.BytesIO(data))


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.reindex(columns=list(FRAME_COLUMNS))
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column])
    return frame


def _atomic_write(path: Path, data: bytes) -> None:
    """Атомарная запись: уникальный tmp (безопасно при параллельных
    экземплярах), fsync файла, os.replace, fsync каталога (durability имён)."""
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + '.', suffix='.tmp'
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    dir_flags = getattr(os, 'O_DIRECTORY', os.O_RDONLY)
    dir_fd = os.open(path.parent, dir_flags)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


class ResultStore:
    """Хранилище строк результата: файл данных (Parquet, fallback CSV) + JSON manifest.

    Ключ конфигурации — основа всех операций: manifest хранит key ->
    {experiment_id, status}; уже записанные ключи не принимаются повторно
    (DuplicateKeyError). Каждая запись атомарна (tmp + os.replace, сперва файл
    данных, затем manifest): окно сбоя между двумя replace детектируется при
    следующем открытии как ManifestMismatchError и лечится только явным
    rebuild_manifest. Коммит сверяет manifest на диске с собственным
    in-memory состоянием: изменение каталога другим процессом/экземпляром
    между открытием и записью — ResultStoreError (контракт: один писатель
    на каталог), тихая потеря строк исключена. Открытие существующего
    хранилища всегда валидирует manifest против фактического содержимого —
    расхождение это исключение, не молчание.
    """

    def __init__(self, directory, *, format: Optional[str] = None, csv_fallback: bool = True):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.directory / MANIFEST_NAME
        if self._manifest_path.exists():
            self._open_existing(format)
        else:
            self._open_fresh(format, csv_fallback)

    @property
    def _data_path(self) -> Path:
        return self.directory / (DATA_PARQUET if self.format == 'parquet' else DATA_CSV)

    def _open_existing(self, requested_format: Optional[str]) -> None:
        try:
            manifest = json.loads(self._manifest_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError as exc:
            raise ResultStoreError(
                f'manifest повреждён (не читается как JSON): {self._manifest_path}: {exc}'
            ) from exc
        if not isinstance(manifest, dict):
            raise ResultStoreError(
                f'manifest повреждён (JSON-значение не объект): {self._manifest_path}'
            )
        if manifest.get('schema_version') != SCHEMA_VERSION:
            raise ResultStoreError(
                f'версия schema manifest {manifest.get("schema_version")!r} != {SCHEMA_VERSION}'
            )
        on_disk = manifest.get('format')
        if on_disk not in STORE_FORMATS:
            raise ResultStoreError(f'неизвестный формат в manifest: {on_disk!r}')
        if requested_format is not None and requested_format != on_disk:
            raise ResultStoreError(f'на диске формат {on_disk!r}, запрошен {requested_format!r}')
        self.format = on_disk
        self._rows = {key: dict(entry) for key, entry in manifest.get('rows', {}).items()}
        self._manifest_n_rows = int(manifest.get('n_rows', -1))
        data_path = self._data_path
        if not data_path.exists():
            raise ManifestMismatchError(f'manifest есть, файла данных нет: {data_path}')
        frame = _deserialize_frame(self.format, data_path.read_bytes())
        if 'config_key' not in frame.columns:
            raise ManifestMismatchError('файл данных повреждён: нет колонки config_key')
        self._frame = _normalize_frame(frame)
        self._validate_against_manifest()

    def _open_fresh(self, requested_format: Optional[str], csv_fallback: bool) -> None:
        if (self.directory / DATA_PARQUET).exists() or (self.directory / DATA_CSV).exists():
            raise ManifestMismatchError('файл данных есть, manifest отсутствует')
        fmt = requested_format or 'parquet'
        if fmt not in STORE_FORMATS:
            raise ValueError(f'формат должен быть одним из {STORE_FORMATS}: {fmt!r}')
        if fmt == 'parquet' and not _pyarrow_available():
            if not csv_fallback:
                raise RuntimeError('pyarrow недоступен: Parquet требует pyarrow')
            warnings.warn('pyarrow недоступен: хранилище переключено на CSV '
                          '(Parquet — требование мастер-плана, ставьте pyarrow)')
            fmt = 'csv'
        self.format = fmt
        self._rows = {}
        self._manifest_n_rows = 0
        self._frame = _normalize_frame(pd.DataFrame(columns=list(FRAME_COLUMNS)))

    def _manifest_dict(self) -> dict:
        return {
            'schema_version': SCHEMA_VERSION,
            'format': self.format,
            'n_rows': len(self._frame),
            'rows': self._rows,
            'updated_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        }

    def _commit(self, frame: pd.DataFrame, key: str, row: ResultRow) -> None:
        if self._manifest_path.exists():
            try:
                disk = json.loads(self._manifest_path.read_text(encoding='utf-8'))
            except json.JSONDecodeError as exc:
                raise ResultStoreError(
                    f'manifest на диске повреждён (не JSON): {self._manifest_path}; '
                    'восстановите через ResultStore.rebuild_manifest'
                ) from exc
            if not isinstance(disk, dict) or set(disk.get('rows', {})) != set(self._rows):
                raise ResultStoreError(
                    'хранилище изменено извне между открытием и записью '
                    '(другой процесс/экземпляр): перезакройте ResultStore'
                )
        frame = _normalize_frame(frame)
        _atomic_write(self._data_path, _serialize_frame(self.format, frame))
        self._frame = frame
        self._rows[key] = {'experiment_id': row.experiment_id, 'status': row.status}
        manifest_bytes = json.dumps(self._manifest_dict(), ensure_ascii=False, indent=2).encode('utf-8')
        _atomic_write(self._manifest_path, manifest_bytes)

    def _validate_against_manifest(self) -> None:
        frame_keys = self._frame['config_key'].astype(str).tolist()
        if len(set(frame_keys)) != len(frame_keys):
            raise ManifestMismatchError('в файле данных дубликаты config_key')
        manifest_keys = set(self._rows)
        frame_key_set = set(frame_keys)
        if frame_key_set != manifest_keys:
            raise ManifestMismatchError(
                f'manifest ({len(manifest_keys)} ключей) не совпадает с данными '
                f'({len(frame_key_set)}): только в manifest='
                f'{sorted(manifest_keys - frame_key_set)}, только в данных='
                f'{sorted(frame_key_set - manifest_keys)}'
            )
        if self._manifest_n_rows != len(self._frame):
            raise ManifestMismatchError(
                f'n_rows в manifest ({self._manifest_n_rows}) != строк в данных ({len(self._frame)})'
            )

    def validate(self) -> None:
        """Явная проверка manifest против содержимого; расхождение — исключение."""
        self._validate_against_manifest()

    def append(self, row: ResultRow) -> str:
        """Добавить строку; повторный ключ — DuplicateKeyError. Возвращает ключ."""
        key = row.config_key()
        if key in self._rows:
            raise DuplicateKeyError(
                f'конфигурация experiment_id={row.experiment_id!r} (key={key}) уже записана'
            )
        record = pd.DataFrame([row.to_dict()])
        record['config_key'] = key
        self._commit(pd.concat([self._frame, record], ignore_index=True), key, row)
        return key

    def replace(self, row: ResultRow) -> str:
        """Заменить уже записанную строку с тем же ключом (например, при
        повторе failed-конфигурации); отсутствующий ключ — ResultStoreError."""
        key = row.config_key()
        if key not in self._rows:
            raise ResultStoreError(f'ключ {key} не записан: replace только для существующих')
        kept = self._frame[self._frame['config_key'] != key]
        record = pd.DataFrame([row.to_dict()])
        record['config_key'] = key
        self._commit(pd.concat([kept, record], ignore_index=True), key, row)
        return key

    @classmethod
    def rebuild_manifest(cls, directory) -> 'ResultStore':
        """Явное восстановление manifest из файла данных после рассинхрона
        (сбой в окне между записью данных и manifest). Файл данных — источник
        истины; вызов перезаписывает manifest и возвращает открытое хранилище."""
        directory = Path(directory)
        matches = [
            (directory / DATA_PARQUET, 'parquet'),
            (directory / DATA_CSV, 'csv'),
        ]
        matches = [(path, fmt) for path, fmt in matches if path.exists()]
        if not matches:
            raise ResultStoreError(f'файл данных не найден в {directory}')
        if len(matches) > 1:
            raise ResultStoreError(f'в {directory} оба файла данных: неоднозначно')
        data_path, fmt = matches[0]
        frame = _deserialize_frame(fmt, data_path.read_bytes())
        if 'config_key' not in frame.columns:
            raise ResultStoreError('файл данных повреждён: нет колонки config_key')
        frame = _normalize_frame(frame)
        rows_map = {}
        for record in frame.to_dict('records'):
            row = ResultRow.from_dict(record)
            rows_map[row.config_key()] = {'experiment_id': row.experiment_id, 'status': row.status}
        manifest = {
            'schema_version': SCHEMA_VERSION,
            'format': fmt,
            'n_rows': len(frame),
            'rows': rows_map,
            'updated_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        }
        manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8')
        _atomic_write(directory / MANIFEST_NAME, manifest_bytes)
        return cls(directory)

    def keys(self) -> set:
        """Множество записанных ключей конфигураций."""
        return set(self._rows)

    def contains(self, config: ExperimentConfig) -> bool:
        """Записана ли уже эта конфигурация (основа resume)."""
        return config.key() in self._rows

    def row_status(self, key: str) -> Optional[str]:
        """status записанной строки по ключу; None, если ключа нет."""
        entry = self._rows.get(key)
        return None if entry is None else entry.get('status')

    def failed_keys(self) -> list:
        """Отсортированный список ключей со status='failed'."""
        return sorted(
            key for key, entry in self._rows.items() if entry.get('status') == 'failed'
        )

    def load_frame(self) -> pd.DataFrame:
        """Копия таблицы строк (колонки §6.2 + config_key)."""
        return self._frame.copy()

    def rows(self) -> list:
        """Все строки как список ResultRow (в порядке записи)."""
        return [ResultRow.from_dict(record) for record in self._frame.to_dict('records')]

    def __len__(self) -> int:
        return len(self._frame)


def run_grid(
    store: ResultStore,
    configs: Iterable[ExperimentConfig],
    runner: Callable[[ExperimentConfig], dict],
    *,
    retry_failed: bool = False,
    on_error: Optional[Callable[[ExperimentConfig, Exception], Any]] = None,
    metric_params: Optional[str] = None,
) -> list:
    """Прогон grid-конфигураций с resume, изоляцией ошибок и таймингом.

    Уже записанные ключи пропускаются (resume по manifest); при
    retry_failed=True перезапускаются только failed-ключи (строка заменяется
    через replace). Ошибка одной конфигурации -> status='failed' +
    error_message, grid продолжается; failed-строка персистится ДО вызова
    on_error, исключение колбека — предупреждение, а не отмена записи.
    KeyboardInterrupt/SystemExit пробрасываются (частичный store валиден
    и продолжает жить resume'ом). metric_params (canonical JSON) подставляется
    конфигурациям без собственного значения и участвует в ключе; конфликт
    с собственным значением конфигурации -> ValueError. Возвращает строки,
    записанные в этом прогоне.
    """
    written = []
    for config in configs:
        if metric_params is not None:
            if config.metric_params is None:
                config = replace(config, metric_params=metric_params)
            elif config.metric_params != metric_params:
                raise ValueError(
                    f'конфликт metric_params: у конфигурации '
                    f'{config.experiment_id!r} задано {config.metric_params!r}, '
                    f'grid передан {metric_params!r}'
                )
        key = config.key()
        stored_status = store.row_status(key)
        if stored_status is not None:
            if not (retry_failed and stored_status == 'failed'):
                continue
        started = time.perf_counter()
        error = None
        try:
            metrics = runner(config)
            row = row_from_compute_all(
                config, metrics, runtime_seconds=time.perf_counter() - started
            )
        except Exception as exc:
            error = exc
            row = failed_row(
                config,
                f'{type(exc).__name__}: {exc}',
                runtime_seconds=time.perf_counter() - started,
            )
        if stored_status is None:
            store.append(row)
        else:
            store.replace(row)
        if error is not None and on_error is not None:
            try:
                on_error(config, error)
            except Exception as callback_error:
                warnings.warn(
                    f'on_error поднял исключение для {config.experiment_id!r} '
                    f'({callback_error!r}); failed-строка уже записана'
                )
        written.append(row)
    return written
