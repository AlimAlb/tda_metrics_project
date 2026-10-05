"""Детерминированный дисковый кэш эмбеддингов LLM (мастер-план §L1, D-006-усл.5).

Каталог кэша — три файла: ``meta.parquet`` (построчные метаданные: строка
на пару (пункт, слой)), ``vectors.npy`` (тензор эмбеддингов) и
``manifest.json`` (конфигурация прогона и сводная статистика). Запись
атомарна (tmp + fsync файла + ``os.replace`` + fsync каталога на каждый
файл): частичный каталог (из-за сбоя посередине записи) не может сойти за
валидный кэш, имена файлов переживают сбой ОС. ``load_cache`` опционально
сверяет фактический manifest с ожидаемым подмножеством полей (``expect``)
и всегда — согласованность manifest с фактом (``n_items`` vs первая ось
``vectors``, ``layer_indices`` vs вторая ось, строки ``meta`` vs
``n_items`` x слоёв): несоответствие даёт ``CacheMismatchError`` с
деталями, а не молчаливую подмену данных.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    'CacheMismatchError',
    'save_cache',
    'load_cache',
    'prompt_ids_hash',
    'META_NAME',
    'VECTORS_NAME',
    'MANIFEST_NAME',
]

META_NAME = 'meta.parquet'
VECTORS_NAME = 'vectors.npy'
MANIFEST_NAME = 'manifest.json'


class CacheMismatchError(Exception):
    """Несоответствие кэша ожиданиям: manifest не сходится с ``expect`` или
    с фактическим содержимым (``vectors``/``meta``)."""


def prompt_ids_hash(frame):
    """sha256 отсортированных уникальных ``prompt_id`` (D-006-усл.5).

    ``frame`` — DataFrame с колонкой ``prompt_id`` либо произвольный
    итерируемый набор идентификаторов. Стабилен при перестановке строк
    и дублях, чувствителен к изменению самого набора уникальных значений.
    """
    if isinstance(frame, pd.DataFrame):
        values = frame['prompt_id'].tolist()
    else:
        values = list(frame)
    unique_sorted = sorted({str(v) for v in values})
    payload = '\n'.join(unique_sorted).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """tmp в той же директории + fsync файла + os.replace + fsync каталога:
    запись видна только целиком, имя файла переживает сбой ОС."""
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


def _serialize_meta(meta: pd.DataFrame) -> bytes:
    import io
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_pandas(meta, preserve_index=False)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return sink.getvalue().to_pybytes()


def _serialize_vectors(vectors: np.ndarray) -> bytes:
    import io
    buffer = io.BytesIO()
    np.save(buffer, vectors)
    return buffer.getvalue()


def save_cache(directory, meta, vectors, manifest):
    """Атомарно сохранить кэш эмбеддингов в ``directory``.

    Порядок записи: сначала ``vectors.npy`` и ``meta.parquet`` (данные),
    затем ``manifest.json`` (последним) — при сбое посередине каталог либо
    пуст, либо содержит данные без manifest (детектируется явно при
    ``load_cache``, не выдаётся за валидный кэш). Каждый файл пишется через
    собственный tmp + fsync + ``os.replace`` + fsync каталога (один писатель
    на каталог).
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    vectors = np.asarray(vectors)
    _atomic_write_bytes(directory / VECTORS_NAME, _serialize_vectors(vectors))
    meta_frame = meta if isinstance(meta, pd.DataFrame) else pd.DataFrame(meta)
    _atomic_write_bytes(directory / META_NAME, _serialize_meta(meta_frame))
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode('utf-8')
    _atomic_write_bytes(directory / MANIFEST_NAME, manifest_bytes)


def _check_expect(manifest: dict, expect: dict) -> None:
    for key, expected_value in expect.items():
        if key not in manifest:
            raise CacheMismatchError(
                f'ключ {key!r} отсутствует в manifest кэша; ожидалось {expected_value!r}'
            )
        actual_value = manifest[key]
        if actual_value != expected_value:
            raise CacheMismatchError(
                f'несовпадение поля {key!r}: ожидалось {expected_value!r}, в кэше {actual_value!r}'
            )


def _check_consistency(manifest: dict, meta: pd.DataFrame, vectors: np.ndarray) -> None:
    """Сверить manifest с фактическим содержимым кэша.

    ``n_items`` — обязательное поле, обязано совпадать с первой осью
    ``vectors``; ``layer_indices`` (если заданы) — со второй осью; число
    строк ``meta`` — с ``n_items`` x число слоёв (строка meta на пару
    (пункт, слой), как пишет ``extract_embeddings``). Любое расхождение —
    ``CacheMismatchError`` с деталями (поле, ожидалось/факт): кэш с
    рассинхронизированными файлами (собранный руками) не должен грузиться
    молча.
    """
    if 'n_items' not in manifest:
        raise CacheMismatchError(
            'в manifest кэша отсутствует ключ n_items — согласованность кэша проверить нельзя'
        )
    n_items = manifest['n_items']
    actual_items = vectors.shape[0] if vectors.ndim else None
    if actual_items != n_items:
        raise CacheMismatchError(
            f'несогласованность кэша: manifest n_items={n_items!r}, '
            f'фактически vectors.shape[0]={actual_items!r}'
        )
    n_layers = len(manifest['layer_indices']) if 'layer_indices' in manifest else None
    if n_layers is not None:
        actual_layers = vectors.shape[1] if vectors.ndim > 1 else None
        if actual_layers != n_layers:
            raise CacheMismatchError(
                f'несогласованность кэша: manifest layer_indices задаёт {n_layers} слоёв, '
                f'фактически vectors.shape[1]={actual_layers!r}'
            )
    expected_meta_rows = n_items * (n_layers if n_layers is not None else 1)
    if len(meta) != expected_meta_rows:
        raise CacheMismatchError(
            f'несогласованность кэша: строк meta={len(meta)!r}, '
            f'ожидалось {expected_meta_rows!r} (n_items={n_items!r}, слоёв={n_layers!r})'
        )


def load_cache(directory, expect=None):
    """Загрузить кэш из ``directory``; вернуть ``(meta, vectors, manifest)``.

    ``expect`` — словарь-подмножество ожидаемых полей manifest (например,
    ``{'model': ..., 'input_format': ..., 'pooling': ...}``); несовпадение
    любого ключа (включая отсутствие) -> ``CacheMismatchError`` с деталями
    (какой ключ, ожидалось/факт). Всегда сверяется согласованность manifest
    с фактом: ``n_items`` с первой осью ``vectors``, ``layer_indices`` (если
    заданы) со второй осью, число строк ``meta`` с ``n_items`` x слоёв —
    рассинхронизированный кэш не грузится молча. Отсутствие любого из трёх
    файлов кэша — ``FileNotFoundError``.
    """
    directory = Path(directory)
    manifest_path = directory / MANIFEST_NAME
    meta_path = directory / META_NAME
    vectors_path = directory / VECTORS_NAME
    for path in (manifest_path, meta_path, vectors_path):
        if not path.exists():
            raise FileNotFoundError(f'файл кэша не найден: {path}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if expect:
        _check_expect(manifest, expect)
    meta = pd.read_parquet(meta_path, engine='pyarrow')
    vectors = np.load(vectors_path)
    _check_consistency(manifest, meta, vectors)
    return meta, vectors, manifest
