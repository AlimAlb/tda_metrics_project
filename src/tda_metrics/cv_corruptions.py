"""Matched-MMD порчи CIFAR.

Проверяет, различают ли топологические метрики ХАРАКТЕР порчи изображений
при сопоставимой общей величине сдвига распределения (matched MMD). Две
стадии поверх единой schema `experiment_schema.ExperimentConfig`:

1. Этап-1 (scan): дешёвый MMD-скан по сетке severity для каждой из четырёх
   порч — `build_c3_scan_configs`.
2. Этап-2 (final): по фактическим MMD из Этапа-1 для каждой порчи выбирается
   severity с ближайшим MMD к целевому значению (`select_matched_severities`),
   строятся финальные конфиги для сравнения всех метрик на matched-MMD порчах
   плюс clean-контроль — `build_c3_matched_configs`.

Единственное пространство — `cifar_clip_pca16`: CLIP ViT-B/32 на CIFAR-10
train, PCA-16, фит на ПОЛНОМ clean-пуле (reference; правило RISKS №3, как в
`cv_grid.prepare_pools`). Модуль чистый: не грузит ни датасеты, ни модели —
`build_pair` принимает уже готовые clean-эмбеддинги/изображения/labels, а
кодирование порчённых изображений через CLIP+PCA — обязанность ран-скрипта
оркестратора (инъекция encode_fn, как metrics_fn в `cv_grid.make_runner`).

Дизайн CRN (common random numbers), аналогично синтетике (S1): P и Q внутри
одного повтора строятся из ОДНОГО сбалансированного по классам набора
индексов (seed 42 + 1000·k) — различие между P и Q только в применённой
порче, флуктуация состава выборки исключена. Разные повторы k — разные
индексы (та же схема сидов, что и в `cv_grid`, но независимо реализованная:
модуль не импортирует `cv_grid`, чтобы ветка оставалась самодостаточной).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from tda_metrics.experiment_schema import ExperimentConfig

__all__ = [
    'CORRUPTIONS',
    'SEVERITY_GRID',
    'gaussian_noise',
    'gaussian_blur',
    'brightness',
    'contrast',
    'corrupt_images',
    'class_balanced_indices',
    'build_c3_scan_configs',
    'build_c3_matched_configs',
    'select_matched_severities',
    'build_pair',
    'parse_experiment_id',
]

DATASET_NAME = 'cifar10'
MODEL_NAME = 'clip-vit-b32'
REPRESENTATION_NAME = 'clip'
PCA_DIM = 16

SEED_P = 42
SEED_STEP = 1000
DEFAULT_REPEATS = 3

SEVERITY_GRID = (0.05, 0.1, 0.2, 0.3, 0.5, 0.75)


def _to_float_image(images):
    images = np.asarray(images)
    return images.astype(np.float64)


def _clip_to_uint8(images):
    return np.clip(images, 0.0, 255.0).astype(np.uint8)


def gaussian_noise(images, severity):
    """Аддитивный гауссов шум: sigma = severity * 255/4 (до ~64 при severity=1).

    severity=0 -> sigma=0 -> результат совпадает со входом (после округления
    до uint8). Шум детерминирован (RNG с фиксированным seed=0) — функция
    остаётся чистой: один и тот же вызов всегда даёт один и тот же результат.
    """
    out = _to_float_image(images)
    sigma = severity * 255.0 / 4.0
    if sigma > 0:
        rng = np.random.default_rng(0)
        out = out + rng.normal(0.0, sigma, size=out.shape)
    return _clip_to_uint8(out)


def gaussian_blur(images, severity):
    """Гауссово размытие по пространственным осям (отдельно по каждому каналу).

    sigma = 0.5 + severity * 3.5 по формуле спецификации; однако при
    severity=0 размытие явно отключается (bypass), а не применяется с
    sigma=0.5 — иначе нарушался бы общий инвариант «severity=0 -> идентичная
    копия», заявленный для всех четырёх порч. Это единственное отклонение от
    буквальной формулы, продиктованное этим инвариантом.
    """
    out = _to_float_image(images)
    if severity == 0:
        return _clip_to_uint8(out)
    sigma = 0.5 + severity * 3.5
    blurred = gaussian_filter(out, sigma=(0, sigma, sigma, 0))
    return _clip_to_uint8(blurred)


def brightness(images, severity):
    """Яркость: умножение на (1 + severity * 0.8), затем клип в [0, 255]."""
    out = _to_float_image(images) * (1.0 + severity * 0.8)
    return _clip_to_uint8(out)


def contrast(images, severity):
    """Контраст: (x - mean) * (1 + severity) + mean, mean — per-image скаляр
    (среднее по высоте/ширине/каналам одного изображения).

    При severity == 0 — идентичная копия (bypass, аналогично gaussian_blur):
    литеральное (x - mean) + mean в float64 допускает ошибку ~1 ulp, которая
    после усечения до uint8 меняла бы отдельные пиксели и нарушала общий
    инвариант «severity=0 -> идентичная копия».
    """
    out = _to_float_image(images)
    if severity == 0:
        return _clip_to_uint8(out)
    mean = out.mean(axis=(1, 2, 3), keepdims=True)
    out = (out - mean) * (1.0 + severity) + mean
    return _clip_to_uint8(out)


CORRUPTIONS = {
    'gaussian_noise': gaussian_noise,
    'gaussian_blur': gaussian_blur,
    'brightness': brightness,
    'contrast': contrast,
}


def corrupt_images(images, corruption, severity):
    """Диспетчер порчи по имени из реестра CORRUPTIONS; неизвестное имя -> ValueError."""
    try:
        fn = CORRUPTIONS[corruption]
    except KeyError:
        raise ValueError(
            f'неизвестная порча {corruption!r}; допустимые: {sorted(CORRUPTIONS)}'
        ) from None
    return fn(images, severity)


def class_balanced_indices(labels, classes, n, seed):
    """Сбалансированные по классам ПОЗИЦИИ строк labels (не сами данные).

    Тот же алгоритм квот, что в `cv_grid.class_balanced_sample` (`n // k` на
    класс плюс +1 первым `n % k` классам, порядок — как в classes; выбор
    внутри класса — `default_rng(seed).permutation` одного общего
    генератора, последовательные классы продолжают его состояние), но
    реализован независимо — модуль не импортирует `cv_grid`. Возвращает
    индексы, а не строки пула: вызывающая сторона сама индексирует и
    эмбеддинги, и сырые изображения одним и тем же набором позиций (CRN).
    Нехватка строк у класса -> ValueError.
    """
    labels = np.asarray(labels)
    classes = list(classes)
    k = len(classes)
    if k == 0:
        raise ValueError('пустой список классов')
    quotas = [n // k + (1 if i < n % k else 0) for i in range(k)]
    rng = np.random.default_rng(seed)
    chunks = []
    for cls, quota in zip(classes, quotas):
        idx = np.flatnonzero(labels == cls)
        if len(idx) < quota:
            raise ValueError(
                f'класс {cls!r}: доступно {len(idx)} строк, требуется {quota}'
            )
        chunks.append(rng.permutation(idx)[:quota])
    return np.concatenate(chunks, axis=0)


def _seed_for_repeat(k):
    return SEED_P + SEED_STEP * k


def _format_severity(severity):
    return f'{severity:g}'


def _common_scan_kwargs(n, corruption, severity, k):
    return dict(
        dataset=DATASET_NAME,
        model=MODEL_NAME,
        representation=REPRESENTATION_NAME,
        pca_dim=PCA_DIM,
        n_P=n, n_Q=n,
        corruption=corruption,
        severity=severity,
        seed=k,
        metric_params=None,
    )


def build_c3_scan_configs(n, repeats=DEFAULT_REPEATS):
    """Этап-1: дешёвый MMD-скан — 4 порчи x 6 severity x repeats конфигов.

    experiment_id: `c3scan/{corruption}/s{severity}/r{k}`. P — clean-выборка
    (сбалансированный по классам набор индексов, seed 42 + 1000k), Q — та же
    выборка, но порчённая (CRN-дизайн, см. докстринг модуля); фактическое
    построение облаков — `build_pair`.
    """
    configs = []
    for corruption in sorted(CORRUPTIONS):
        for severity in SEVERITY_GRID:
            for k in range(repeats):
                eid = f'c3scan/{corruption}/s{_format_severity(severity)}/r{k}'
                configs.append(ExperimentConfig(
                    experiment_id=eid,
                    experiment_family='c3_scan',
                    **_common_scan_kwargs(n, corruption, severity, k),
                ))
    return configs


def select_matched_severities(scan_frame, target_mmd):
    """Для каждой порчи — severity с ближайшим средним MMD к target_mmd.

    Чистая функция над DataFrame строк schema (колонки 'corruption',
    'severity', 'mmd'): группировка по (corruption, severity), усреднение
    mmd по повторам, выбор severity с минимальным |среднее_mmd - target_mmd|.
    При равенстве расстояний выбирается МЕНЬШИЙ severity (severities
    перебираются по возрастанию, замена лучшего кандидата — только при
    строго меньшем расстоянии). Возвращает dict corruption -> (severity,
    mmd) с выбранным severity и достигнутым средним MMD.
    """
    result = {}
    for corruption, group in scan_frame.groupby('corruption'):
        by_severity = group.groupby('severity')['mmd'].mean()
        best_severity, best_mmd, best_diff = None, None, None
        for severity in sorted(by_severity.index):
            mmd_value = float(by_severity.loc[severity])
            diff = abs(mmd_value - target_mmd)
            if best_diff is None or diff < best_diff:
                best_severity, best_mmd, best_diff = float(severity), mmd_value, diff
        result[corruption] = (best_severity, best_mmd)
    return result


def build_c3_matched_configs(target_mmd, scan_rows, n, repeats=DEFAULT_REPEATS):
    """Этап-2: matched-MMD конфиги по результатам Этапа-1 + clean-контроль.

    Для каждой порчи, присутствующей в scan_rows, выбирается severity с
    ближайшим средним MMD к target_mmd (`select_matched_severities`);
    строится `repeats` конфигов `c3final/{corruption}/r{k}`
    (family='c3_final'). Плюс clean-контроль `c3final/clean/r{k}`
    (family='c3_final', corruption='clean', severity=None — порчи нет,
    значение из SEVERITY_GRID не применимо).
    """
    selected = select_matched_severities(scan_rows, target_mmd)
    configs = []
    for corruption in sorted(selected):
        severity, _mmd = selected[corruption]
        for k in range(repeats):
            configs.append(ExperimentConfig(
                experiment_id=f'c3final/{corruption}/r{k}',
                experiment_family='c3_final',
                **_common_scan_kwargs(n, corruption, severity, k),
            ))
    for k in range(repeats):
        configs.append(ExperimentConfig(
            experiment_id=f'c3final/clean/r{k}',
            experiment_family='c3_final',
            **_common_scan_kwargs(n, 'clean', None, k),
        ))
    return configs


def _parse_int(text, field, eid):
    try:
        return int(text)
    except ValueError:
        raise ValueError(
            f'непарсимое поле {field} {text!r} в experiment_id {eid!r}'
        ) from None


def _parse_float(text, field, eid):
    try:
        return float(text)
    except ValueError:
        raise ValueError(
            f'непарсимое поле {field} {text!r} в experiment_id {eid!r}'
        ) from None


def _parse_scan_corruption(name, eid):
    if name not in CORRUPTIONS:
        raise ValueError(
            f'неизвестная порча {name!r} в experiment_id {eid!r}; '
            f'допустимые: {sorted(CORRUPTIONS)}'
        )
    return name


def _parse_final_corruption(name, eid):
    if name != 'clean' and name not in CORRUPTIONS:
        raise ValueError(
            f'неизвестная порча {name!r} в experiment_id {eid!r}; '
            f'допустимые: {sorted(CORRUPTIONS) + ["clean"]}'
        )
    return name


def _parse_scan_severity(text, eid):
    severity = _parse_float(text, 'severity', eid)
    if not any(severity == allowed for allowed in SEVERITY_GRID):
        raise ValueError(
            f'несуществующее значение severity {severity!r} в experiment_id {eid!r}; '
            f'допустимые: {list(SEVERITY_GRID)}'
        )
    return severity


def parse_experiment_id(eid):
    """Разбор experiment_id (c3scan/c3final) в dict полей; мусор -> ValueError.

    c3scan: `c3scan/{corruption}/s{severity}/r{k}` ->
    {'experiment': 'c3scan', 'corruption', 'severity' (float), 'repeat' (int)}.
    c3final: `c3final/{corruption|clean}/r{k}` ->
    {'experiment': 'c3final', 'corruption', 'repeat' (int)}.
    Порча сверяется с реестром CORRUPTIONS (плюс 'clean' для c3final),
    severity — с сеткой SEVERITY_GRID: опечатка или несуществующее значение
    дают ValueError со списком допустимых, а не KeyError ниже по стеку;
    числовой/дробный суффикс парсится int()/float() с контекстной ошибкой
    при пустом или нечисловом значении.
    """
    parts = eid.split('/')
    family = parts[0]
    if family == 'c3scan':
        if len(parts) != 4 or not parts[2].startswith('s') or not parts[3].startswith('r'):
            raise ValueError(f'неизвестный experiment_id: {eid}')
        return {
            'experiment': 'c3scan',
            'corruption': _parse_scan_corruption(parts[1], eid),
            'severity': _parse_scan_severity(parts[2][1:], eid),
            'repeat': _parse_int(parts[3][1:], 'repeat', eid),
        }
    if family == 'c3final':
        if len(parts) != 3 or not parts[2].startswith('r'):
            raise ValueError(f'неизвестный experiment_id: {eid}')
        return {
            'experiment': 'c3final',
            'corruption': _parse_final_corruption(parts[1], eid),
            'repeat': _parse_int(parts[2][1:], 'repeat', eid),
        }
    raise ValueError(f'неизвестный experiment_id: {eid}')


def build_pair(config, clean_images, clean_embeddings_pca, labels):
    """(P_matrix, Q_images) для c3-конфигов (scan и final, включая clean).

    P — строки clean_embeddings_pca по сбалансированному набору индексов
    (class_balanced_indices, классы — все уникальные значения labels, seed
    42 + 1000·k, k — повтор из experiment_id). Q_images — СЫРЫЕ (не
    закодированные) изображения: для порчи corrupt_images(clean_images[idx],
    corruption, severity), для clean-контроля — clean_images[idx] напрямую
    (без вызова corrupt_images). Кодирование Q_images через CLIP+PCA —
    обязанность ран-скрипта оркестратора (модуль чистый, моделей не грузит).
    Несовпадение n_P/n_Q в конфиге -> ValueError (инвариант RTD |P|=|Q|).
    """
    if config.n_P != config.n_Q:
        raise ValueError(
            f'конфигурация {config.experiment_id!r}: n_P={config.n_P} != n_Q={config.n_Q}'
        )
    if config.corruption is None:
        raise ValueError(f'конфигурация {config.experiment_id!r} не задаёт corruption')
    clean_images = np.asarray(clean_images)
    clean_embeddings_pca = np.asarray(clean_embeddings_pca)
    labels = np.asarray(labels)
    if not (len(clean_images) == len(clean_embeddings_pca) == len(labels)):
        raise ValueError(
            'clean_images, clean_embeddings_pca и labels должны быть одной длины: '
            f'{len(clean_images)}, {len(clean_embeddings_pca)}, {len(labels)}'
        )
    parsed = parse_experiment_id(config.experiment_id)
    k = parsed['repeat']
    seed = _seed_for_repeat(k)
    classes = [int(c) for c in np.unique(labels)]
    idx = class_balanced_indices(labels, classes, config.n_P, seed)
    p = clean_embeddings_pca[idx]
    if config.corruption == 'clean':
        q_images = clean_images[idx]
    else:
        q_images = corrupt_images(clean_images[idx], config.corruption, config.severity)
    return p, q_images
