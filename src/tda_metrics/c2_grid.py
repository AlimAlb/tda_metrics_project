"""Конфигурации и облака эксперимента C2 «цикл вращения» (мастер-план §C2).

MNIST digit 3: N_DIGITS исходных картинок на сторону × 18 значений параметра
(сетка N_GRID_VALUES) — облака фиксированного размера n=450: RTD требует
|P| == |Q|, а значения метрик при разных n несравнимы. P — «данные» — всегда
полный круг вращения; Q — «модель» — вариант сравнения: полный круг
(same-law контроль cc), дуга 180° (arc), масштабный свип (zoom, незамкнутый
контроль) или арка [0, alpha_max) в скане замыкания цикла. Сиды — конвенция
проекта: P = 42 + 100·k, Q = 7 + 100·k, k — индекс независимого ресемпла
исходных цифр (k=0 даёт канонические 42/7).

Реестр C2_SPACES задаёт четыре пространства (pixels/CLIP/DINO с PCA-16 и
латент VAE без PCA). Модуль не вызывает ни эмбеддеры, ни метрики: сырые
матрицы представлений инъектируются через raw_fn в `project_pair` (заглушки
в тестах, кэш сырых облаков в ран-скрипте), PCA обучается на P-стороне
каждой пары («данные» — референс) и проецирует обе стороны одной моделью.
"""
from __future__ import annotations

import numpy as np

from tda_metrics.datasets import rotate_images, select_class, sweep_images, zoom_images
from tda_metrics.experiment_schema import ExperimentConfig
from tda_metrics.reduction import pca_fit, pca_transform

__all__ = [
    'ANGLES_ARC180', 'ANGLES_CIRCLE', 'C2_COMPS', 'C2_SPACES', 'DIGIT',
    'FULL_SPACES', 'N_DIGITS', 'N_GRID_VALUES', 'N_POINTS', 'RESAMPLES',
    'SCAN_ALPHA_MAX', 'SCAN_REPEATS', 'SCAN_SUBSET_ALPHAS', 'SCALES_ZOOM',
    'SEED_P', 'SEED_Q', 'SEED_STEP', 'SUBSET_SPACES',
    'angles_for', 'build_c2_resample_configs', 'build_c2_scan_configs',
    'cloud_images', 'pair_clouds', 'parse_c2_experiment_id',
    'project_pair', 'seed_p', 'seed_q',
]

C2_SPACES = {
    'pixels_pca16': dict(dataset='mnist', kind='pixels', pca=16, model=''),
    'clip_pca16':   dict(dataset='mnist', kind='clip',   pca=16, model='clip-vit-b32'),
    'dino_pca16':   dict(dataset='mnist', kind='dino',   pca=16, model='dinov2-vits14'),
    'vae_latent16': dict(dataset='mnist', kind='vae',    pca=None, model='conv-vae-16'),
}

FULL_SPACES = ('pixels_pca16', 'vae_latent16')
SUBSET_SPACES = ('clip_pca16', 'dino_pca16')

DIGIT = 3
N_DIGITS = 25
N_GRID_VALUES = 18
N_POINTS = N_DIGITS * N_GRID_VALUES

ANGLES_CIRCLE = np.arange(0.0, 360.0, 20.0)
ANGLES_ARC180 = np.arange(0.0, 180.0, 10.0)
SCALES_ZOOM = np.linspace(0.6, 1.4, N_GRID_VALUES)

C2_COMPS = ('cc', 'arc', 'zoom')
RESAMPLES = range(8)
SCAN_REPEATS = 3

SCAN_ALPHA_MAX = (180, 270, 315, 330, 345, 350, 360)
SCAN_SUBSET_ALPHAS = (180, 330, 360)

SEED_P = 42
SEED_Q = 7
SEED_STEP = 100


def seed_p(k):
    """Сид P-стороны («данные») ресемпла k: 42 + 100·k."""
    return SEED_P + SEED_STEP * k


def seed_q(k):
    """Сид Q-стороны («модель») ресемпла k: 7 + 100·k."""
    return SEED_Q + SEED_STEP * k


def _common_config_kwargs(space, info, k, **extra):
    return dict(
        dataset=info['dataset'],
        model=info['model'],
        representation=space,
        pca_dim=info['pca'],
        n_P=N_POINTS, n_Q=N_POINTS,
        seed=k,
        metric_params=None,
        **extra,
    )


def build_c2_resample_configs():
    """c2_resample: 8 ресемплов × 4 пространства × 3 сравнения (cc/arc/zoom) — 96.

    cc — same-law контроль (шумовой пол), arc — дуга 180° против полного
    круга, zoom — незамкнутый масштабный свип против круга.
    """
    configs = []
    for space, info in C2_SPACES.items():
        for comp in C2_COMPS:
            for k in RESAMPLES:
                configs.append(ExperimentConfig(
                    experiment_id=f'c2r/{space}/{comp}/r{k}',
                    experiment_family='c2_resample',
                    **_common_config_kwargs(space, info, k),
                ))
    return configs


def build_c2_scan_configs():
    """c2_scan: замыкание цикла — арка [0, alpha_max) против полного круга.

    pixels_pca16 и vae_latent16 (FULL_SPACES) — все 7 значений SCAN_ALPHA_MAX
    × 3 ресемпла (42); clip_pca16 и dino_pca16 (SUBSET_SPACES) — только
    SCAN_SUBSET_ALPHAS × 3 ресемпла (18); всего 60. severity — градус
    замыкания alpha_max (360 — same-law точка кривой).
    """
    configs = []
    for space, info in C2_SPACES.items():
        alphas = SCAN_ALPHA_MAX if space in FULL_SPACES else SCAN_SUBSET_ALPHAS
        for alpha_max in alphas:
            for k in range(SCAN_REPEATS):
                configs.append(ExperimentConfig(
                    experiment_id=f'c2s/{space}/a{alpha_max}/r{k}',
                    experiment_family='c2_scan',
                    severity=float(alpha_max),
                    **_common_config_kwargs(space, info, k),
                ))
    return configs


def _parse_space(name, eid):
    if name not in C2_SPACES:
        raise ValueError(
            f'неизвестное пространство {name!r} в experiment_id {eid!r}; '
            f'допустимые: {sorted(C2_SPACES)}'
        )
    return name


def _parse_int(text, field, eid):
    try:
        return int(text)
    except ValueError:
        raise ValueError(
            f'непарсимое поле {field} {text!r} в experiment_id {eid!r}'
        ) from None


def parse_c2_experiment_id(eid):
    """Разбор experiment_id (c2r/c2s) в dict полей; мусорный eid -> ValueError.

    c2r/{space}/{comp}/r{k} -> family='c2_resample', space, comp, repeat;
    c2s/{space}/a{alpha_max}/r{k} -> family='c2_scan', space, alpha_max,
    repeat. space сверяется с реестром C2_SPACES, comp — с C2_COMPS
    (опечатка — ValueError со списком допустимых, а не KeyError в
    pair_clouds); числовые суффиксы парсятся int()-ом с контекстной
    ошибкой (пустой или нечисловой repeat/alpha_max).
    """
    parts = eid.split('/')
    family = parts[0]
    if family == 'c2r':
        if len(parts) != 4 or not parts[3].startswith('r'):
            raise ValueError(f'неизвестный experiment_id: {eid}')
        if parts[2] not in C2_COMPS:
            raise ValueError(
                f'неизвестное сравнение {parts[2]!r} в experiment_id {eid!r}; '
                f'допустимые: {list(C2_COMPS)}'
            )
        return {
            'family': 'c2_resample',
            'space': _parse_space(parts[1], eid),
            'comp': parts[2],
            'repeat': _parse_int(parts[3][1:], 'repeat', eid),
        }
    if family == 'c2s':
        if len(parts) != 4 or not parts[2].startswith('a') or not parts[3].startswith('r'):
            raise ValueError(f'неизвестный experiment_id: {eid}')
        return {
            'family': 'c2_scan',
            'space': _parse_space(parts[1], eid),
            'alpha_max': _parse_int(parts[2][1:], 'alpha_max', eid),
            'repeat': _parse_int(parts[3][1:], 'repeat', eid),
        }
    raise ValueError(f'неизвестный experiment_id: {eid}')


def angles_for(cloud):
    """Сетка углов углового облака: имя сравнения, тип облака или alpha_max.

    'cc'/'circle' — полный круг (N_GRID_VALUES значений, шаг 20°);
    'arc'/'arc180' — дуга 180° (шаг 10°); 'arc{m}' или число m — арка
    [0, m) из N_GRID_VALUES значений (при m=360 совпадает с сеткой полного
    круга — same-law точка скана). Арка строится как j·m/18, а не
    np.arange(0, m, m/18): при нецелом шаге arange даёт 17-19 значений
    из-за округления, а фиксированное n — обязательное условие RTD.
    """
    if cloud in ('cc', 'circle'):
        return np.array(ANGLES_CIRCLE)
    if cloud in ('arc', 'arc180'):
        return np.array(ANGLES_ARC180)
    if isinstance(cloud, str):
        suffix = cloud[3:] if cloud.startswith('arc') else ''
        if not suffix.isdigit():
            raise ValueError(f'неизвестный тип облака для сетки углов: {cloud!r}')
        alpha = float(suffix)
    else:
        alpha = float(cloud)
    if not 0.0 < alpha <= 360.0:
        raise ValueError(f'alpha_max должен быть в (0, 360]: {cloud!r}')
    return alpha * np.arange(N_GRID_VALUES, dtype=float) / float(N_GRID_VALUES)


def cloud_images(images, labels, side_seed, n_digits=N_DIGITS, angles=None):
    """Облако картинок одной стороны: n_digits исходных цифр × сетка углов.

    Исходные цифры — select_class(digit=DIGIT, seed=side_seed), каждая
    поворачивается на каждое значение angles (sweep_images, порядок
    n-major); результат (n_digits * len(angles), 28, 28) uint8.
    angles=None — сетка полного круга. side_seed — готовый сид стороны
    (seed_p(k) / seed_q(k)).
    """
    if angles is None:
        angles = ANGLES_CIRCLE
    base = select_class(images, labels, DIGIT, n_digits, side_seed)
    return sweep_images(base, np.asarray(angles, dtype=float), rotate_images)


def project_pair(p_cloud, q_cloud, space, raw_fn, pca_dim=16):
    """(P_proj, Q_proj) одного сравнения в пространстве space.

    raw_fn(kind, cloud) -> сырая матрица (n, d) ДО PCA — инъекция
    представления: детерминированные заглушки в тестах, lookup кэша
    сырых облаков в ран-скрипте. Для пространств с pca в реестре PCA
    обучается на сырой матрице P («данные» — референс) через pca_fit и
    проецирует обе стороны одной моделью (замена Q не меняет модель);
    vae_latent16 — латенты как есть, без PCA. raw_fn обязан сохранить
    число строк облака — иначе ValueError (защита от промаха кэша).
    """
    if space not in C2_SPACES:
        raise ValueError(
            f'неизвестное пространство {space!r}; допустимые: {sorted(C2_SPACES)}'
        )
    info = C2_SPACES[space]
    kind = info['kind']
    p_raw = np.asarray(raw_fn(kind, p_cloud), dtype=np.float64)
    q_raw = np.asarray(raw_fn(kind, q_cloud), dtype=np.float64)
    if len(p_raw) != len(p_cloud) or len(q_raw) != len(q_cloud):
        raise ValueError(
            f'raw_fn изменил число строк: P {len(p_cloud)} -> {len(p_raw)}, '
            f'Q {len(q_cloud)} -> {len(q_raw)}'
        )
    if info['pca'] is None:
        return p_raw, q_raw
    model = pca_fit(p_raw, n_components=pca_dim)
    p_proj, _ = pca_transform(p_raw, model=model)
    q_proj, _ = pca_transform(q_raw, model=model)
    return p_proj, q_proj


def pair_clouds(images, labels, parsed, k, n_digits=N_DIGITS):
    """(P_cloud, Q_cloud) — картинки (n, 28, 28) по разобранному experiment_id.

    Сиды сторон: seed_p(k) / seed_q(k). P — всегда полный круг; Q — по
    разобранному сравнению: cc — полный круг, arc — дуга 180°, zoom —
    масштабный свип SCALES_ZOOM, c2_scan — арка angles_for(alpha_max).
    Все сетки состоят из N_GRID_VALUES значений, поэтому |P| == |Q| —
    обязательное условие RTD; расхождение — ValueError.
    """
    family = parsed.get('family')
    if family == 'c2_resample':
        q_cloud_type = parsed['comp']
    elif family == 'c2_scan':
        q_cloud_type = parsed['alpha_max']
    else:
        raise ValueError(f'неизвестное семейство: {family!r}')
    p = cloud_images(images, labels, seed_p(k), n_digits=n_digits,
                     angles=ANGLES_CIRCLE)
    if family == 'c2_resample' and q_cloud_type == 'zoom':
        base = select_class(images, labels, DIGIT, n_digits, seed_q(k))
        q = sweep_images(base, SCALES_ZOOM, zoom_images)
    else:
        q = cloud_images(images, labels, seed_q(k), n_digits=n_digits,
                         angles=angles_for(q_cloud_type))
    if len(p) != len(q):
        raise ValueError(
            f'облака разного размера: |P|={len(p)}, |Q|={len(q)} — '
            f'RTD требует |P| == |Q|'
        )
    return p, q
