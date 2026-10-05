"""Эмпирический Monte Carlo power-анализ метрик на смесях mode dropping/invention.

Запускается на Colab VM (heavy-стек: mtd/rtd/PR через TopologyMetrics).
Для каждого сценария эффекта (drop{d} или invent{i}) и каждого n из n-grid
генерируется блок control+effect пар (control — same-law, эффект — модель с
выброшенными/добавленными модами референса), прогоняется через
TopologyMetrics.compute_all, результат персистится как долгая таблица
(power_values.parquet) с manifest.json для resume по уже посчитанным
(n, effect)-блокам. В конце вызывается aggregate_power раздельно на каждый
эффект (control/effect пары разных сценариев несравнимы внутри одной
(n, metric)-группы — эффекты персистятся в общем файле только для удобства
хранения, агрегируются порознь; ошибка одного effect-свода не валит
остальные — traceback и продолжение) и печатается select_min_n при alpha
из фактической alpha-сетки — ближайшем к 0.05 из --alpha-grid (если 0.05
в сетке нет, используется ближайший alpha с явной пометкой в выводе).

Параметры прогона (n_control, n_effect, seed_step, skip_heavy и параметры
метрик rtd_trials/rtd_batch/nhood_sizes/js_k/rtd_mode/ntd_repeats)
фиксируются в manifest. При несовпадении сохранённых и текущих параметров
ok-блоки не переиспользуются (иначе таблица получит смешанную семантику —
одни (n, effect)-группы посчитаны со старыми параметрами, другие с
новыми): если в manifest есть блоки со статусом ok — диагностическая
печать и завершение прогона без записи (нужен новый --out-dir или
удаление старого); если ok-блоков нет, новые параметры просто
записываются и прогон продолжается.

Ошибка одного блока не прерывает прогон: блок помечается failed в manifest,
скрипт печатает трассировку и продолжает со следующим блоком. SystemExit не
поднимается нигде внутри логики прогона (только штатное завершение
интерпретатора); разбор аргументов командной строки использует argparse —
его собственные ошибки использования CLI (--help, отсутствующий обязательный
аргумент) остаются стандартным поведением argparse, это не относится к
обработке ошибок внутри самого прогона.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from tda_metrics.metrics import TopologyMetrics
from tda_metrics.mixtures import make_mode_reference, sample_q_dropping_invention, sample_reference
from tda_metrics.power import aggregate_power, empirical_power, select_min_n

SCHEMA_VERSION = 1
VALUES_FILE_PARQUET = 'power_values.parquet'
VALUES_FILE_CSV = 'power_values.csv'
MANIFEST_FILE = 'manifest.json'
SUMMARY_CSV = 'power_summary.csv'
SUMMARY_PARQUET = 'power_summary.parquet'

SEED_P_BASE = 42
SEED_Q_BASE = 7
EFFECT_SEED_OFFSET = 500000

_EFFECT_PATTERN = re.compile(r'^(drop|invent)([0-9]*\.?[0-9]+)$')


def parse_effect_label(label):
    """Разбор ярлыка эффекта вида 'drop0.1' / 'invent0.25' -> (kind, fraction)."""
    match = _EFFECT_PATTERN.match(label)
    if match is None:
        raise ValueError(
            f"эффект {label!r} не распознан: ожидается 'drop<frac>' или 'invent<frac>'"
        )
    kind, fraction_text = match.groups()
    return kind, float(fraction_text)


def make_control_sampler(ref, seed_step):
    """control: P, Q — независимые same-law выборки из общего референса.

    k — индекс повтора; seed_p = 42 + seed_step*k, seed_q = 7 + seed_step*k
    (конвенция проекта P/Q seed; seed_step — множитель, настраиваемый
    через --seed CLI-параметр, по умолчанию 1000).
    """
    def sampler(n, k):
        seed_p = SEED_P_BASE + seed_step * k
        seed_q = SEED_Q_BASE + seed_step * k
        p = sample_reference(ref, n, seed_p)
        q = sample_reference(ref, n, seed_q)
        return p, q

    return sampler


def make_effect_sampler(ref, seed_step, kind, fraction):
    """effect: P — same-law, Q — референс с dropping/invention заданной доли.

    Сид смещён на +500000 относительно control той же пары k: control и
    effect не парны (независимые облака, смещение исключает совпадение P
    между control и effect при одинаковом k).
    """
    def sampler(n, k):
        seed_p = SEED_P_BASE + seed_step * k + EFFECT_SEED_OFFSET
        seed_q = SEED_Q_BASE + seed_step * k + EFFECT_SEED_OFFSET
        p = sample_reference(ref, n, seed_p)
        if kind == 'drop':
            q = sample_q_dropping_invention(ref, n, seed_q, drop_fraction=fraction)
        else:
            q = sample_q_dropping_invention(ref, n, seed_q, invent_fraction=fraction)
        return p, q

    return sampler


def make_metrics_fn(skip_heavy):
    """metrics_fn для empirical_power: один TopologyMetrics на весь прогон.

    skip_heavy=True исключает rtd и ntd (самые дорогие метрики) — для
    дешёвых разведывательных прогонов по всей сетке n.

    Параметры метрик — конвенция серии проекта (как в синтетической фазе):
    rtd_trials=2, rtd_batch=500; фиксируются в manifest прогона.
    """
    metrics = TopologyMetrics()
    skip = ('rtd', 'ntd') if skip_heavy else ()

    def fn(p, q):
        return metrics.compute_all(
            p, q, rtd_trials=2, rtd_batch=500, skip=skip,
        )

    return fn


def _pyarrow_available():
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return False
    return True


def _atomic_write_bytes(path, data):
    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + '.', suffix='.tmp')
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


def load_existing_values(out_dir):
    """Загрузить уже накопленную долгую таблицу значений (parquet либо csv)."""
    parquet_path = out_dir / VALUES_FILE_PARQUET
    csv_path = out_dir / VALUES_FILE_CSV
    if parquet_path.exists():
        return pd.read_parquet(parquet_path, engine='pyarrow'), 'parquet'
    if csv_path.exists():
        return pd.read_csv(csv_path), 'csv'
    columns = ['n', 'kind', 'repeat', 'metric', 'value', 'effect']
    return pd.DataFrame(columns=columns), ('parquet' if _pyarrow_available() else 'csv')


def persist_values(out_dir, frame, fmt):
    """Атомарная перезапись долгой таблицы значений на диск."""
    if fmt == 'parquet' and _pyarrow_available():
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.Table.from_pandas(frame, preserve_index=False)
        sink = pa.BufferOutputStream()
        pq.write_table(table, sink)
        _atomic_write_bytes(out_dir / VALUES_FILE_PARQUET, sink.getvalue().to_pybytes())
    else:
        _atomic_write_bytes(out_dir / VALUES_FILE_CSV, frame.to_csv(index=False).encode('utf-8'))


def load_manifest(out_dir):
    path = out_dir / MANIFEST_FILE
    if not path.exists():
        return {'schema_version': SCHEMA_VERSION, 'params': {}, 'blocks': {}, 'n_rows': 0}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        print(f'manifest повреждён, не читается как JSON: {exc}; начинаю с пустого manifest '
              '(данные на диске не трогаю, но дубли блоков придётся разбирать вручную)')
        return {'schema_version': SCHEMA_VERSION, 'params': {}, 'blocks': {}, 'n_rows': 0}


def persist_manifest(out_dir, manifest):
    manifest['updated_utc'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    data = json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8')
    _atomic_write_bytes(out_dir / MANIFEST_FILE, data)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Эмпирический Monte Carlo power-анализ метрик (mode dropping/invention).'
    )
    parser.add_argument('--n-grid', type=int, nargs='+', default=[100, 250, 500, 1000])
    parser.add_argument('--n-control', type=int, default=10)
    parser.add_argument('--n-effect', type=int, default=10)
    parser.add_argument(
        '--effects', type=str, nargs='+',
        default=['drop0.1', 'drop0.25', 'invent0.15', 'invent0.5'],
    )
    parser.add_argument('--alpha-grid', type=float, nargs='+', default=[0.05, 0.10, 0.20])
    parser.add_argument('--seed', type=int, default=1000, help='шаг seed-формулы 42+seed*k / 7+seed*k')
    parser.add_argument('--out-dir', type=str, default='results/raw/power')
    parser.add_argument('--skip-heavy', action='store_true', help='прогон без rtd и ntd')
    return parser


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = make_mode_reference(n_modes=8)
    metrics_fn = make_metrics_fn(args.skip_heavy)
    alpha_grid = tuple(args.alpha_grid)

    values_frame, fmt = load_existing_values(out_dir)
    manifest = load_manifest(out_dir)
    manifest.setdefault('blocks', {})

    current_params = {
        'n_control': args.n_control,
        'n_effect': args.n_effect,
        'seed_step': args.seed,
        'skip_heavy': args.skip_heavy,
        'rtd_trials': 2,
        'rtd_batch': 500,
        'nhood_sizes': [1, 3, 10],
        'js_k': 5,
        'rtd_mode': 'random-coupling-exploratory',
        'ntd_repeats': 5,
    }
    stored_params = manifest.get('params') or {}
    if stored_params and stored_params != current_params:
        has_ok_blocks = any(
            block.get('status') == 'ok' for block in manifest['blocks'].values()
        )
        if has_ok_blocks:
            print('ОШИБКА: параметры прогона отличаются от сохранённых в manifest '
                  f'(сохранено={stored_params}, сейчас={current_params}), и в manifest '
                  'есть блоки со статусом ok — переиспользовать их с другими параметрами '
                  'нельзя: таблица получит смешанную семантику (одни (n, effect)-группы '
                  'посчитаны со старыми параметрами, другие с новыми). Укажите новый '
                  f'--out-dir или удалите старый: {out_dir}. Прогон завершён без записи '
                  'данных.')
            return
        print(f'Параметры прогона отличаются от сохранённых в manifest '
              f'(сохранено={stored_params}), но ok-блоков нет — записываю новые '
              'параметры и продолжаю')
    manifest['params'] = current_params

    for effect_label in args.effects:
        kind, fraction = parse_effect_label(effect_label)
        effect_sampler = make_effect_sampler(ref, args.seed, kind, fraction)
        control_sampler = make_control_sampler(ref, args.seed)

        for n in args.n_grid:
            block_key = f'n{n}_{effect_label}'
            existing = manifest['blocks'].get(block_key)
            if existing is not None and existing.get('status') == 'ok':
                print(f'[{block_key}] уже посчитан ({existing.get("n_rows")} строк) — пропуск')
                continue

            print(f'[{block_key}] старт: n={n}, effect={effect_label}, '
                  f'n_control={args.n_control}, n_effect={args.n_effect}')
            started = time.perf_counter()
            try:
                block_frame = empirical_power(
                    effect_sampler, control_sampler, metrics_fn,
                    n_grid=(n,), n_control=args.n_control, n_effect=args.n_effect,
                    seed=0, alpha_grid=alpha_grid,
                )
                block_frame = block_frame.copy()
                block_frame['effect'] = effect_label

                values_frame = values_frame[
                    ~((values_frame['n'] == n) & (values_frame['effect'] == effect_label))
                ]
                values_frame = pd.concat([values_frame, block_frame], ignore_index=True)
                persist_values(out_dir, values_frame, fmt)

                elapsed = time.perf_counter() - started
                manifest['blocks'][block_key] = {
                    'n': n, 'effect': effect_label, 'status': 'ok',
                    'n_rows': int(len(block_frame)), 'seconds': elapsed, 'error': None,
                }
                manifest['n_rows'] = int(len(values_frame))
                persist_manifest(out_dir, manifest)
                print(f'[{block_key}] готово: {len(block_frame)} строк за {elapsed:.1f} c')
            except Exception as exc:
                elapsed = time.perf_counter() - started
                traceback.print_exc()
                manifest['blocks'][block_key] = {
                    'n': n, 'effect': effect_label, 'status': 'failed',
                    'n_rows': 0, 'seconds': elapsed,
                    'error': f'{type(exc).__name__}: {exc}',
                }
                persist_manifest(out_dir, manifest)
                print(f'[{block_key}] FAILED за {elapsed:.1f} c: {type(exc).__name__}: {exc}')

    print('\n=== агрегация power_summary ===')
    summary_frames = []
    for effect_label in args.effects:
        subset = values_frame[values_frame['effect'] == effect_label]
        if subset.empty:
            print(f'{effect_label}: нет данных (все блоки failed или пропущены) — пропуск агрегации')
            continue
        try:
            raw_columns = ['n', 'kind', 'repeat', 'metric', 'value']
            aggregated = aggregate_power(subset[raw_columns], alpha_grid=alpha_grid)
            aggregated = aggregated.copy()
            aggregated['effect'] = effect_label
            summary_frames.append(aggregated)
        except Exception as exc:
            traceback.print_exc()
            print(f'{effect_label}: агрегация не удалась ({type(exc).__name__}: {exc}) — '
                  'сводка этого эффекта пропущена, остальные продолжаются')

    if summary_frames:
        summary = pd.concat(summary_frames, ignore_index=True)
    else:
        summary = pd.DataFrame(columns=[
            'n', 'metric', 'alpha', 'power', 'threshold', 'n_control', 'n_effect',
            'mean_control', 'sd_control', 'mean_effect', 'z_mean_effect',
            'target_met', 'effect',
        ])

    summary.to_csv(out_dir / SUMMARY_CSV, index=False)
    if _pyarrow_available():
        summary.to_parquet(out_dir / SUMMARY_PARQUET, engine='pyarrow', index=False)
    print(f'power_summary сохранён: {out_dir / SUMMARY_CSV}'
          + (f' и {out_dir / SUMMARY_PARQUET}' if _pyarrow_available() else ''))

    report_alpha = min(alpha_grid, key=lambda a: abs(a - 0.05))
    if not np.isclose(report_alpha, 0.05):
        print(f'ВНИМАНИЕ: alpha=0.05 отсутствует в alpha-grid {list(alpha_grid)}; '
              f'select_min_n считается при ближайшем alpha={report_alpha}')
    print(f'\n=== select_min_n (alpha={report_alpha}) по сценариям эффекта ===')
    for effect_label in args.effects:
        effect_summary = summary[summary['effect'] == effect_label]
        if effect_summary.empty:
            print(f'{effect_label}: нет данных для select_min_n')
            continue
        min_n = select_min_n(effect_summary, target_power=0.8, alpha=report_alpha)
        print(f'{effect_label}: {min_n}')

    print('\nИТОГ: блоков в manifest =', len(manifest['blocks']),
          '| строк в values =', len(values_frame),
          '| failed =', sum(1 for b in manifest['blocks'].values() if b.get('status') == 'failed'))


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
