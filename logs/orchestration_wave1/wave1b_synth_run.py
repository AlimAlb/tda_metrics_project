"""Прогон синтетических экспериментов S0/S1/S3/S4 (мастер-план §Часть I) по секциям.

Секции для запуска читаются из /content/wave1b_sections.txt (пробелы/переводы
строк: s0 s1samelaw s1 s3 s4) — это позволяет гнать длинную серию
последовательными exec-вызовами с resume через ResultStore. Запускать на VM
после зелёной wave1b_validate.py (setup стоит). Store:
/content/results/raw/synthetic — скачать немедленно после DONE.
SystemExit не поднимается; Tee (append) в /content/wave1b_synth_log.txt.
"""
import sys
import time
import traceback

LOG_PATH = '/content/wave1b_synth_log.txt'
PROGRESS_PATH = '/content/progress.txt'
STORE_DIR = '/content/results/raw/synthetic'
SECTIONS_FILE = '/content/wave1b_sections.txt'
ALL_SECTIONS = ('s0', 's1samelaw', 's1', 's3', 's4')
PROGRESS_EVERY = 10


class Tee:
    def __init__(self, path):
        self.file = open(path, 'a', encoding='utf-8')

    def write(self, text):
        sys.__stdout__.write(text)
        self.file.write(text)

    def flush(self):
        sys.__stdout__.flush()
        self.file.flush()


def progress_line(text):
    """Дублирует строку прогресса в отдельный tail-able файл на VM."""
    try:
        with open(PROGRESS_PATH, 'a', encoding='utf-8') as f:
            f.write(f'[{time.strftime("%H:%M:%S")}] {text}\n')
    except Exception:
        pass


sys.stdout = Tee(LOG_PATH)

print('=== WAVE1B SYNTH RUN (секция), старт', time.strftime('%Y-%m-%d %H:%M:%S'), '===')

results = {}

try:
    with open(SECTIONS_FILE) as f:
        requested = [s for s in f.read().split() if s]
    unknown = [s for s in requested if s not in ALL_SECTIONS]
    if unknown:
        raise ValueError(f'неизвестные секции: {unknown}')
    print('запрошенные секции:', requested)
    results['sections'] = 'OK'
except Exception:
    results['sections'] = 'FAIL'
    traceback.print_exc(file=sys.stdout)

try:
    from tda_metrics import TopologyMetrics
    from tda_metrics.experiment_schema import ResultStore, run_grid
    from tda_metrics.synthetic_grid import (
        build_s0_configs,
        build_s1_configs,
        build_s1_samelaw_configs,
        build_s3_configs,
        build_s4_configs,
        make_runner,
    )
    metrics = TopologyMetrics(seed=42, device='cuda')
    store = ResultStore(STORE_DIR)
    print('store открыт, строк уже в store:', len(store))
    runner = make_runner(lambda p, q: metrics.compute_all(p, q, rtd_trials=2, rtd_batch=500))
    results['imports'] = 'OK'
except Exception:
    results['imports'] = 'FAIL'
    traceback.print_exc(file=sys.stdout)

BUILDERS = {
    's0': build_s0_configs,
    's1samelaw': build_s1_samelaw_configs,
    's1': build_s1_configs,
    's3': build_s3_configs,
    's4': build_s4_configs,
}

total_new = 0
if results.get('sections') == 'OK' and results.get('imports') == 'OK':
    for name in requested:
        started = time.perf_counter()
        try:
            configs = BUILDERS[name]()
            total = len(configs)
            done_before = sum(1 for c in configs if store.contains(c))
            print(f'\n=== [{name}] {total} конфигураций; записано до запуска: {done_before} ===')
            progress_line(f'[{name}] старт: всего {total}, resume пропустит {done_before}')

            counter = {'n': 0, 'failed': 0}
            section_started = time.perf_counter()

            def progress_runner(config, _runner=runner, _name=name, _total=total, _done=done_before):
                t0 = time.perf_counter()
                try:
                    result = _runner(config)
                except Exception as exc:
                    counter['n'] += 1
                    counter['failed'] += 1
                    note = (f'[{_name}] {_done + counter["n"]}/{_total} FAILED '
                            f'{config.experiment_id} ({exc.__class__.__name__}: {str(exc)[:120]})')
                    print(note)
                    progress_line(note)
                    raise
                counter['n'] += 1
                elapsed = time.perf_counter() - section_started
                done = _done + counter['n']
                if counter['n'] % PROGRESS_EVERY == 0:
                    rate = counter['n'] / elapsed if elapsed > 0 else 0.0
                    eta = (_total - done) / rate / 60 if rate > 0 else float('nan')
                    note = (f'[{_name}] {done}/{_total} | последний: {config.experiment_id}'
                            f' | {time.perf_counter() - t0:.0f}s/строка'
                            f' | {rate * 60:.1f} строк/мин | ETA {eta:.0f} мин'
                            f' | failed: {counter["failed"]}')
                    print(note)
                    progress_line(note)
                return result

            written = run_grid(store, configs, progress_runner)
            failed = [row for row in written if row.status == 'failed']
            for row in failed:
                print(f'  FAILED {row.experiment_id}: {row.error_message[:200]}')
            elapsed = time.perf_counter() - started
            total_new += len(written)
            print(f'[{name}] записано {len(written)} строк ({len(failed)} failed) за '
                  f'{elapsed / 60:.1f} мин; итого в store {len(store)}')
            progress_line(f'[{name}] СЕКЦИЯ ЗАВЕРШЕНА: +{len(written)} строк, failed={len(failed)}, '
                          f'за {elapsed / 60:.1f} мин, итого в store {len(store)}')
            results[name] = f'OK ({len(written)} новых, {len(failed)} failed, {elapsed / 60:.1f} мин)'
        except Exception:
            results[name] = 'FAIL'
            progress_line(f'[{name}] СЕКЦИЯ УПАЛА (см. wave1b_synth_log.txt)')
            traceback.print_exc(file=sys.stdout)

try:
    frame = store.load_frame()
    print('\n=== ИТОГ ===')
    print('строк в store:', len(frame))
    print('статусы:', frame['status'].value_counts().to_dict())
    print('эксперименты:', frame['experiment_id'].str.split('_').str[0].value_counts().to_dict())
    frame.to_csv('/content/results/raw/synthetic_summary.csv', index=False)
    print('summary CSV сохранён')
    results['summary'] = 'OK'
except Exception:
    results['summary'] = 'FAIL'
    traceback.print_exc(file=sys.stdout)

print('\n=== ВЕРДИКТ СЕКЦИОННОГО ПРОГОНА ===')
for key, value in results.items():
    print(f'  {key}: {value}')
print('total новых строк:', total_new)
progress_line(f'FINISHED: total новых строк {total_new}; вердикты: ' + '; '.join(f'{k}={v}' for k, v in results.items()))
print('FINISHED', time.strftime('%Y-%m-%d %H:%M:%S'))
sys.stdout.flush()