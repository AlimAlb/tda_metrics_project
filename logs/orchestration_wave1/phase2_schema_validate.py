"""Валидация Фазы 2 (schema): окружение -> pip -> pytest -> smoke ResultStore.

Мега-скрипт одной ячейкой по образцу logs/orchestration_phase1/phase1_setup.py.
Тесты и smoke чисто механические (heavy-стек mtd/rtd/PR не нужен).
Запускает оркестратор на Colab-VM после завершения ветки B.
SystemExit не поднимается никогда: только print и естественное завершение.
Вывод дублируется в /content/phase2_schema_log.txt (Tee).
"""
import json
import os
import shutil
import subprocess
import sys
import traceback

LOG_PATH = '/content/phase2_schema_log.txt'
STORE_DIR = '/content/phase2_store'
REPO_DIR = '/content/tda_metrics_repo'
ZIP_PATH = '/content/tda_metrics_src.zip'


class Tee:
    def __init__(self, path):
        self.file = open(path, 'w', encoding='utf-8')

    def write(self, text):
        sys.__stdout__.write(text)
        self.file.write(text)

    def flush(self):
        sys.__stdout__.flush()
        self.file.flush()


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def pip_install(*args):
    r = run([sys.executable, '-m', 'pip', 'install', *args])
    tail = (r.stdout + r.stderr).strip().splitlines()
    print(f"pip install {' '.join(args)} -> rc={r.returncode}")
    for line in tail[-3:]:
        print('   ', line)
    if r.returncode != 0:
        raise RuntimeError(f'pip install failed: {args}')


sys.stdout = Tee(LOG_PATH)

results = {}

# ============ Секция 1: окружение (до установки) ============
try:
    print('=== [1] ENV (до установки) ===')
    print('python:', sys.version)
    for pkg in ('numpy', 'pandas', 'pyarrow', 'pytest'):
        r = run([sys.executable, '-c', f'import {pkg}; print({pkg}.__version__)'])
        print(f'{pkg}:', r.stdout.strip() if r.returncode == 0 else 'ОТСУТСТВУЕТ')
    heavy = []
    for pkg in ('tensorflow', 'mtd', 'rtd', 'precision_recall'):
        r = run([sys.executable, '-c', f'import {pkg}'])
        heavy.append(f'{pkg}={"OK" if r.returncode == 0 else "НЕТ"}')
    print('heavy-стек (нужен пакету на уровне импорта):', ', '.join(heavy))
    if any(msg.endswith('НЕТ') for msg in heavy):
        print('ВНИМАНИЕ: heavy-стек неполон — сначала phase1_setup секции 2-5, '
              'иначе import tda_metrics упадёт на этапе collection/pytest')
    results['env'] = 'OK'
except Exception:
    results['env'] = 'FAIL'
    traceback.print_exc(file=sys.stdout)

# ============ Секция 2: зависимости ============
try:
    print('\n=== [2] pip install pandas pyarrow pytest ===')
    pip_install('pandas', 'pyarrow', 'pytest')
    r = run([sys.executable, '-c', 'import pyarrow; print(pyarrow.__version__)'])
    print('pyarrow после установки:', r.stdout.strip() or f'FAIL {r.stderr.strip()[-200:]}')
    results['deps'] = 'OK'
except Exception:
    results['deps'] = 'FAIL'
    traceback.print_exc(file=sys.stdout)

# ============ Секция 3: пакет tda_metrics из zip ============
try:
    print('\n=== [3] tda_metrics (unzip -o + install) ===')
    if not os.path.exists(ZIP_PATH):
        raise RuntimeError(f'{ZIP_PATH} не найден: сначала colab upload')
    r = run(['unzip', '-q', '-o', ZIP_PATH, '-d', REPO_DIR])
    print('unzip -o ->', r.returncode, r.stderr.strip()[:200])
    r = run(['unzip', '-l', ZIP_PATH])
    for line in (r.stdout or '').strip().splitlines():
        if 'experiment_schema' in line or line.strip().endswith('pyproject.toml') \
                or line.strip().endswith('README.md') or 'test_experiment_schema' in line:
            print('   ', line.strip())
    pip_install('-q', '--force-reinstall', '--no-deps', REPO_DIR)
    r = run([sys.executable, '-c',
             'import tda_metrics.experiment_schema as es; print("import OK, SCHEMA_VERSION", es.SCHEMA_VERSION)'])
    print(r.stdout.strip() or f'FAIL {r.stderr.strip()[-300:]}')
    results['package'] = 'OK'
except Exception:
    results['package'] = 'FAIL'
    traceback.print_exc(file=sys.stdout)

# ============ Секция 4: pytest (только механический файл schema) ============
try:
    print('\n=== [4] PYTEST tests/test_experiment_schema.py ===')
    r = run([sys.executable, '-m', 'pytest', 'tests/test_experiment_schema.py', '-q', '--tb=short'],
            cwd=REPO_DIR, timeout=600)
    tail = (r.stdout + r.stderr).strip().splitlines()
    for line in tail[-15:]:
        print(line)
    results['pytest'] = tail[-1] if tail else 'NO OUTPUT'
    with open('/content/phase2_pytest_out.txt', 'w') as f:
        f.write(r.stdout + '\n=== STDERR ===\n' + r.stderr)
except Exception:
    results['pytest'] = 'FAIL'
    traceback.print_exc(file=sys.stdout)

# ============ Секция 5: smoke ResultStore mini-grid ============
try:
    print('\n=== [5] SMOKE: ResultStore mini-grid (6 конфигураций, 1 падает, resume) ===')
    from tda_metrics.experiment_schema import (
        ExperimentConfig,
        ResultStore,
        DuplicateKeyError,
        ManifestMismatchError,
        row_from_compute_all,
        run_grid,
    )

    shutil.rmtree(STORE_DIR, ignore_errors=True)

    def make_config(i):
        return ExperimentConfig(
            experiment_id='phase2_smoke',
            experiment_family='synthetic',
            dataset='ring',
            model='toy',
            representation='raw',
            n_P=900,
            n_Q=900,
            seed=42,
            alpha=i / 10,
        )

    configs = [make_config(i) for i in range(6)]
    calls = []

    def runner(config):
        calls.append(config.alpha)
        if config.alpha == 0.5:
            raise RuntimeError('искусственный сбой конфигурации alpha=0.5')
        return {
            'mtd_PQ': 1.0, 'mtd_QP': 1.1, 'ntd_PQ': 1.2, 'ntd_QP': 1.3, 'rtd': 35.0,
            'mmd': 0.1, 'frechet': 0.2, 'js': 0.3,
            'precision@1': 0.9, 'recall@1': 0.8,
            'precision@3': 0.7, 'recall@3': 0.6,
            'precision@10': 0.5, 'recall@10': 0.4,
        }

    def healed(config):
        return {'mmd': 0.0}

    store = ResultStore(STORE_DIR)
    written = run_grid(store, configs, runner)
    n_failed = sum(1 for row in written if row.status == 'failed')
    print(f'первый прогон: записано {len(written)} строк, из них failed={n_failed}, '
          f'runner вызван {len(calls)} раз')
    assert len(written) == 6 and n_failed == 1 and len(store) == 6

    written2 = run_grid(store, configs, runner)
    print(f'повторный запуск (resume): записано {len(written2)} строк, runner вызван {len(calls)} раз суммарно')
    assert len(written2) == 0 and len(calls) == 6 and len(store) == 6

    written3 = run_grid(store, configs, healed, retry_failed=True)
    print(f'retry_failed с «лечением»: перезаписано {len(written3)} строк, '
          f'status={written3[0].status}, mmd={written3[0].mmd}')
    assert len(written3) == 1 and written3[0].status == 'ok' and store.failed_keys() == []

    try:
        store.append(row_from_compute_all(configs[0], {'mmd': 9.9}))
        raise AssertionError('дубликат не обнаружен')
    except DuplicateKeyError as exc:
        print(f'duplicate key detection: пойман DuplicateKeyError: {str(exc)[:120]}')

    manifest_path = os.path.join(STORE_DIR, 'manifest.json')
    manifest = json.loads(open(manifest_path, encoding='utf-8').read())
    first_key = next(iter(manifest['rows']))
    del manifest['rows'][first_key]
    manifest['n_rows'] = len(manifest['rows'])
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f)
    try:
        ResultStore(STORE_DIR)
        raise AssertionError('manifest mismatch не обнаружен')
    except ManifestMismatchError as exc:
        print(f'manifest mismatch detection: пойман ManifestMismatchError: {str(exc)[:120]}')
    store = ResultStore.rebuild_manifest(STORE_DIR)
    print(f'rebuild_manifest: хранилище открыто, строк={len(store)}, '
          f'статусы={sorted(set(r.status for r in store.rows()))}')
    assert len(store) == 6

    frame = store.load_frame()
    print('load_frame:', frame.shape, 'колонок; статус-счётчик:',
          frame['status'].value_counts().to_dict())
    print(frame[['alpha', 'status', 'error_message']].to_string(index=False))

    summary = {
        'n_configs': len(configs),
        'n_rows': len(store),
        'first_pass_written': len(written),
        'first_pass_failed': n_failed,
        'resume_added': len(written2),
        'retry_fixed': len(written3),
        'duplicate_detected': True,
        'mismatch_detected': True,
        'rebuilt_rows': len(store),
    }
    with open('/content/phase2_smoke_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print('smoke summary:', json.dumps(summary, ensure_ascii=False))
    results['smoke'] = 'OK'
except Exception:
    results['smoke'] = 'FAIL'
    traceback.print_exc(file=sys.stdout)

print('\n=== ИТОГ ФАЗЫ 2 (schema, ветка A) ===')
for key, value in results.items():
    print(f'  {key}: {value}')
sys.stdout.flush()
