"""Валидация Wave 1 (ветки A+B): setup -> канонический pytest (66 + новые) -> smoke ResultStore.

Мега-скрипт одной ячейкой: секции 1-7 из phase1_setup.py, pytest — канонической
командой D-002 (models-первым) с добавленными в КОНЕЦ очереди новыми файлами
веток A (test_experiment_schema.py) и B (test_nlp_datasets.py), затем smoke
ResultStore mini-grid из phase2_schema_validate.py (секция 5).
SystemExit не поднимается никогда. Вывод — Tee в /content/wave1_log.txt.
"""
import subprocess
import sys
import os
import json
import platform
import traceback

LOG_PATH = '/content/wave1_log.txt'
STORE_DIR = '/content/wave1_store'
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

# ============ Секция 1: окружение ============
try:
    print('=== [1] ENV ===')
    print('python:', sys.version)
    r = run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv'])
    print('nvidia-smi:', r.stdout.strip() or r.stderr.strip())
    r = run(['g++', '--version'])
    print('g++:', r.stdout.strip().splitlines()[0] if r.stdout else 'not found')
    results['env'] = 'OK'
except Exception:
    results['env'] = 'FAIL'
    traceback.print_exc(file=sys.stdout)

# ============ Секция 2: ripser++ с C++17-патчем ============
try:
    print('\n=== [2] ripser++ (clone + C++17 patch + build) ===')
    if not os.path.isdir('/content/ripser-plusplus'):
        r = run(['git', 'clone', '--recursive', '-q',
                 'https://github.com/simonzhang00/ripser-plusplus.git', '/content/ripser-plusplus'])
        print('git clone ripser-plusplus ->', r.returncode)
    cml = '/content/ripser-plusplus/CMakeLists.txt'
    src = open(cml).read()
    if 'c++17' not in src:
        patched = src.replace('-std=c++11', '-std=c++17').replace(
            'set(CMAKE_CXX_STANDARD 11)', 'set(CMAKE_CXX_STANDARD 17)')
        open(cml, 'w').write(patched)
        print('CMakeLists: c++11 -> c++17 APPLIED')
    else:
        print('CMakeLists: c++17 уже присутствует (патч idempotent)')
    pip_install('-q', '/content/ripser-plusplus')
    r = run([sys.executable, '-c', 'import ripserplusplus as rpp; print("ripser++ import OK")'])
    print(r.stdout.strip() or f'FAIL {r.stderr.strip()[-300:]}')
    results['ripser_patch'] = 'OK'
except Exception:
    results['ripser_patch'] = 'FAIL'
    traceback.print_exc(file=sys.stdout)

# ============ Секция 3: MTopDiv ============
try:
    print('\n=== [3] MTopDiv ===')
    if not os.path.isdir('/content/MTopDiv'):
        r = run(['git', 'clone', '-q', 'https://github.com/IlyaTrofimov/MTopDiv.git', '/content/MTopDiv'])
        print('git clone MTopDiv ->', r.returncode)
    pip_install('-q', '/content/MTopDiv')
    results['mtopdiv'] = 'OK'
except Exception:
    results['mtopdiv'] = 'FAIL'
    traceback.print_exc(file=sys.stdout)

# ============ Секция 4: RTD ============
try:
    print('\n=== [4] RTD ===')
    pip_install('-q', 'git+https://github.com/IlyaTrofimov/RTD.git')
    results['rtd'] = 'OK'
except Exception:
    results['rtd'] = 'FAIL'
    traceback.print_exc(file=sys.stdout)

# ============ Секция 5: improved precision/recall (форк) ============
try:
    print('\n=== [5] improved precision/recall ===')
    repo = '/content/improved-precision-and-recall-metric'
    if not os.path.isdir(repo):
        r = run(['git', 'clone', '-q',
                 'https://github.com/AlimAlb/improved-precision-and-recall-metric.git', repo])
        print('git clone PR-fork ->', r.returncode)
    pip_install('-q', repo)
    results['pr'] = 'OK'
except Exception:
    results['pr'] = 'FAIL'
    traceback.print_exc(file=sys.stdout)

# ============ Секция 6: пакет tda_metrics из zip (ветки A+B) ============
try:
    print('\n=== [6] tda_metrics (unzip -o + force-reinstall) ===')
    if not os.path.exists(ZIP_PATH):
        raise RuntimeError(f'{ZIP_PATH} не найден: сначала colab upload')
    r = run(['unzip', '-q', '-o', ZIP_PATH, '-d', REPO_DIR])
    print('unzip -o ->', r.returncode, r.stderr.strip()[:200])
    r = run(['unzip', '-l', ZIP_PATH])
    new_files = [l.strip() for l in (r.stdout or '').strip().splitlines()
                 if 'experiment_schema' in l or 'nlp_datasets' in l]
    for line in new_files:
        print('   NEW:', line)
    pip_install('-q', 'pandas', 'pyarrow', 'pytest')
    pip_install('-q', '--force-reinstall', '--no-deps', REPO_DIR)
    r = run([sys.executable, '-c',
             'import tda_metrics.experiment_schema as es; import tda_metrics.nlp_datasets as nd;'
             ' print("import OK, SCHEMA_VERSION", es.SCHEMA_VERSION)'])
    print(r.stdout.strip() or f'FAIL {r.stderr.strip()[-300:]}')
    results['package'] = 'OK'
except Exception:
    results['package'] = 'FAIL'
    traceback.print_exc(file=sys.stdout)

# ============ Секция 7: импорты mtd/rtd ============
try:
    print('\n=== [7] IMPORTS ===')
    import mtd
    print('import mtd OK')
    import rtd
    print('import rtd OK')
    import tda_metrics
    print('import tda_metrics OK')
    results['imports'] = 'OK'
except Exception:
    results['imports'] = 'FAIL'
    traceback.print_exc(file=sys.stdout)

# ============ Секция 8: pytest — канонический порядок D-002 + новые файлы в конец ============
try:
    print('\n=== [8] PYTEST (канонический порядок + ветки A/B) ===')
    cmd = [sys.executable, '-m', 'pytest',
           'tests/test_models.py', 'tests/test_metrics.py', 'tests/test_datasets.py',
           'tests/test_experiments.py', 'tests/test_reduction.py', 'tests/test_samplers.py',
           'tests/test_experiment_schema.py', 'tests/test_nlp_datasets.py',
           '-q', '--tb=short']
    r = run(cmd, cwd=REPO_DIR, timeout=1800)
    tail = (r.stdout + r.stderr).strip().splitlines()
    for line in tail[-25:]:
        print(line)
    results['pytest'] = tail[-1] if tail else 'NO OUTPUT'
    with open('/content/wave1_pytest_out.txt', 'w') as f:
        f.write(r.stdout + '\n=== STDERR ===\n' + r.stderr)
except Exception:
    results['pytest'] = 'FAIL'
    traceback.print_exc(file=sys.stdout)

# ============ Секция 9: smoke ResultStore mini-grid (schema, ветка A) ============
try:
    print('\n=== [9] SMOKE: ResultStore mini-grid (6 конфигураций, 1 падает, resume) ===')
    import shutil
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
            experiment_id='wave1_smoke',
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
    print(f'первый прогон: записано {len(written)} строк, failed={n_failed}, '
          f'runner вызван {len(calls)} раз')
    assert len(written) == 6 and n_failed == 1 and len(store) == 6

    written2 = run_grid(store, configs, runner)
    print(f'resume: добавлено {len(written2)}, runner суммарно {len(calls)} вызовов')
    assert len(written2) == 0 and len(calls) == 6 and len(store) == 6

    written3 = run_grid(store, configs, healed, retry_failed=True)
    print(f'retry_failed: перезаписано {len(written3)}, status={written3[0].status}')
    assert len(written3) == 1 and written3[0].status == 'ok' and store.failed_keys() == []

    try:
        store.append(row_from_compute_all(configs[0], {'mmd': 9.9}))
        raise AssertionError('дубликат не обнаружен')
    except DuplicateKeyError as exc:
        print(f'duplicate key: пойман DuplicateKeyError: {str(exc)[:120]}')

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
        print(f'manifest mismatch: пойман ManifestMismatchError: {str(exc)[:120]}')
    store = ResultStore.rebuild_manifest(STORE_DIR)
    print(f'rebuild_manifest: строк={len(store)}')
    assert len(store) == 6

    frame = store.load_frame()
    print('load_frame:', frame.shape, 'колонок; статусы:',
          frame['status'].value_counts().to_dict())

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
    with open('/content/wave1_smoke_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print('smoke summary:', json.dumps(summary, ensure_ascii=False))
    results['smoke'] = 'OK'
except Exception:
    results['smoke'] = 'FAIL'
    traceback.print_exc(file=sys.stdout)

# ============ Секция 10: manifest окружения ============
try:
    print('\n=== [10] MANIFEST ===')
    import numpy, scipy, pandas
    manifest = {
        'date': '2026-10-05',
        'phase': 'wave1_validation (schema + nlp-audit)',
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'gpu': run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader']).stdout.strip(),
        'versions': {
            'numpy': numpy.__version__, 'scipy': scipy.__version__,
            'pandas': pandas.__version__,
        },
        'ripser_patch': 'CMakeLists -std=c++11 -> -std=c++17',
    }
    for pkg in ('tensorflow', 'torch', 'pyarrow'):
        r = run([sys.executable, '-c', f'import {pkg}; print({pkg}.__version__)'])
        if r.returncode == 0:
            manifest['versions'][pkg] = r.stdout.strip()
    with open('/content/wave1_env_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    results['manifest'] = 'OK'
except Exception:
    results['manifest'] = 'FAIL'
    traceback.print_exc(file=sys.stdout)

print('\n=== ИТОГ ВАЛИДАЦИИ WAVE 1 ===')
for key, value in results.items():
    print(f'  {key}: {value}')
sys.stdout.flush()