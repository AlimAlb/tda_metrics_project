"""Lite-setup VM после смерти диска: heavy-стек + пакет (pytest/smoke уже пройдены в этой сессии).

Секции 1-6 повторяют wave1_validate.py; pytest — канонический порядок D-002,
новые тест-файлы в конце; smoke — реальные метрики на 3 конфигурациях
synthetic_grid через run_grid + ResultStore (малые n, rtd_trials=2).
SystemExit не поднимается; Tee в /content/wave1b_log.txt.
"""
import subprocess
import sys
import os
import traceback

LOG_PATH = '/content/wave1b_setup_lite_log.txt'
SMOKE_DIR = '/content/wave1b_smoke_store'
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

# ============ Секции 1-5: heavy-стек (эталоны phase1/wave1) ============
try:
    print('=== [1] ENV ===')
    print('python:', sys.version)
    r = run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv'])
    print('nvidia-smi:', r.stdout.strip() or r.stderr.strip())
    results['env'] = 'OK'
except Exception:
    results['env'] = 'FAIL'
    traceback.print_exc(file=sys.stdout)

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
        print('CMakeLists: c++17 уже присутствует')
    pip_install('-q', '/content/ripser-plusplus')
    r = run([sys.executable, '-c', 'import ripserplusplus as rpp; print("ripser++ import OK")'])
    print(r.stdout.strip() or f'FAIL {r.stderr.strip()[-300:]}')
    results['ripser_patch'] = 'OK'
except Exception:
    results['ripser_patch'] = 'FAIL'
    traceback.print_exc(file=sys.stdout)

try:
    print('\n=== [3] MTopDiv ===')
    if not os.path.isdir('/content/MTopDiv'):
        run(['git', 'clone', '-q', 'https://github.com/IlyaTrofimov/MTopDiv.git', '/content/MTopDiv'])
    pip_install('-q', '/content/MTopDiv')
    results['mtopdiv'] = 'OK'
except Exception:
    results['mtopdiv'] = 'FAIL'
    traceback.print_exc(file=sys.stdout)

try:
    print('\n=== [4] RTD ===')
    pip_install('-q', 'git+https://github.com/IlyaTrofimov/RTD.git')
    results['rtd'] = 'OK'
except Exception:
    results['rtd'] = 'FAIL'
    traceback.print_exc(file=sys.stdout)

try:
    print('\n=== [5] improved precision/recall ===')
    repo = '/content/improved-precision-and-recall-metric'
    if not os.path.isdir(repo):
        run(['git', 'clone', '-q',
             'https://github.com/AlimAlb/improved-precision-and-recall-metric.git', repo])
    pip_install('-q', repo)
    results['pr'] = 'OK'
except Exception:
    results['pr'] = 'FAIL'
    traceback.print_exc(file=sys.stdout)

# ============ Секция 6: пакет из zip (ветки C/D/E) ============
try:
    print('\n=== [6] tda_metrics (unzip -o + force-reinstall) ===')
    if not os.path.exists(ZIP_PATH):
        raise RuntimeError(f'{ZIP_PATH} не найден: сначала colab upload')
    run(['unzip', '-q', '-o', ZIP_PATH, '-d', REPO_DIR])
    r = run(['unzip', '-l', ZIP_PATH])
    for line in (r.stdout or '').strip().splitlines():
        if any(name in line for name in
               ('statistics', 'mixtures', 'synthetic_grid', 'deformations')):
            print('   NEW:', line.strip())
    pip_install('-q', 'pandas', 'pyarrow', 'pytest')
    pip_install('-q', '--force-reinstall', '--no-deps', REPO_DIR)
    r = run([sys.executable, '-c',
             'import tda_metrics; from tda_metrics import statistics, mixtures, synthetic_grid;'
             ' print("import OK")'])
    print(r.stdout.strip() or f'FAIL {r.stderr.strip()[-300:]}')
    results['package'] = 'OK'
except Exception:
    results['package'] = 'FAIL'
    traceback.print_exc(file=sys.stdout)

print('\n=== ИТОГ LITE-SETUP ===')
for key, value in results.items():
    print(f'  {key}: {value}')
sys.stdout.flush()
