"""Валидация веток C/D/E синтетической фазы: setup -> полный pytest -> smoke compute_all.

Секции 1-6 повторяют wave1_validate.py; pytest — канонический порядок D-002,
новые тест-файлы в конце; smoke — реальные метрики на 3 конфигурациях
synthetic_grid через run_grid + ResultStore (малые n, rtd_trials=2).
SystemExit не поднимается; Tee в /content/wave1b_log.txt.
"""
import subprocess
import sys
import os
import traceback

LOG_PATH = '/content/wave1b_log.txt'
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

# ============ Секция 7: полный pytest (канонический порядок + новые) ============
try:
    print('\n=== [7] PYTEST (канонический порядок D-002 + новые ветки C/D/E) ===')
    cmd = [sys.executable, '-m', 'pytest',
           'tests/test_models.py', 'tests/test_metrics.py', 'tests/test_datasets.py',
           'tests/test_experiments.py', 'tests/test_reduction.py', 'tests/test_samplers.py',
           'tests/test_experiment_schema.py', 'tests/test_nlp_datasets.py',
           'tests/test_statistics.py', 'tests/test_mixtures.py',
           'tests/test_deformations.py', 'tests/test_synthetic_grid.py',
           '-q', '--tb=short']
    r = run(cmd, cwd=REPO_DIR, timeout=1800)
    tail = (r.stdout + r.stderr).strip().splitlines()
    for line in tail[-25:]:
        print(line)
    results['pytest'] = tail[-1] if tail else 'NO OUTPUT'
    with open('/content/wave1b_pytest_out.txt', 'w') as f:
        f.write(r.stdout + '\n=== STDERR ===\n' + r.stderr)
except Exception:
    results['pytest'] = 'FAIL'
    traceback.print_exc(file=sys.stdout)

# ============ Секция 8: smoke compute_all через synthetic_grid ============
try:
    print('\n=== [8] SMOKE: реальные метрики на 3 конфигурациях synthetic_grid ===')
    import shutil
    import numpy as np
    from tda_metrics import TopologyMetrics
    from tda_metrics.experiment_schema import ResultStore, run_grid
    from tda_metrics.synthetic_grid import build_clouds, build_s0_configs, build_s3_configs, make_runner

    device = 'cuda' if run([sys.executable, '-c', 'import torch; assert torch.cuda.is_available()']).returncode == 0 else 'cpu'
    print('device:', device)
    metrics = TopologyMetrics(seed=42, device=device)

    smoke_configs = [
        c for c in build_s0_configs()
        if c.experiment_id == 's0_ring_samelaw_n250_k0'
    ] + [
        c for c in build_s3_configs()
        if c.experiment_id in ('s3_d0.25_i0.0_k0', 's3_d0.0_i0.25_k0')
    ]
    print('smoke configs:', [c.experiment_id for c in smoke_configs])

    shutil.rmtree(SMOKE_DIR, ignore_errors=True)
    store = ResultStore(SMOKE_DIR)
    runner = make_runner(lambda p, q: metrics.compute_all(p, q, rtd_trials=2, rtd_batch=500))
    written = run_grid(store, smoke_configs, runner)
    for row in written:
        print(f'  {row.experiment_id}: status={row.status} runtime={row.runtime_seconds:.1f}s')
        if row.status == 'ok':
            values = {k: v for k, v in row.to_dict().items()
                      if k in ('mtd_PQ', 'mtd_QP', 'ntd_PQ', 'ntd_QP', 'rtd', 'mmd',
                               'precision@3', 'recall@3') and v is not None}
            print('   ', {k: round(v, 4) for k, v in values.items()})
            assert all(np.isfinite(v) for v in values.values())
    assert all(row.status == 'ok' for row in written) and len(written) == 3
    print('resume-повтор: добавлено', len(run_grid(store, smoke_configs, runner)))
    results['smoke'] = 'OK'
except Exception:
    results['smoke'] = 'FAIL'
    traceback.print_exc(file=sys.stdout)

print('\n=== ИТОГ ВАЛИДАЦИИ СИНТЕТИЧЕСКОЙ ФАЗЫ (код) ===')
for key, value in results.items():
    print(f'  {key}: {value}')
sys.stdout.flush()