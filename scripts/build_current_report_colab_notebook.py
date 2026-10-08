"""Собрать Colab-ноутбук для воспроизведения актуальных серий отчёта.

Ноутбук намеренно не затрагивает удалённые пакетные L9-эксперименты. Он
даёт две безопасные точки входа: короткий smoke-прогон для проверки среды и
полный GPU-маршрут для power-анализа и LLM-серий 5.2--5.3. Для 5.4 в
текущем репозитории доступны зафиксированные raw-артефакты и пересборка
визуализаций; отдельный исходный runner послойного расчёта отсутствует, и
это явно зафиксировано в ноутбуке.

Скрипт не требует nbformat: это облегчает пересборку ноутбука на чистой
машине. Запускать из корня репозитория:

    python scripts/build_current_report_colab_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "current_report_colab.ipynb"


def markdown(source: str, cell_id: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "id": cell_id,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


CELLS = [
    markdown(
        """# Воспроизведение актуальных экспериментов курсовой работы в Google Colab

Этот ноутбук запускает только серии, которые остаются в отчёте: синтетическую
калибровку и power-анализ, а также LLM-контроли и смеси из разделов 5.2--5.3.
Удалённые пакетные серии L9 и confidence-бейзлайны здесь намеренно не используются.

Перед запуском в Colab выберите **Runtime → Change runtime type → T4 GPU**
или более мощный GPU. Полный LLM-прогон требует много времени и скачивает
Qwen2.5-3B-Instruct; smoke-профиль безопасен для проверки установки.

## Что воспроизводится

* `run_power_analysis.py` — Monte Carlo power-анализ на синтетических смесях;
* `run_s5_grid.py` — чистая синтетика направленной селективности (S5a/b/c);
* `run_precision_recall_visualization.py` — 2D-иллюстрация механики improved
precision/recall@k (CPU);
* `extract_llm_embeddings.py` — извлечение скрытых состояний Qwen с
  answer-masking и чекпоинтами;
* `run_l3r_chain.py` — same-law, length-matched и permutation-контроли,
  смеси по доле галлюцинаций и их анализ;
* `run_l3v_artifacts.py` — пересборка PCA-облаков, баркодов и послойных фигур
  из сохранённых артефактов.

В текущем дереве отсутствует отдельный runner, который с нуля считает raw
послойной серии 5.4. Поэтому последний пункт **перестраивает её фигуры из
зафиксированных `results/raw/llm` и `results/raw/wave5`**, а не выдаёт это за
новый независимый расчёт. Это ограничение кода, а не свойство Colab.
""",
        "title",
    ),
    code(
        """# Настройки запуска. Для точного повторения своего прогона закрепите REPO_REF на SHA коммита.
REPO_URL = "https://github.com/AlimAlb/tda_metrics_project.git"
REPO_REF = "main"
REPO_DIR = "/content/tda_metrics_project"

# "smoke" проверяет окружение на малой сетке. "full" запускает тяжёлые серии.
PROFILE = "smoke"

# В full-режиме очистить только воспроизводимые промежуточные артефакты в
# одноразовой VM Colab перед запуском серий. По умолчанию False: это исключает
# случайную потерю checkpoint/resume при повторном запуске ячеек.
RESET_WORKDIR = False

# Smoke-извлечение загружает Qwen2.5-3B даже для нескольких примеров. Оставьте
# False, если проверяете только компиляцию и синтетическую часть.
RUN_LLM_SMOKE = False

assert PROFILE in {"smoke", "full"}
""",
        "configuration",
    ),
    code(
        """import os
import shutil
import subprocess
import sys
from pathlib import Path

def run(cmd):
    print("+", " ".join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), check=True)

repo = Path(REPO_DIR)
if repo.exists():
    print(f"Удаляю прежнюю копию репозитория: {repo}")
    shutil.rmtree(repo)
run(["git", "clone", "--recursive", "--branch", REPO_REF, REPO_URL, REPO_DIR])
os.chdir(repo)
print("Рабочая директория:", Path.cwd())
run(["git", "rev-parse", "HEAD"])
""",
        "clone",
    ),
    markdown(
        """## Установка

Базовые зависимости проекта фиксируются в `pyproject.toml`. Топологические
метрики используют отдельные исследовательские реализации: ripser++, MTopDiv,
RTD и improved precision/recall. `ripser++` собирается с CUDA; патч C++17 нужен
современным образам Colab с CUDA 13.
""",
        "install-note",
    ),
    code(
        """from pathlib import Path

run([sys.executable, "-m", "pip", "install", "-q", "."])
run([sys.executable, "-m", "pip", "install", "-q", "pytest", "pyarrow", "transformers", "accelerate", "sentencepiece"])

def clone_once(url, directory, recursive=False):
    path = Path(directory)
    if path.exists():
        print(f"{directory}: уже есть")
        return
    command = ["git", "clone"]
    if recursive:
        command.append("--recursive")
    command.extend([url, directory])
    run(command)

clone_once("https://github.com/simonzhang00/ripser-plusplus.git", "third_party/ripser-plusplus", recursive=True)
cmake = Path("third_party/ripser-plusplus/CMakeLists.txt")
text = cmake.read_text(encoding="utf-8")
patched = text.replace("-std=c++11", "-std=c++17").replace("set(CMAKE_CXX_STANDARD 11)", "set(CMAKE_CXX_STANDARD 17)")
if patched != text:
    cmake.write_text(patched, encoding="utf-8")
run([sys.executable, "-m", "pip", "install", "-q", "./third_party/ripser-plusplus"])

clone_once("https://github.com/IlyaTrofimov/MTopDiv.git", "third_party/MTopDiv")
run([sys.executable, "-m", "pip", "install", "-q", "./third_party/MTopDiv"])
run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/IlyaTrofimov/RTD.git"])
clone_once("https://github.com/AlimAlb/improved-precision-and-recall-metric.git", "third_party/improved-precision-recall")
run([sys.executable, "-m", "pip", "install", "-q", "./third_party/improved-precision-recall"])
""",
        "install",
    ),
    code(
        """import json
import platform
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy
import torch

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA доступна:", torch.cuda.is_available())
if not torch.cuda.is_available():
    message = "GPU не найден: тяжёлые серии (MTD/RTD, LLM) требуют T4."
    if PROFILE == "full":
        raise RuntimeError(message + " Runtime → Change runtime type → T4 GPU.")
    print("ПРЕДУПРЕЖДЕНИЕ (smoke):", message, "Продолжаю — smoke не считает RTD/NTD.")
else:
    print("GPU:", torch.cuda.get_device_name(0))
run(["nvidia-smi"])

# Импорт TopologyMetrics проверяет, что CUDA-стек для MTD/RTD действительно собран.
from tda_metrics.metrics import TopologyMetrics

run_root = Path("results/colab_runs") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
run_root.mkdir(parents=True, exist_ok=True)
environment = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0),
    "profile": PROFILE,
}
(run_root / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")
run([sys.executable, "-m", "pip", "freeze"])
print("Метаданные запуска:", run_root / "environment.json")
""",
        "environment",
    ),
    markdown(
        """## Проверка кода и синтетический smoke-прогон

Smoke-прогон проверяет установку, фиксированные сэмплеры и запись manifest,
но намеренно не считает RTD и NTD: это короткая инженерная проверка, а не
замена полному анализу мощности.
""",
        "smoke-note",
    ),
    code(
        """run([
    sys.executable, "-m", "pytest", "-q",
    "tests/test_samplers.py",
    "tests/test_power.py",
    "tests/test_llm_embeddings.py",
    "tests/test_sample_length_matched_pair.py",
    "tests/test_s5_samplers.py",
    "tests/test_precision_recall_viz.py",
    "tests/test_barcodes.py",
])

smoke_power = run_root / "power_smoke"
run([
    sys.executable, "scripts/run_power_analysis.py",
    "--n-grid", "100",
    "--n-control", "2",
    "--n-effect", "2",
    "--effects", "drop0.25", "invent0.25",
    "--skip-heavy",
    "--out-dir", str(smoke_power),
])
print("Smoke-артефакты:", smoke_power)

# PRV: CPU-иллюстрация механики improved PR@k (детерминированная, ~1 мин).
run([sys.executable, "scripts/run_precision_recall_visualization.py"])
print("PRV-фигуры: results/figures/prv_precision_recall_semantics.png, "
      "prv_precision_recall_k_sweep.png")
""",
        "smoke",
    ),
    markdown(
        """## Полный power-анализ

Запускайте эту ячейку только при `PROFILE = "full"`. Результаты пишутся в
новый каталог с меткой времени, поэтому опубликованные таблицы не смешиваются
с новым расчётом. Повторный запуск с теми же параметрами можно продолжить:
`run_power_analysis.py` читает manifest и пропускает готовые блоки.
""",
        "power-note",
    ),
    code(
        """if PROFILE == "full":
    full_power = run_root / "power_full"
    run([
        sys.executable, "scripts/run_power_analysis.py",
        "--n-grid", "100", "250", "500", "1000",
        "--n-control", "20",
        "--n-effect", "20",
        "--effects", "drop0.1", "drop0.25", "invent0.15", "invent0.5",
        "--alpha-grid", "0.05",
        "--out-dir", str(full_power),
    ])
    print("Полный power-анализ завершён:", full_power)
else:
    print('Пропуск: установите PROFILE = "full", чтобы запустить полный power-анализ.')
""",
        "power-full",
    ),
    markdown(
        """## Синтетика S5: направленная селективность

`run_s5_grid.py` требует `protocols/s5_protocol.json` (есть в репозитории);
грид resume-безопасен по `results/raw/s5/rows.jsonl`. В клонах с
закокоммиченными результатами грид мгновенно пропускает готовые строки и
повторяет только analysis/figures поверх них — это сверка
воспроизводимости, а не новый расчёт; для честного пересчёта с нуля
включите `RESET_WORKDIR`. RTD считается только при n<=250 (решение
протокола по бюджету времени).
""",
        "s5-note",
    ),
    code(
        """
import os
import subprocess

env = dict(os.environ)
if PROFILE != "full":
    env["S5_SMOKE"] = "1"
    print("S5 smoke: усечённая сетка (n=100, 2 повтора, s5b alpha {0, 0.5})")
else:
    print("S5 full: 1680 строк (s5a/s5b/s5c x 4 n x 30 повторов); RTD только n<=250")
subprocess.run([sys.executable, "scripts/run_s5_grid.py"], env=env, check=True)
""",
        "synthetic-s5",
    ),
    markdown(
        """## LLM: извлечение hidden states и серии 5.2--5.3

`run_l3r_chain.py` сам запускает извлечение, если в `embeddings/l3r_cache`
нет готового кэша. Поэтому отдельная короткая ячейка ниже нужна только для
проверки извлечения. Полный маршрут использует те же настройки, что и
проектный протокол: Qwen2.5-3B-Instruct, `chat_knowledge_v1`, `mean_answer`,
слой 18, PCA-16 и фиксированный split по `prompt_id`.

Для действительно нового полного расчёта включите `RESET_WORKDIR`.
Очистка происходит только внутри временной VM Colab и удаляет лишь кэш и
результаты, которые полностью пересчитываются этим скриптом.
""",
        "llm-note",
    ),
    code(
        """if RUN_LLM_SMOKE:
    llm_smoke = run_root / "llm_smoke_cache"
    run([
        sys.executable, "scripts/extract_llm_embeddings.py",
        "--limit", "8",
        "--batch-size", "2",
        "--out-dir", str(llm_smoke),
    ])
    print("Smoke-кэш LLM:", llm_smoke)
else:
    print("LLM smoke отключён. Установите RUN_LLM_SMOKE = True, если хотите проверить извлечение.")
""",
        "llm-smoke",
    ),
    code(
        """if PROFILE == "full":
    if RESET_WORKDIR:
        for relative in ("embeddings/l3r_cache", "results/raw/l3r", "results/raw/l3v",
                         "results/raw/s5"):
            path = Path(relative)
            if path.exists():
                print("Очищаю воспроизводимый промежуточный каталог:", path)
                shutil.rmtree(path)
    run([sys.executable, "scripts/run_l3r_chain.py"])
    print("Серия 5.2--5.3 завершена; см. results/raw/l3r и results/tables/l3r_*.csv")
else:
    print('Пропуск: установите PROFILE = "full", чтобы запустить серию LLM.')
""",
        "llm-full",
    ),
    markdown(
        """## Пересборка фигур 5.2--5.4

Эта ячейка безопасно пересобирает визуализации из кэша и committed raw-файлов.
Она полезна для проверки происхождения рисунков и оформления отчёта. Послойные
raw-таблицы 5.4 (`results/raw/llm`, `results/raw/wave5`) в текущей версии
репозитория считаются входными артефактами: не удаляйте их, если хотите
пересобрать опубликованные рисунки.
""",
        "figures-note",
    ),
    code(
        """required_54 = [Path("results/raw/llm/results.parquet"), Path("results/raw/wave5/layerwise.parquet")]
missing_54 = [str(path) for path in required_54 if not path.exists()]
if missing_54:
    raise FileNotFoundError("Для пересборки фигур 5.4 не хватает: " + ", ".join(missing_54))
cache = Path("embeddings/l3r_cache/vectors.npy")
if cache.exists():
    run([sys.executable, "scripts/run_l3v_artifacts.py"])
    print("Готовы фигуры: results/figures/l3d_length_matched_pca2_layers.png, "
          "l4a_mixture_pca2_layers.png, l3_direct_pca2_layers.png, "
          "l3_barcodes_layer18.png, layerwise_llm_dynamics.png, layerwise_rtd_cka.png.")
else:
    print("Кэш embeddings/l3r_cache отсутствует: пересборка облаков/баркодов "
          "пропущена. Для полного запуска запустите LLM-серию (PROFILE = 'full') "
          "или положите кэш; послойные фигуры строятся из committed raw.")
    run([sys.executable, "scripts/run_l3v_artifacts.py"])
""",
        "figures",
    ),
    code(
        """# Финальная проверка: показываем ключевые manifest и статус результатов.
for path in [
    Path("results/raw/l3r/manifest.json"),
    Path("results/raw/l3r/summary.json"),
    Path("results/raw/l3v/manifest.json"),
]:
    if path.exists():
        print(f"\\n--- {path} ---")
        payload = json.loads(path.read_text(encoding="utf-8"))
        print(json.dumps(payload, ensure_ascii=False, indent=2)[:4000])
    else:
        print(f"Не создан: {path}")
""",
        "verify",
    ),
]


NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "colab": {"name": "current_report_colab.ipynb", "provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(NOTEBOOK, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
