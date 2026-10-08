# tda_metrics

Курсовой проект: сравнение топологических (MTD, RTD, NTD) и статистических
(improved precision/recall, MMD, Fréchet, JS) метрик схожести облаков точек
P («данные») и Q («модель») — на управляемой 2D-синтетике, эмбеддингах CV
(CLIP/DINO/VAE, MNIST/CIFAR) и скрытых состояниях LLM (Qwen2.5-3B, HaluEval).

## Структура

```
├── src/tda_metrics/                 # пакет: сэмплеры, метрики, схемы результатов,
│                                    #   LLM-эксперименты, баркоды, CV-сетки
├── notebooks/
│   ├── current_report_colab.ipynb  # точка входа воспроизведения (Colab, T4)
│   └── topology_metrics_final.ipynb# финальный отчёт-ноутбук (артефакт: results/)
├── scripts/                         # раннеры серий (см. ниже)
├── protocols/                       # locked internal protocols (коммитятся до прогонов)
├── tests/                           # pytest-инварианты (запускать на Colab-VM)
├── results/{raw,tables,figures}     # артефакты серий: parquet/manifest, CSV, PNG
└── report/                          # текст курсовой
```

## Запуск тестов (на Colab-VM)

```
pip install pytest && pytest tests/ -q
```

## Воспроизведение актуального отчёта

Ноутбук `notebooks/current_report_colab.ipynb` — точка входа в Colab
(Runtime → T4 GPU): `smoke`-профиль проверяет среду (тесты, power-smoke,
S5-smoke, PRV-иллюстрация на CPU), `full`-профиль воспроизводит серии:
Monte Carlo power-анализ, чистую синтетику S5, извлечение hidden states
Qwen с чекпоинтами, LLM-контроли и калибровку порогов (разделы 5.2–5.3) и
послойную серию 5.4 с нуля. Все прогоны пишут manifest и resume-безопасны.

Ключевые раннеры:

- `scripts/run_power_analysis.py` — Monte Carlo power на синтетических смесях;
- `scripts/run_s5_grid.py` — направленная селективность (S5a/b/c, 1680 строк);
- `scripts/run_precision_recall_visualization.py` — 2D-иллюстрация механики
  improved precision/recall@k (CPU, без тяжёлого стека);
- `scripts/extract_llm_embeddings.py` — hidden states Qwen2.5-3B с
  answer-masking и чекпоинтами;
- `scripts/run_l3r_chain.py` — LLM-native калибровка порогов (null/val/test),
  length-matched и permutation-контроли, смеси по доле галлюцинаций;
- `scripts/run_llm_layerwise_chain.py` — послойные L3/L3d/L3e + RTD/CKA
  (серия 5.4 с нуля; перенос исторических сидов, точное воспроизведение);
- `scripts/run_l3v_artifacts.py` — PCA-облака, баркоды и послойные фигуры
  из кэша и committed raw.

Пересобрать ноутбук после редактирования генератора:

```bash
python scripts/build_current_report_colab_notebook.py
```

Исторические raw-таблицы (`results/raw/llm`, `results/raw/wave5`) не
перезаписываются новыми прогонами: серия 5.4 с нуля пишется в
`results/raw/layerwise_rerun`, свежая LLM-калибровка — в
`results/raw/l3r`; собственный manifest есть у каждой серии.

## Заметки

- MTD/RTD требуют GPU-рантайм; improved PR и статистические метрики работают на CPU.
- Значения MTD при разных `batch_size` несравнимы — обертка всегда считает на полных облаках.
- RTD инвариантен к сдвигам облаков (строится только из внутренних попарных расстояний);
  RTD-пороги сравнимы только при равном n.
