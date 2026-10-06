# Реестр следующих work packet'ов

Это versioned-замена локальным заметкам `docs/`, которые в данном репозитории
игнорируются. Каждый packet независим, не перезаписывает готовые артефакты и
должен создавать новую серию в `results/raw/`.

| Порядок | ID | Цель | Зависимости |
|---:|---|---|---|
| 1 | [Wave PRV](wave_prv_precision_recall_visualization.md) | Объяснить механику precision/recall@k в нативном 2D | Нет |
| 2 | [Wave L3R](wave_l3r_llm_calibration_and_length_control.md) | Исправить L3d и получить LLM-native null/power | Существующий LLM cache |
| 3 | [Wave ST](wave_st_source_truth_factorial.md) | Отделить truth-effect от source/style-effect | Wave L3R protocol |
| 4 | [Wave L9R](wave_l9r_external_packet_validation.md) | Проверить пакетную задачу без вложенных пакетов и leakage | Wave ST data; L3R calibration |
| 5 | [Wave S5](wave_s5_clean_synthetic_directionality.md) | Изолировать mode drop/invention от moment/density-конфаундов | Нет |

## Общие правила оркестрации

- До дорогого запуска создать и закоммитить неизменяемый protocol-файл с
  конфигурацией, hypotheses, primary endpoint, порогами и кодовым SHA.
- Dev выбирает параметры; val подтверждает; test запускается только после
  фиксации решения на val. Нельзя называть этот процесс pre-registration без
  отдельного timestamped внешнего или более раннего Git-артефакта.
- Каждый раннер пишет отдельные `manifest.json`, `summary.json`, сырые строки,
  hashes входов/сплитов/кода и статусы `ok`/`failed`; ошибки одного блока не
  скрываются и не уничтожают результаты остальных.
- Existing artifacts read-only. Не редактировать `report/coursework.md` до
  review новых артефактов; текущие пользовательские изменения в нём чужие.
- Любой headline claim требует cluster/bootstrap CI и явно указанного
  статистического объекта: prompt, независимый пакет или внешний корпус.
