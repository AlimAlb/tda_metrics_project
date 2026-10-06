# Wave ST — факторный source × truth эксперимент для LLM

**Статус:** к реализации после фиксации L3R protocol.  
**Тип:** главный подтверждающий эксперимент LLM-части.  
**Связь с мастер-планом:** проверка вопроса §1.2(5–6), ограничения 1–3.

## Вопрос и гипотезы

Текущий HaluEval contrast смешивает truth, источник, длину и стиль. Wave ST
проверяет, существует ли effect корректности после контроля источника.

- **H0:** метрики различают только source/style; truth-effect после контроля
  источника отсутствует.
- **H1:** при фиксированном источнике `correct` и `wrong` различимы, а знак и
  величина эффекта переносятся на source-held-out генератор.

Primary endpoint: заранее заданная метрика из L3R, calibrated threshold и
power/FPR на source-held-out test. Secondary: dose-response и сравнение с
length/confidence baselines.

## Дизайн данных

Для каждого из 500–1000 новых QA prompt’ов с предоставленным knowledge собрать
минимум четыре проверенных ответа:

| Source | Truth | Содержимое |
|---|---|---|
| A | correct | Подтверждается knowledge |
| A | wrong | Локально ложная сущность/дата/число или unsupported claim |
| B | correct | То же условие для независимого генератора |
| B | wrong | То же условие для независимого генератора |

`wrong` должен быть сопоставим по длине и шаблону с `correct` того же source.
Не использовать пару gold HotpotQA против ChatGPT как primary data. Создание
ответов фиксируется: model revision, system prompt, temperature, seed и шаблон.
Каждый ответ получает source, factuality label, error type, length и verifier
status. Минимум 10% каждой cell вручную проверяются двумя аннотаторами; хранить
adjudication и agreement. Неоднозначные/неверифицируемые случаи исключать по
заранее указанному правилу до анализа hidden states.

## Splits и анализ

- group split строго по `prompt_id`; все четыре ответа prompt остаются вместе;
- dev выбирает PCA/threshold только по L3R protocol; val фиксирует решение;
- test включает within-source и cross-source analyses: A→B и B→A;
- source не должен быть feature модели;
- baseline models: answer length, confidence и lexical/style baseline;
- CI: cluster bootstrap по prompt; multiple endpoints объявить заранее и
скорректировать или чётко разделить primary/secondary.

Обязательные негативные контроли: correct против length-matched correct
paraphrase; source A correct против source B correct. Они показывают остаточный
style-effect отдельно от truth-effect.

## Оркестрация и артефакты

До generation/extraction отдельно закоммитить
`protocols/st_protocol.json`. Добавить adapter/loader, extraction runner и
analysis runner; не изменять HaluEval cache. Новые пути:

- `data/st/` — только если лицензия позволяет; иначе content hashes,
  deterministic generation recipe и private-data manifest;
- `results/raw/st/`; `results/tables/st_*.csv`; `results/figures/st_*.png`.

Manifest: code SHA, prompts/source hashes, generators and revisions, annotation
schema and agreement, splits, exclusions, environment and all failures.

## Acceptance criteria

- source and truth факторы реально ортогональны по дизайну, а не только по
  post-hoc regression;
- length balance опубликован для каждой source×truth cell;
- primary truth-effect тестируется на source-held-out prompts и generator;
- заявлять «factuality/hallucination signal» разрешено только если primary
  effect сохраняет direction на обоих A→B и B→A, имеет locked FPR и CI,
  исключающий null;
- если effect исчезает, это честный отрицательный результат: отчёт должен
  ограничить предыдущий claim source-conditioned различением.

## Out of scope

Не смешивать ST с пакетной регрессией L9 и не оптимизировать множество метрик
по test. Нет утверждений об индивидуальном детекторе, если endpoint пакетный.
