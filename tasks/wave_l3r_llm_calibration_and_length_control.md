# Wave L3R — LLM-native calibration и настоящий length-matched контроль

**Статус:** к реализации.  
**Тип:** исправляющая валидация существующей LLM-серии; новая серия, не
перезапись `results/raw/llm` или `results/raw/wave5`.  
**Связь с мастер-планом:** §3.3, §7.2–7.3 и ограничения 2, 3, 14, 15.

## Проблема

Текущий `length_matched_pair` детерминирован: аргумент `seed` не участвует в
выборке, а пять L3d строк описывают одну и ту же выборку (`sd=0`). Кроме того,
absolute thresholds primary-метрик перенесены из 2D-синтетики, хотя PR зависит
от `k`, размерности и геометрии пространства.

## Цели

1. Получить независимые стратифицированные length-matched повторения L3d.
2. Откалибровать null, FPR и power в той же LLM-конфигурации, не в синтетике.
3. Зафиксировать protocol до val/test и корректно отделить exploratory от
confirmatory результатов.

## Scope

### A. Новый API и тесты

Добавить новый режим, не меняя семантику старого API:

- `sample_length_matched_pair(..., seed, n_cloud, bins=10)`;
- внутри каждого bin случайно выбирать равную квоту без возвращения;
- не брать «первые строки» после сортировки;
- возвращать matrices, selected prompt IDs, границы bins и диагностику длин.

Тесты обязаны проверять: детерминизм при одном seed, отличие выборок при разных
seed, отсутствие пересечения внутри стороны, равные bin-counts, допустимый
KS/mean/median balance и сохранение группового split по prompt_id.

### B. Locked protocol

До любого val/test создать `protocols/l3r_protocol.json` и закоммитить его
отдельным коммитом. В него включить:

- commit SHA, cache checksum, dataset/model/tokenizer revisions;
- layer=18, pooling, PCA-rule, primary metrics, directions;
- sizes `n={50,100,250,500,1000}`, repeats/null count, alpha grid;
- threshold rule, primary endpoint и exact split hashes;
- дату и утверждённый список допустимых analyses.

Если внешняя пре-регистрация отсутствует, все отчётные формулировки используют
«locked internal protocol», не «pre-registration».

### C. LLM-native null/power

На dev-correct:

- для каждого `n` построить не менее 500 independent same-law
  correct-vs-correct пар;
- выполнить не менее 500 label permutations, стратифицированных по prompt;
- для каждого n/metric зафиксировать threshold с nominal FPR=0.05;
- оценить observed FPR с binomial CI.

После lock на val/test оценить power на заданной alpha-сетке, **включая alpha=0**.
Для L3d провести минимум 30 независимых matched ресэмплов; CI строить
cluster-bootstrap по prompt, не трактуя пересекающиеся подвыборки как внешние
репликации.

## Артефакты

- `results/raw/l3r/{manifest.json,results.parquet,summary.json,protocol_ref.json}`;
- `results/tables/l3r_length_balance.csv`, `l3r_null_power.csv`;
- figures с length balance и power/FPR curves;
- новый CPU unit-test и heavy-env integration test.

В manifest записать code SHA, cache hash, список input prompt hashes, версию
CUDA/всех heavy dependencies и число failed rows.

## Acceptance criteria

- старые L3/L4/Wave5 артефакты не меняются;
- L3d больше не имеет искусственного нулевого SD;
- выбранный на dev threshold показывает FPR с 95% CI, совместимым с 0.05,
  на val и test alpha=0;
- новые результаты не называются доказательством truth-effect: HaluEval
  source/style confound остаётся до Wave ST;
- `pytest` для нового API проходит, а runner resume-safe и записывает failures.

## Out of scope

Никакого выбора `k`, слоя, PCA dimension или endpoint по val/test. Не править
настоящий отчёт и не ретроактивно переименовывать старый D-010 как verified
pre-registration.
