# Wave L9M — length-matched packet dose response без confidence-признаков

**Статус:** к реализации.  
**Тип:** внутренняя исправляющая калибровка L9; новая серия, не перезапись
`results/raw/l9/`.  
**Зависимости:** исправленный `build_packet` с независимыми P/Q_correct и
протестированный стратифицированный sampler, реализованный в рамках L9M.
Wave L3R и Wave 5 не запускаются.

## Проблема

Исходный L9 строит P/Q из пулов без matching по длине. В HaluEval QA длина
correct и hallucinated ответов резко различается, поэтому средняя длина Q
почти линейно кодирует заданную вручную долю замены alpha. Кроме того,
alpha=0 в старом артефакте был точной копией P; этот дефект исправляется в
packet builder, но старые результаты `results/raw/l9/` не должны
переинтерпретироваться.

L9M проверяет более узкий вопрос: сохраняется ли dose response топологических
и coverage-метрик, когда P, Q_correct и hallucinated-пул сбалансированы по
длине, а confidence-признаки не подаются в модель.

## Locked design

До запуска создать и закоммитить `protocols/l9m_protocol.json` с code SHA,
cache hash, split hashes, seed-ами, слоями, PCA rule, первичными endpoints и
всеми нижеуказанными параметрами.

1. Использовать тот же dev split, слой 18, `mean_answer` и PCA-16, обученную
   только на dev-correct reference. Это *внутренняя калибровка*, не внешний
   deployment claim.
2. В каждом prompt-fold строить общие length bins по `n_pooled`; default 10.
   Из них выбирать без возвращения три набора: P-correct, Q_correct и
   hallucinated. P и Q_correct не пересекаются по item/prompt id.
3. Квоты по bin должны быть одинаковы для P, Q_correct и hallucinated.
   Если 10 bins не позволяют набрать заданный `n_cloud`, заранее зафиксировать
   правило уменьшения числа bins/`n_cloud` на dev и сохранить фактический
   размер; нельзя silently повторять элементы или ослаблять баланс на val/test.
4. Q_alpha получается заменой точек Q_correct hallucinated-точками. Вложенные
   alpha допустимы **только для dose-response curves**; они не считаются
   независимыми пакетами и не служат headline deployment validation.
5. Сетка по умолчанию: `alpha={0,.05,.1,.15,.25,.5,.75,1}`, пять
   prompt-folds и четыре repeat. Все alpha одного `(fold, repeat)` используют
   общую базу исключительно для наглядной дозовой кривой.

## Анализы

### Primary: калиброванные метрики без обучения

На dev собрать independent same-law P-vs-Q_correct пары в том же
length-matched протоколе. Для каждой topology/coverage/statistics метрики
зафиксировать направление, threshold при nominal FPR=0.05 и observed FPR с
binomial CI. На val/test сообщить power по alpha, стандартизированный эффект
относительно same-law и минимальную alpha с power >= 0.8.

Первичные группы признаков:

- `topology_only = {mtd_PQ, mtd_QP, ntd_PQ, ntd_QP, rtd}`;
- `coverage_only = {precision@1,@3,@10, recall@1,@3,@10}`;
- `statistics_only = {mmd, frechet, js}`.

`llm_confidence`, `q_len_mean`, NLL, entropy, `alpha` и `n_hall` запрещены
как входы primary analysis.

### Secondary: абляция простого предиктора

Повторить OOF GroupKFold logistic/ridge **только** для трёх групп выше. Это
иллюстрация калибровки смеси, не доказательство практической детекции.
Отчёт обязан включать single-feature OOF baseline для каждой метрики и
отдельные результаты `alpha<=.1`.

## Diagnostics и тесты

- В каждой строке сохранять bin-counts, mean/median/std длины P/Q и KS/SMD.
  Acceptance: одинаковые bin-counts, нулевое пересечение P/Q item ids и
  заранее установленный length-balance tolerance.
- Явно проверить, что q_len_mean не разделяет alpha=0 и alpha>0 лучше chance
  после matching; если это не выполнено, серия invalid и не получает
  метрик-результатов.
- Unit tests: детерминизм одного seed, иной состав при другом seed, отсутствие
  повторов, exact bin quotas, nested alpha replacement и отсутствие
  `P == Q` при alpha=0.
- Сохранить packet composition либо hashes всех item ids; runner resume-safe.

## Артефакты и acceptance

- `results/raw/l9m/{manifest.json,packet_table.parquet,results.parquet,summary.json,verdict.json}`;
- `results/tables/l9m_length_balance.csv`, `l9m_dose_response.csv`,
  `l9m_single_feature.csv`;
- новая figure: topology/statistics/coverage effect и power по alpha;
- старые `results/raw/l9/` и `report/coursework.md` read-only;
- headline допустим только как «length-matched dose response on HaluEval
  candidate-answer representations», без claims о detector-е собственных
  галлюцинаций Qwen.
