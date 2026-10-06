# Wave S5 — чистая синтетика направленной селективности

**Статус:** к реализации; независима от LLM-waves.  
**Тип:** подтверждающая синтетическая серия, уточняющая S3/S4.  
**Связь с мастер-планом:** §5.3–5.6 и ограничение 12.

## Проблема

В S3 drop меняет среднее Q, а `invent_fraction` не является буквально долей
внешних точек: оно задаёт число внешних мод, а равновесные веса меняют density
всех мод. Направленный signal остаётся интересным, но из него нельзя выделить
чисто coverage/topology effect.

## Серии

### S5a — symmetric mode drop

Удалять противоположные пары мод из восьмимодовой смеси. Центр остаётся нулевым;
веса оставшихся мод равны. Сетка dropped mass: `{0,.125,.25,.5,.75}`.

### S5b — fixed-mass external invention

Сохранять базовые mode weights, а ровно `alpha` точек Q заменять внешними
симметричными модами. `alpha` — фактическая доля заменённых наблюдений,
записанная в config и проверенная в каждой строке. Сетка `{0,.01,.05,.125,.25,.5}`.

### S5c — moment-matched control

Сконструировать P/Q с близкими средним и covariance, но разным mode coverage.
До запуска задать numerical tolerances для difference mean/cov и discard/retry
правило. Этот режим проверяет, не объясняется ли эффект первым/вторым моментом.

## Общий protocol

- отдельный `protocols/s5_protocol.json` перед запуском;
- `n={100,250,500,1000}`, минимум 30 independent repeats на configuration;
- P/Q independent except when explicitly testing paired construction;
- primary: directional effect versus own same-law control; secondary: MMD,
  Fréchet, energy distance и при доступности Sinkhorn/Wasserstein;
- report mean, SD, bootstrap CI, empirical FPR/power и mean/cov diagnostics;
- direction, metric family и multiple-comparison policy фиксируются до запуска.

## Артефакты и tests

Добавить sampler module, grid runner, schema configs and unit tests:

- zero-mean symmetry for S5a/S5b;
- exact replacement count S5b;
- moment tolerance S5c;
- deterministic seeds and no duplicate config key;
- calculation of all diagnostics and save/resume behavior.

Создать `results/raw/s5/`, figures `s5_*.png`, tables `s5_*.csv` и manifest
с code SHA, seeds, sampler parameters and failures.

## Acceptance criteria

- удалить/добавить modes можно без material mean shift;
- alpha в invention equals observed external-point share exactly;
- primary direction reproduces across independent repeats with a reported CI,
  либо отрицательный результат сохраняется без post-hoc изменения дизайна;
- Fréchet/MMD comparisons не выдаются за matched без опубликованных moment
  diagnostics;
- S3/S4 артефакты остаются неизменными, S5 описывается как уточняющая серия.
