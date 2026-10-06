# Wave L9R — независимая внешняя валидация пакетной задачи

**Статус:** к реализации после Wave ST.  
**Тип:** подтверждающая оценка deployment-like packet detection/regression.  
**Связь с мастер-планом:** §7.5 и ограничение 17.

## Проблема

Текущий L9 корректно показывает дозовую зависимость, но 160 пакетов не являются
160 независимыми единицами: alpha-варианты вложены, базовые выборки пересекаются,
PCA fit предшествует GroupKFold и оценка происходит на одном dev-корпусе.

## Дизайн

Разделить две цели:

- existing L9 оставить как `dose-response/exploratory`;
- L9R провести на source×truth data Wave ST как внешний packet benchmark.

Для каждого outer split:

1. PCA fit только на correct responses training prompt’ов этого split.
2. Train/val/test packets строить из непересекающихся prompt sets.
3. Каждый `(packet, alpha, repeat)` собирать fresh independent sampling;
   вложенные замены запрещены в оценочных пакетах.
4. Сетка: `n={25,50,100,250,500}`, `alpha={0,.01,.02,.05,.1,.25,.5,1}`;
   для регрессии дополнительно random alpha ~ Uniform(0,.5).
5. Выполнить leave-source-out, leave-error-type-out и, если возможно,
   leave-generator-out evaluation.

Classifier/regressor остаются простыми и фиксированными до test. Feature groups:
topology, coverage, statistics, confidence, length-only, lexical/style-only и
combined. Любая настройка выполняется только на train/val.

## Метрики и reporting

Task A: AUROC, AUPRC, balanced accuracy, FPR при фиксированной sensitivity,
calibration curve/Brier score. Task B: MAE, RMSE, R², calibration slope.
Все CI — block/cluster bootstrap по packet-generating prompt set; выводить
таблицу effective independent packet counts.

Отдельно сообщить quality при alpha<=.05 и n<=100. AUROC без operating threshold
не называть «надёжной детекцией».

## Артефакты и acceptance

- `protocols/l9r_protocol.json` коммитится до val/test;
- `results/raw/l9r/` включает composition каждого пакета или hashes IDs,
  split/PCA provenance, seed, alpha и nesting=false;
- tables/figures содержат uncertainty, FPR и baseline comparison;
- new tests доказывают отсутствие prompt overlap, PCA train-only fitting,
  independence packet compositions и корректность outer grouping;
- существующий L9 и его conclusion не перезаписываются;
- headline допускается только как external/source-held-out result; иначе
  формулировка остаётся exploratory.
