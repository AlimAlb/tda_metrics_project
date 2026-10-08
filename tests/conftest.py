"""Локальный запуск лёгких тестов без heavy-стека (tensorflow, ripser++, pandas, scipy).

Импорт ``tda_metrics`` выполняет пакетный ``__init__``, который безусловно
тянет тяжёлые зависимости (metrics → tensorflow/mtd/rtd/scipy,
experiments/nlp_datasets → pandas). В окружении без них (локальный WSL:
только numpy и pytest) вместо падения на ``__init__`` подставывается скелет
пакета с корректным ``__path__``: подмодули (mixtures, l9m_pools, ...)
импортируются как обычно, пакетные атрибуты недоступны. При рабочем
full-стеке (Colab VM) штатный импорт проходит и подстановка не выполняется.
"""
import importlib
import sys
import types
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent.parent / 'src' / 'tda_metrics'

try:
    importlib.import_module('tda_metrics')
except Exception:
    _skeleton = types.ModuleType('tda_metrics')
    _skeleton.__path__ = [str(_PACKAGE_DIR)]
    sys.modules['tda_metrics'] = _skeleton
