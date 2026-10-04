"""Адаптер парного датасета галлюцинаций HaluEval QA к единому контракту.

Фактическая schema оригинала (RUCAIBox/HaluEval, файл data/qa_data.json —
JSONL: одна JSON-запись на строку, несмотря на расширение): поля
``knowledge``, ``question``, ``right_answer``, ``hallucinated_answer``;
10 000 записей; колонки категории нет. Единый контракт строк результата:
``prompt_id, prompt, correct_answer, hallucinated_answer, category, source``.
Поле ``knowledge`` (контекст ответа) сохраняется дополнительной колонкой —
фаза L1 нуждается в нём при построении входа модели, и отбрасывать его
в адаптере было бы потерей данных. ``prompt`` = сырой текст вопроса,
``category`` = константа ``'qa'``, ``source`` = строка-идентификатор
источника с пиннанным commit.
"""
import hashlib
import json
import os
import urllib.error
import urllib.request

import pandas as pd

__all__ = [
    'CONTRACT_COLUMNS',
    'HALUEVAL_QA_URL',
    'HALUEVAL_QA_COMMIT',
    'HALUEVAL_QA_SHA256',
    'HALUEVAL_QA_SOURCE',
    'load_halueval_qa',
    'load_halueval_qa_canonical',
    'make_prompt_id',
    'to_canonical',
    'audit_pairs',
]

HALUEVAL_QA_COMMIT = 'b7253db3cdaa0ab2c382f92b26b390109174f77e'
HALUEVAL_QA_URL = ('https://raw.githubusercontent.com/RUCAIBox/HaluEval/'
                   f'{HALUEVAL_QA_COMMIT}/data/qa_data.json')
HALUEVAL_QA_SOURCE = f'RUCAIBox/HaluEval@{HALUEVAL_QA_COMMIT} qa'
HALUEVAL_QA_SHA256 = '89ed139ec5e3a3169a0b30e45569ac1283846f76f27f7bb5e908ee6deed57e88'

CONTRACT_COLUMNS = [
    'prompt_id',
    'prompt',
    'correct_answer',
    'hallucinated_answer',
    'category',
    'source',
]

PROMPT_ID_DELIMITER = '\x1f'

_RAW_COLUMNS = ['knowledge', 'question', 'right_answer', 'hallucinated_answer']


def _text(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return None
    return str(value)


def _is_empty(value):
    return value is None or value.strip() == ''


def _parse_records(text):
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return parsed
    records = []
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f'строка {number}: {exc}') from exc
    return records


def _download_bytes(url):
    for attempt in (1, 2):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == 2:
                raise
            print(f'загрузка {url} не удалась ({exc!r}), выполняю единственный retry')


def _write_cache(path, data):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = f'{path}.tmp.{os.getpid()}'
    with open(tmp, 'wb') as f:
        f.write(data)
    os.replace(tmp, path)


def load_halueval_qa(path=None, url=HALUEVAL_QA_URL, expected_sha256=None):
    """Сырой DataFrame HaluEval QA с фактическими колонками источника.

    Если ``path`` существует — читается локальный файл; иначе датасет
    скачивается по ``url`` (пиннанный commit оригинального репозитория;
    timeout 60 с и один retry при сетевой ошибке — зависшее скачивание
    исключено) и, при заданном ``path``, атомарно сохраняется туда целиком
    (tmp + os.replace: частичный файл не может сойти за кэш). ``expected_sha256``
    (эталон — HALUEVAL_QA_SHA256) проверяется и для скачанных, и для локальных
    данных; несовпадение -> ValueError. Формат — JSONL (одна JSON-запись на
    строку) с поддержкой JSON-массива как fallback; ошибка битой строки
    содержит её номер.
    """
    from_local = path is not None and os.path.isfile(path)
    if from_local:
        with open(path, 'rb') as f:
            data = f.read()
    else:
        data = _download_bytes(url)
    if expected_sha256 is not None:
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected_sha256.lower():
            raise ValueError(
                f'sha256 данных {actual} не совпадает с ожидаемым {expected_sha256}'
            )
    if not from_local and path is not None:
        _write_cache(path, data)
    return pd.DataFrame(_parse_records(data.decode('utf-8')))


def make_prompt_id(knowledge, question):
    """Детерминированный идентификатор prompt-объекта.

    sha1 от пары (knowledge, question) после strip: одинаковые prompt-объекты
    получают одинаковый идентификатор между строками и запусками; редкий
    разделитель исключает совпадения склеек полей.
    """
    key_knowledge = '' if knowledge is None else str(knowledge).strip()
    key_question = '' if question is None else str(question).strip()
    payload = key_knowledge + PROMPT_ID_DELIMITER + key_question
    return hashlib.sha1(payload.encode('utf-8')).hexdigest()


def to_canonical(raw, source=HALUEVAL_QA_SOURCE, drop_empty=True,
                 drop_duplicate_prompts=False):
    """Сырой HaluEval QA -> DataFrame единого контракта.

    ``prompt`` — сырой текст вопроса (как в источнике, без нормализации
    пробелов: форматирование входа модели решает фаза L1); ``knowledge``
    сохраняется дополнительной колонкой; ``category`` = 'qa' (колонки
    категории в источнике нет); ``prompt_id`` = make_prompt_id(knowledge,
    question). ``drop_empty`` отбрасывает строки с пустыми (после strip)
    вопросом или любым из ответов; ``drop_duplicate_prompts`` оставляет
    только первое вхождение каждого prompt_id.
    """
    frame = raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
    missing = [c for c in _RAW_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            f'ожидались колонки {_RAW_COLUMNS}, отсутствуют: {missing}; '
            f'фактические колонки: {list(frame.columns)}'
        )
    knowledge = [_text(v) for v in frame['knowledge'].tolist()]
    prompts = [_text(v) for v in frame['question'].tolist()]
    correct = [_text(v) for v in frame['right_answer'].tolist()]
    hallucinated = [_text(v) for v in frame['hallucinated_answer'].tolist()]
    canon = pd.DataFrame({
        'prompt_id': [make_prompt_id(k, p) for k, p in zip(knowledge, prompts)],
        'prompt': prompts,
        'correct_answer': correct,
        'hallucinated_answer': hallucinated,
        'category': 'qa',
        'source': source,
        'knowledge': knowledge,
    })
    if drop_empty:
        keep = [not (_is_empty(p) or _is_empty(c) or _is_empty(h))
                for p, c, h in zip(prompts, correct, hallucinated)]
        canon = canon.loc[keep].reset_index(drop=True)
    if drop_duplicate_prompts:
        canon = canon.drop_duplicates(subset='prompt_id', keep='first')
        canon = canon.reset_index(drop=True)
    return canon


def load_halueval_qa_canonical(path=None, source=HALUEVAL_QA_SOURCE,
                               expected_sha256=None, **kwargs):
    """Удобная цепочка: загрузка HaluEval QA -> единый контракт."""
    return to_canonical(
        load_halueval_qa(path=path, expected_sha256=expected_sha256),
        source=source,
        **kwargs,
    )


def _length_stats(values):
    series = pd.Series(values)
    return {
        'count': int(series.count()),
        'mean': float(series.mean()),
        'std': float(series.std()),
        'min': float(series.min()),
        'q25': float(series.quantile(0.25)),
        'median': float(series.median()),
        'q75': float(series.quantile(0.75)),
        'max': float(series.max()),
    }


def audit_pairs(canon):
    """Аудит пар канонического DataFrame: пустые, дубликаты, длины.

    Возвращает словарь, сериализуемый в JSON: долю полных пар, пустые поля,
    совпадения текстов correct/hallucinated, дубликаты prompt_id и текстов
    ответов, распределения длин (символы и слова) с парной разницей
    hallucinated - correct. Разница длин — кандидат в конфаунды
    (контроль L3d), поэтому считаются оба масштаба.
    """
    n_rows = len(canon)
    prompts = canon['prompt'].tolist()
    correct = canon['correct_answer'].tolist()
    hallucinated = canon['hallucinated_answer'].tolist()

    def chars(x):
        return 0 if x is None else len(x)

    def words(x):
        return 0 if x is None else len(x.split())

    correct_chars = [chars(x) for x in correct]
    hallucinated_chars = [chars(x) for x in hallucinated]
    diff_chars = [h - c for h, c in zip(hallucinated_chars, correct_chars)]
    diff_words = [words(h) - words(c)
                  for h, c in zip(hallucinated, correct)]

    def duplicate_keys(values):
        counts = {}
        for value in values:
            key = '' if value is None else value.strip()
            counts[key] = counts.get(key, 0) + 1
        return sum(1 for v in counts.values() if v > 1)

    return {
        'n_rows': n_rows,
        'n_empty_prompt': sum(1 for x in prompts if _is_empty(x)),
        'n_empty_correct': sum(1 for x in correct if _is_empty(x)),
        'n_empty_hallucinated': sum(1 for x in hallucinated if _is_empty(x)),
        'n_with_both_answers': sum(
            1 for c, h in zip(correct, hallucinated)
            if not _is_empty(c) and not _is_empty(h)
        ),
        'n_unique_prompt_ids': int(canon['prompt_id'].nunique()),
        'n_duplicate_prompt_ids': n_rows - int(canon['prompt_id'].nunique()),
        'n_correct_equals_hallucinated': sum(
            1 for c, h in zip(correct, hallucinated)
            if not _is_empty(c) and c.strip() == (h or '').strip()
        ),
        'n_duplicate_correct_answers': duplicate_keys(correct),
        'n_duplicate_hallucinated_answers': duplicate_keys(hallucinated),
        'lengths': {
            'correct_chars': _length_stats(correct_chars),
            'hallucinated_chars': _length_stats(hallucinated_chars),
            'correct_words': _length_stats([words(x) for x in correct]),
            'hallucinated_words': _length_stats([words(x) for x in hallucinated]),
            'diff_chars': _length_stats(diff_chars),
            'diff_words': _length_stats(diff_words),
            'n_hallucinated_longer_chars': sum(1 for d in diff_chars if d > 0),
            'n_hallucinated_shorter_chars': sum(1 for d in diff_chars if d < 0),
            'n_equal_chars': sum(1 for d in diff_chars if d == 0),
        },
    }
