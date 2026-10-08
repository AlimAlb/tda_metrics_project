"""Датасет Wave ST: свежие QA из HotpotQA, канонические записи и исключение HaluEval.

Источник — HotpotQA (distractor, train split): HaluEval построен из него же,
поэтому домен и формат knowledge совпадают с уже использованными в L3R/L9M,
но промпты свежие — пересечения с HaluEval исключаются по нормализованному
тексту вопроса (sha256). Knowledge-пассаж собирается из supporting facts
в порядке их перечисления (как в HaluEval). Сэмплирование детерминировано
``rng(seed)``; записи сериализуются в JSON с контрольной суммой.
"""
import hashlib
import json
import os

__all__ = [
    'ST_PROMPT_SOURCE',
    'normalize_question',
    'question_hash',
    'assemble_knowledge',
    'st_records_from_hotpot',
    'save_st_records',
    'load_st_records',
    'records_sha256',
]

ST_PROMPT_SOURCE = 'hotpotqa_distractor_train_v1'
_KNOWLEDGE_MIN_WORDS = 8
_KNOWLEDGE_MAX_WORDS = 120
_KNOWLEDGE_MAX_SENTENCES = 4


def normalize_question(text):
    """Нормализация вопроса для хэширования: lower, склейка пробелов, без пунктуации."""
    cleaned = ''.join(
        character.lower() if character.isalnum() else ' '
        for character in str(text))
    return ' '.join(cleaned.split())


def question_hash(text):
    """sha256 нормализованного вопроса — ключ исключения пересечений с HaluEval."""
    return hashlib.sha256(normalize_question(text).encode('utf-8')).hexdigest()


def assemble_knowledge(context, supporting_facts):
    """Knowledge-пассаж из supporting facts HotpotQA-записи.

    ``context`` — {'title': [...], 'sentences': [[...], ...]}; ``supporting_facts``
    — {'title': [...], 'sent_id': [...]}. Предложения supporting facts берутся
    в порядке перечисления и склеиваются пробелом; повторяющиеся (title,
    sent_id) не дублируются. Возвращает (knowledge_text, n_sentences).
    """
    sentences = []
    seen = set()
    for title, sent_id in zip(supporting_facts['title'], supporting_facts['sent_id']):
        key = (title, int(sent_id))
        if key in seen:
            continue
        seen.add(key)
        try:
            position = context['title'].index(title)
        except ValueError:
            continue
        sentence = context['sentences'][position][int(sent_id)]
        sentences.append(str(sentence).strip())
    return ' '.join(sentences), len(sentences)


def st_records_from_hotpot(hotpot_rows, n_prompts, seed, exclude_hashes):
    """Детерминированная выборка свежих ST-промптов из строк HotpotQA.

    Фильтры (зафиксированы протоколом): непустые вопрос и knowledge, 1..4
    предложения supporting facts, длина knowledge 8..120 слов, вопрос не
    пересекается с ``exclude_hashes`` (sha256 нормализованных вопросов
    HaluEval). Выборка ``rng(seed).choice`` без возвращения из прошедших
    фильтры строк в исходном порядке данных; записи нумеруются
    ``st_{i:04d}`` в порядке выборки. Возвращает список dict:
    ``prompt_id, question, knowledge, gold_answer, question_hash,
    n_knowledge_sentences, hotpot_type``.
    """
    import numpy as np

    excluded = set(exclude_hashes)
    candidates = []
    for row in hotpot_rows:
        question = str(row.get('question') or '').strip()
        knowledge, n_sentences = assemble_knowledge(row['context'], row['supporting_facts'])
        if not question or not knowledge:
            continue
        if not 1 <= n_sentences <= _KNOWLEDGE_MAX_SENTENCES:
            continue
        words = len(knowledge.split())
        if not _KNOWLEDGE_MIN_WORDS <= words <= _KNOWLEDGE_MAX_WORDS:
            continue
        digest = question_hash(question)
        if digest in excluded:
            continue
        candidates.append({
            'question': question,
            'knowledge': knowledge,
            'gold_answer': str(row.get('answer') or '').strip(),
            'question_hash': digest,
            'n_knowledge_sentences': n_sentences,
            'hotpot_type': str(row.get('type') or ''),
        })
    if len(candidates) < n_prompts:
        raise ValueError(
            f'после фильтров прошло {len(candidates)} строк — меньше n_prompts={n_prompts}')
    order = np.random.default_rng(seed).choice(
        len(candidates), size=n_prompts, replace=False)
    records = []
    for index in order:
        record = dict(candidates[index])
        record['prompt_id'] = f'st_{len(records):04d}'
        records.append(record)
    return records


def records_sha256(records):
    """Контрольная сумма списка записей (стабильный порядок ключей)."""
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def save_st_records(records, path):
    """Атомарная запись записей + их контрольной суммы (suffix .sha256)."""
    tmp = f'{path}.tmp'
    with open(tmp, 'w', encoding='utf-8') as handle:
        json.dump(records, handle, ensure_ascii=False, indent=1)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    with open(f'{path}.sha256', 'w', encoding='utf-8') as handle:
        handle.write(records_sha256(records))
    return records_sha256(records)


def load_st_records(path, expected_sha256=None):
    """Чтение записей с проверкой контрольной суммы (если передана)."""
    with open(path, encoding='utf-8') as handle:
        records = json.load(handle)
    if expected_sha256 is not None and records_sha256(records) != expected_sha256:
        raise ValueError(f'контрольная сумма {path} не совпала')
    return records
