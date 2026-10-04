"""Механические инварианты адаптера HaluEval QA на синтетическом мини-датасете.

Без реальной сети и heavy-стека: загрузка локального файла, контракт колонок,
отбрасывание пустых, дубликаты, детерминированность prompt_id, числа аудита.
Сетевые пути (timeout/retry, целостность sha256, атомарный кэш) закрыты
unittest.mock.
"""
import hashlib
import json
import urllib.error
from unittest import mock

import pandas as pd
import pytest

from tda_metrics.nlp_datasets import (
    CONTRACT_COLUMNS,
    HALUEVAL_QA_URL,
    audit_pairs,
    load_halueval_qa,
    load_halueval_qa_canonical,
    make_prompt_id,
    to_canonical,
)

RECORDS = [
    {'knowledge': 'Paris is the capital of France.',
     'question': 'What is the capital of France?',
     'right_answer': 'Paris',
     'hallucinated_answer': 'The capital of France is Lyon.'},
    {'knowledge': 'The Oberoi Group has its head office in Delhi.',
     'question': 'Where is the head office of The Oberoi Group?',
     'right_answer': 'Delhi',
     'hallucinated_answer': "The Oberoi family's hotel company is based in Mumbai."},
    {'knowledge': 'k3', 'question': 'q3',
     'right_answer': '', 'hallucinated_answer': 'Some hallucination.'},
    {'knowledge': 'k4', 'question': 'q4',
     'right_answer': 'London', 'hallucinated_answer': '   '},
    {'knowledge': 'Paris is the capital of France.',
     'question': 'What is the capital of France?',
     'right_answer': 'Paris',
     'hallucinated_answer': 'The capital of France is Lyon.'},
    {'knowledge': 'k6', 'question': 'q6',
     'right_answer': 'yes', 'hallucinated_answer': 'yes'},
    {'knowledge': None, 'question': 'q7',
     'right_answer': '42', 'hallucinated_answer': 'The answer is 43.'},
]

RAW_COLUMNS = ['knowledge', 'question', 'right_answer', 'hallucinated_answer']


@pytest.fixture(scope='module')
def raw_jsonl_path(tmp_path_factory):
    path = tmp_path_factory.mktemp('halueval') / 'qa_data.json'
    with open(path, 'w', encoding='utf-8') as f:
        for record in RECORDS:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    return path


@pytest.fixture(scope='module')
def raw(raw_jsonl_path):
    return load_halueval_qa(path=str(raw_jsonl_path))


@pytest.fixture(scope='module')
def canon(raw):
    return to_canonical(raw)


def test_load_local_jsonl(raw_jsonl_path):
    frame = load_halueval_qa(path=str(raw_jsonl_path))
    assert frame.shape == (7, 4)
    assert list(frame.columns) == RAW_COLUMNS


def test_load_local_json_array(tmp_path):
    path = tmp_path / 'qa_array.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(RECORDS, f, ensure_ascii=False)
    frame = load_halueval_qa(path=str(path))
    assert frame.shape == (7, 4)
    assert list(frame.columns) == RAW_COLUMNS


def test_canonical_contract_columns(canon):
    assert list(canon.columns)[:len(CONTRACT_COLUMNS)] == CONTRACT_COLUMNS
    assert 'knowledge' in canon.columns
    assert set(CONTRACT_COLUMNS).issubset(set(canon.columns))


def test_canonical_category_and_source(raw):
    frame = to_canonical(raw, source='test-source')
    assert (frame['category'] == 'qa').all()
    assert (frame['source'] == 'test-source').all()


def test_canonical_rejects_wrong_columns():
    with pytest.raises(ValueError):
        to_canonical(pd.DataFrame({'a': [1]}))


def test_empty_answers_rejected(canon):
    assert len(canon) == 5
    assert canon['correct_answer'].map(lambda x: x.strip() != '').all()
    assert canon['hallucinated_answer'].map(lambda x: x.strip() != '').all()


def test_empty_answers_kept_without_drop(raw):
    frame = to_canonical(raw, drop_empty=False)
    assert len(frame) == 7
    numbers = audit_pairs(frame)
    assert numbers['n_empty_correct'] == 1
    assert numbers['n_empty_hallucinated'] == 1
    assert numbers['n_with_both_answers'] == 5


def test_prompt_id_deterministic():
    first = make_prompt_id('knowledge text', 'question text')
    assert first == make_prompt_id('knowledge text', 'question text')
    assert first != make_prompt_id('knowledge text', 'other question')
    assert first != make_prompt_id('other knowledge', 'question text')


def test_prompt_id_stable_to_whitespace():
    assert make_prompt_id('  knowledge  ', ' question ') == \
        make_prompt_id('knowledge', 'question')


def test_duplicate_rows_share_prompt_id(canon):
    assert len(canon) == 5
    assert canon['prompt_id'].nunique() == 4
    assert canon.duplicated(subset='prompt_id').sum() == 1


def test_duplicate_prompts_dropped(raw):
    frame = to_canonical(raw, drop_duplicate_prompts=True)
    assert len(frame) == 4
    assert frame['prompt_id'].is_unique


def test_audit_pairs_numbers(canon):
    numbers = audit_pairs(canon)
    assert numbers['n_rows'] == 5
    assert numbers['n_with_both_answers'] == 5
    assert numbers['n_unique_prompt_ids'] == 4
    assert numbers['n_duplicate_prompt_ids'] == 1
    assert numbers['n_correct_equals_hallucinated'] == 1
    assert numbers['n_duplicate_correct_answers'] == 1
    assert numbers['n_duplicate_hallucinated_answers'] == 1
    assert numbers['n_empty_prompt'] == 0


def test_audit_pairs_lengths(canon):
    lengths = audit_pairs(canon)['lengths']
    assert lengths['correct_chars']['mean'] == pytest.approx(4.0)
    assert lengths['hallucinated_chars']['mean'] > lengths['correct_chars']['mean']
    assert lengths['diff_chars']['mean'] > 0
    assert lengths['diff_words']['mean'] > 0
    assert lengths['n_hallucinated_longer_chars'] == 4
    assert lengths['n_equal_chars'] == 1


def test_load_canonical_chain(raw_jsonl_path, canon):
    frame = load_halueval_qa_canonical(path=str(raw_jsonl_path))
    pd.testing.assert_frame_equal(frame, canon)


def _records_payload():
    return ''.join(
        json.dumps(record, ensure_ascii=False) + '\n' for record in RECORDS
    ).encode('utf-8')


class _FakeResponse:
    """Заглушка HTTP-ответа urllib: контекст-менеджер, read() -> bytes."""

    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self):
        return self._payload


def test_download_retry_then_success_and_atomic_cache(tmp_path, capsys):
    target = tmp_path / 'qa_data.json'
    payload = _records_payload()
    with mock.patch(
        'tda_metrics.nlp_datasets.urllib.request.urlopen',
        side_effect=[urllib.error.URLError('сеть'), _FakeResponse(payload)],
    ) as urlopen_mock:
        frame = load_halueval_qa(path=str(target))
    assert urlopen_mock.call_args_list == [
        mock.call(HALUEVAL_QA_URL, timeout=60),
        mock.call(HALUEVAL_QA_URL, timeout=60),
    ]
    assert 'retry' in capsys.readouterr().out
    assert frame.shape == (7, 4)
    assert target.read_bytes() == payload
    assert [p.name for p in tmp_path.iterdir()] == ['qa_data.json']


def test_download_retry_exhausted_raises_loudly(tmp_path):
    target = tmp_path / 'qa_data.json'
    with mock.patch(
        'tda_metrics.nlp_datasets.urllib.request.urlopen',
        side_effect=urllib.error.URLError('сеть недоступна'),
    ) as urlopen_mock:
        with pytest.raises(urllib.error.URLError):
            load_halueval_qa(path=str(target))
    assert urlopen_mock.call_count == 2
    assert list(tmp_path.iterdir()) == []


def test_expected_sha256_guard(raw_jsonl_path):
    with open(raw_jsonl_path, 'rb') as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    frame = load_halueval_qa(path=str(raw_jsonl_path), expected_sha256=digest)
    assert frame.shape == (7, 4)
    canon = load_halueval_qa_canonical(path=str(raw_jsonl_path), expected_sha256=digest)
    assert len(canon) == 5
    with pytest.raises(ValueError, match='sha256'):
        load_halueval_qa(path=str(raw_jsonl_path), expected_sha256='0' * 64)


def test_parse_error_reports_line_number(tmp_path):
    path = tmp_path / 'broken.json'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(RECORDS[0], ensure_ascii=False) + '\n')
        f.write('{"question": "незакрытый объект"\n')
    with pytest.raises(ValueError, match='строка 2'):
        load_halueval_qa(path=str(path))
