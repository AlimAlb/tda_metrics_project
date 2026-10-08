"""Тесты Wave ST: датасетный слой (HotpotQA -> ST-промпты) — CPU, без heavy-стека."""
import json

import pytest

from tda_metrics.st_datasets import (
    assemble_knowledge,
    load_st_records,
    normalize_question,
    question_hash,
    records_sha256,
    save_st_records,
    st_records_from_hotpot,
)


def _hotpot_row(question, sentences, titles, fact_titles, fact_ids, answer='gold',
                hotpot_type='bridge'):
    return {
        'question': question,
        'answer': answer,
        'type': hotpot_type,
        'context': {'title': titles, 'sentences': sentences},
        'supporting_facts': {'title': fact_titles, 'sent_id': fact_ids},
    }


def test_normalize_question_strips_punctuation_and_case():
    assert normalize_question('Who??  Wrote — "Romeo"!') == 'who wrote romeo'
    assert normalize_question('AbC  dEf') == normalize_question('abc def')


def test_question_hash_deterministic_and_sensitive():
    assert question_hash('Where?') == question_hash('where')
    assert question_hash('Where?') != question_hash('When?')


def test_assemble_knowledge_order_dedupe_and_unknown_title():
    sentences = [['s0', 's1'], ['t0', 't1']]
    knowledge, n = assemble_knowledge(
        {'title': ['A', 'B'], 'sentences': sentences},
        {'title': ['B', 'A', 'B'], 'sent_id': [1, 0, 1]})
    assert knowledge == 't1 s0'
    assert n == 2


def test_records_from_hotpot_determinism_and_exclusion():
    rows = [
        _hotpot_row(f'Question number {index}?',
                    [[f'Knowledge sentence {index} with enough words to pass filters '
                      f'and another chunk of text here' for _ in range(1)]],
                    ['A'], ['A'], [0])
        for index in range(30)
    ]
    exclude = {question_hash(rows[0]['question'])}
    first = st_records_from_hotpot(rows, n_prompts=5, seed=1,
                                   exclude_hashes=exclude)
    second = st_records_from_hotpot(rows, n_prompts=5, seed=1,
                                    exclude_hashes=exclude)
    other = st_records_from_hotpot(rows, n_prompts=5, seed=2,
                                    exclude_hashes=exclude)
    assert [r['prompt_id'] for r in first] == [r['prompt_id'] for r in second]
    assert first[0]['question'] == second[0]['question']
    assert first != other
    assert len(first) == 5
    assert first[0]['prompt_id'] == 'st_0000'
    hashes = {r['question_hash'] for r in first}
    assert not (hashes & exclude)


def test_records_from_hotpot_filters_apply():
    short = _hotpot_row('q?', [['tiny fact']], ['A'], ['A'], [0])
    long_knowledge = _hotpot_row(
        'q2?', [['word ' * 200]], ['A'], ['A'], [0])
    good = _hotpot_row(
        'q3?', [['enough words here to pass the knowledge filter easily ok']],
        ['A'], ['A'], [0])
    records = st_records_from_hotpot([short, long_knowledge, good],
                                     n_prompts=1, seed=0, exclude_hashes=set())
    assert len(records) == 1
    assert records[0]['question'] == 'q3?'


def test_records_from_hotpot_not_enough_candidates_raises():
    rows = [_hotpot_row('only one?', [['just enough words to be accepted here']],
                        ['A'], ['A'], [0])]
    with pytest.raises(ValueError):
        st_records_from_hotpot(rows, n_prompts=2, seed=0, exclude_hashes=set())


def test_save_load_roundtrip_and_sha(tmp_path):
    records = [{'prompt_id': 'st_0000', 'question': 'q', 'knowledge': 'k',
                'gold_answer': 'a', 'question_hash': 'h' * 64,
                'n_knowledge_sentences': 1, 'hotpot_type': 'bridge'}]
    path = tmp_path / 'prompts.json'
    digest = save_st_records(records, str(path))
    assert (tmp_path / 'prompts.json.sha256').exists()
    loaded = load_st_records(str(path), expected_sha256=digest)
    assert loaded == records
    with pytest.raises(ValueError):
        load_st_records(str(path), expected_sha256='bad' * 8)


def test_records_sha256_independent_of_key_order():
    first = records_sha256([{'a': 1, 'b': 2}])
    second = records_sha256([{'b': 2, 'a': 1}])
    assert first == second
