"""Инварианты llm_confidence: NLL/perplexity/энтропия/длина на мок-модели.

Мок NextTokenModel строит logits[i, t] = onehot(input_ids[i, t]) масштаба
BIG/SMALL и запоминает kwargs каждого вызова: распределение следующего
тока на позиции t имеет моду в текущем токене, поэтому log p целевых
токенов диапазона ответа считаются вручную — позиции внутри ответа
(текущий токен равен токенам ответа) дают «сильную» вероятность, первая
позиция span (текущий токен из префикса) — «слабую». Паддинг правый и в
диапазоны ответа не попадает — скоринг в батче совпадает с поодиночке.
Всё на CPU, без сети и реальных HF-моделей.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tda_metrics.llm_confidence import score_answers

VOCAB = 50
BIG = 8.0
SMALL = -1.0


class NextTokenModel:
    """logits[i, t] = onehot(input_ids[i, t]) масштаба BIG/SMALL; запоминает kwargs вызовов."""

    def __init__(self, vocab=VOCAB, big=BIG, small=SMALL):
        self.vocab = vocab
        self.big = big
        self.small = small
        self.calls = []
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        input_ids = kwargs['input_ids']
        logits = torch.full(
            (input_ids.shape[0], input_ids.shape[1], self.vocab),
            self.small, dtype=torch.float32)
        logits.scatter_(2, input_ids.unsqueeze(2), self.big)
        return SimpleNamespace(logits=logits)


def make_item(prompt_id, answer_kind, prefix_ids, answer_ids):
    ids = list(prefix_ids) + list(answer_ids)
    return {
        'prompt_id': prompt_id,
        'answer_kind': answer_kind,
        'input_ids': ids,
        'answer_span': (len(prefix_ids), len(ids)),
    }


def logprob_of(current_token, target_token, vocab=VOCAB, big=BIG, small=SMALL):
    """log p(target) распределения log_softmax(onehot(current_token) масштаба big/small)."""
    logits = torch.full((vocab,), small, dtype=torch.float64)
    logits[current_token] = big
    return float(torch.log_softmax(logits, dim=-1)[target_token])


def entropy_of(current_token, vocab=VOCAB, big=BIG, small=SMALL):
    """Энтропия распределения из logprob_of (натуральные логарифмы)."""
    logits = torch.full((vocab,), small, dtype=torch.float64)
    logits[current_token] = big
    log_probs = torch.log_softmax(logits, dim=-1)
    return float(-(log_probs.exp() * log_probs).sum())


def test_score_answers_nll_ppl_entropy_manual():
    model = NextTokenModel()
    answer_token = 9
    prefix = [3, 5, 7]
    answer = [answer_token] * 4
    item = make_item('p1', 'correct', prefix, answer)
    scores = score_answers([item], model, batch_size=4, device='cpu', dtype='float32')

    strong = logprob_of(answer_token, answer_token)
    weak = logprob_of(prefix[-1], answer_token)
    expected_nll = -((weak + 3 * strong) / 4.0)
    assert scores['nll'][0] == pytest.approx(expected_nll, rel=1e-5)
    assert scores['ppl'][0] == pytest.approx(float(np.exp(expected_nll)), rel=1e-5)
    expected_entropy = (entropy_of(prefix[-1]) + 3 * entropy_of(answer_token)) / 4.0
    assert scores['entropy'][0] == pytest.approx(expected_entropy, rel=1e-5)


def test_score_answers_model_call_kwargs():
    model = NextTokenModel()
    items = [
        make_item('p1', 'correct', [1, 2], [3, 3]),
        make_item('p2', 'hallucinated', [4], [5, 5, 5]),
    ]
    score_answers(items, model, batch_size=2, device='cpu')
    assert model.eval_called is True
    assert len(model.calls) == 1
    for call in model.calls:
        assert set(call) == {'input_ids', 'attention_mask', 'use_cache'}
        assert call['use_cache'] is False


def test_score_answers_length_and_fields_passthrough():
    model = NextTokenModel()
    items = [
        make_item('pA', 'correct', [1, 2, 3], [4, 5, 6, 7, 8]),
        make_item('pB', 'hallucinated', [9, 8], [7, 6]),
    ]
    scores = score_answers(items, model, device='cpu')
    assert list(scores['prompt_id']) == ['pA', 'pB']
    assert list(scores['answer_kind']) == ['correct', 'hallucinated']
    assert list(scores['answer_length']) == [5, 2]
    assert scores['nll'][0] < 10.0 and scores['nll'][1] < 10.0


def test_score_answers_padding_invariant_across_batch():
    model = NextTokenModel()
    short = make_item('short', 'correct', [1, 2], [3, 3, 3])
    long = make_item('long', 'hallucinated',
                     [4, 5, 6, 7, 8, 9, 10, 11], [2, 2, 2, 2, 2, 2, 2, 2, 2])
    solo_short = score_answers([short], model, device='cpu')
    solo_long = score_answers([long], model, device='cpu')
    batch = score_answers([short, long], model, batch_size=2, device='cpu')
    columns = ['nll', 'ppl', 'entropy']
    assert np.allclose(batch.loc[0, columns].to_numpy(dtype=float),
                       solo_short.loc[0, columns].to_numpy(dtype=float))
    assert np.allclose(batch.loc[1, columns].to_numpy(dtype=float),
                       solo_long.loc[0, columns].to_numpy(dtype=float))
    assert batch.loc[1, 'answer_length'] == solo_long.loc[0, 'answer_length']


def test_score_answers_rejects_invalid_spans_before_forward():
    model = NextTokenModel()
    item = make_item('p1', 'correct', [1, 2, 3], [4, 4])
    for span in [(0, 2), (2, 2), (4, 9)]:
        bad = dict(item, answer_span=span)
        with pytest.raises(ValueError):
            score_answers([bad], model, device='cpu')
    assert model.calls == []


def test_score_answers_empty_inputs():
    model = NextTokenModel()
    scores = score_answers([], model, device='cpu')
    assert len(scores) == 0
    assert list(scores.columns) == ['prompt_id', 'answer_kind', 'nll', 'ppl',
                                    'entropy', 'answer_length']
    assert model.calls == []
