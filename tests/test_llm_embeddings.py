"""Инварианты llm_embeddings: resolve_layers, build_tokenized_inputs, extract_embeddings.

Полностью на моках (FakeTokenizer/FakeModel), без сети, CUDA и реальных
HF-моделей. FakeTokenizer кодирует текст посимвольно (id = 10 + ord(char)),
служебные токены шаблона — отдельные зарезервированные id (BOS/роль/генерация/
EOS); decode обратим и отбрасывает служебные id, поэтому decode-проверка
диапазона ответа в build_tokenized_inputs честная, а не заглушка. FakeModel
делает hidden[layer][pos] детерминированной функцией от input_id (id *
(layer+1), broadcast по hidden-размерности) — mean-пулинг проверяется точным
совпадением со средним input_id по диапазону ответа.
"""
import torch
import pytest

from tda_metrics.llm_embeddings import (
    AnswerSpanMismatchError,
    EmptyAnswerSpanError,
    PAD_ID,
    build_tokenized_inputs,
    extract_embeddings,
    resolve_layers,
)

BOS_ID = 1
USER_ID = 2
ASSISTANT_ID = 3
GEN_ID = 4
EOS_ID = 5
SPECIAL_IDS = {BOS_ID, USER_ID, ASSISTANT_ID, GEN_ID, EOS_ID}
CHAR_OFFSET = 10


def char_to_id(char):
    return CHAR_OFFSET + ord(char)


def id_to_char(token_id):
    return chr(token_id - CHAR_OFFSET)


class FakeTokenizer:
    """Посимвольный токенизатор с честным chat-шаблоном и обратимым decode.

    Шаблон моделирует соглашение реальных chat-шаблонов (Qwen и др.):
    сериализация с add_generation_prompt=True обрывается НА ОТКРЫТИИ
    assistant-оборота (роль без контента) и потому является строгим
    токен-префиксом полной сериализации с ответом.
    """

    def encode(self, text, add_special_tokens=False):
        return [char_to_id(c) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return ''.join(id_to_char(i) for i in ids if i not in SPECIAL_IDS)

    def apply_chat_template(self, messages, add_generation_prompt):
        ids = [BOS_ID]
        for message in messages:
            role_id = USER_ID if message['role'] == 'user' else ASSISTANT_ID
            ids.append(role_id)
            ids.extend(self.encode(message['content']))
        ids.extend(self._tail(add_generation_prompt))
        return ids

    def _tail(self, add_generation_prompt):
        return [ASSISTANT_ID] if add_generation_prompt else [EOS_ID]


class ShiftedFakeTokenizer(FakeTokenizer):
    """Сдвинутый шаблон генерации: лишний токен после открытия оборота — prefix не совпадает с началом полной сериализации (ловится префикс-проверкой и decode-проверкой)."""

    def _tail(self, add_generation_prompt):
        return [ASSISTANT_ID, GEN_ID] if add_generation_prompt else [EOS_ID]


class BatchEncodingTokenizer(FakeTokenizer):
    """Формат transformers 5.x: apply_chat_template возвращает BatchEncoding-подобный
    dict с 'input_ids' (плоский список) и 'attention_mask', а не list[int]."""

    def apply_chat_template(self, messages, add_generation_prompt):
        ids = super().apply_chat_template(messages, add_generation_prompt)
        return {'input_ids': ids, 'attention_mask': [1] * len(ids)}


class DriftedTemplateTokenizer(FakeTokenizer):
    """Дрейф шаблона, слепой для decode: спец-токен user в полной сериализации
    заменён на assistant; decode вырезает оба, префикс-проверка обязана ловить."""

    def apply_chat_template(self, messages, add_generation_prompt):
        ids = super().apply_chat_template(messages, add_generation_prompt)
        if not add_generation_prompt:
            for pos, token in enumerate(ids):
                if token == USER_ID:
                    ids[pos] = ASSISTANT_ID
                    break
        return ids


class TruncationDriftTokenizer(FakeTokenizer):
    """Первая полная сериализация честная, пересобранные (truncation-цикл) — с дрейфом спец-токена: ловится пост-truncation префикс-проверкой."""

    def __init__(self):
        self.full_calls = 0

    def apply_chat_template(self, messages, add_generation_prompt):
        ids = super().apply_chat_template(messages, add_generation_prompt)
        if not add_generation_prompt:
            self.full_calls += 1
            if self.full_calls > 1:
                for pos, token in enumerate(ids):
                    if token == USER_ID:
                        ids[pos] = ASSISTANT_ID
                        break
        return ids


def make_record(prompt_id='p0', prompt='Q', knowledge='K', correct_answer='yes',
                hallucinated_answer='no'):
    return {
        'prompt_id': prompt_id,
        'prompt': prompt,
        'knowledge': knowledge,
        'correct_answer': correct_answer,
        'hallucinated_answer': hallucinated_answer,
    }


class FakeOutputs:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class FakeModel:
    """hidden[layer][pos] = input_id * (layer + 1), broadcast по hidden-размерности."""

    def __init__(self, hidden_size=4, n_layers=3):
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, input_ids, attention_mask, output_hidden_states=True, use_cache=False):
        assert output_hidden_states is True
        assert use_cache is False
        values = input_ids.to(torch.float32)
        hidden_states = tuple(
            (values * (layer + 1)).unsqueeze(-1).expand(-1, -1, self.hidden_size)
            for layer in range(self.n_layers + 1)
        )
        return FakeOutputs(hidden_states)


def expected_pooled_value(ids, layer_index, pooling):
    if pooling == 'mean_answer':
        return sum(ids) / len(ids) * (layer_index + 1)
    return ids[-1] * (layer_index + 1)


@pytest.mark.parametrize('n_layers,expected', [
    (36, {0.25: 9, 0.5: 18, 0.75: 27, 1.0: 36}),
    (24, {0.25: 6, 0.5: 12, 0.75: 18, 1.0: 24}),
])
def test_resolve_layers_default_fractions(n_layers, expected):
    assert resolve_layers(n_layers) == expected


def test_resolve_layers_f1_is_last_layer():
    mapping = resolve_layers(7, fractions=(1.0,))
    assert mapping[1.0] == 7


def test_resolve_layers_duplicate_indices_small_n():
    mapping = resolve_layers(3, fractions=(0.1, 0.2, 0.3))
    assert mapping[0.1] == 0
    assert mapping[0.2] == 1
    assert mapping[0.3] == 1


def test_build_tokenized_inputs_span_matches_answer_and_excludes_prompt():
    tokenizer = FakeTokenizer()
    records = [make_record(prompt='What?', knowledge='Fact.', correct_answer='Yes.')]
    items = build_tokenized_inputs(records, tokenizer, answer_fields=('correct_answer',))

    item = items[0]
    start, end = item['answer_span']
    decoded = tokenizer.decode(item['input_ids'][start:end]).strip()
    assert decoded == 'Yes.'
    assert start > 0
    prefix_decoded = tokenizer.decode(item['input_ids'][:start])
    assert 'Yes' not in prefix_decoded


def test_build_tokenized_inputs_preserves_order_of_answer_fields():
    tokenizer = FakeTokenizer()
    records = [make_record(prompt_id='p0')]
    items = build_tokenized_inputs(records, tokenizer)
    assert [item['answer_kind'] for item in items] == ['correct', 'hallucinated']
    assert all(item['prompt_id'] == 'p0' for item in items)


def test_build_tokenized_inputs_empty_knowledge_uses_only_prompt():
    tokenizer = FakeTokenizer()
    records = [make_record(prompt='Hi', knowledge='', correct_answer='Ok')]
    items = build_tokenized_inputs(records, tokenizer, answer_fields=('correct_answer',))
    item = items[0]
    start, _ = item['answer_span']
    user_text = tokenizer.decode(item['input_ids'][:start])
    assert user_text == 'Hi'
    assert item['knowledge_tokens_dropped'] == 0


def test_build_tokenized_inputs_none_knowledge_treated_as_empty():
    tokenizer = FakeTokenizer()
    records = [make_record(prompt='Hi', knowledge=None, correct_answer='Ok')]
    items = build_tokenized_inputs(records, tokenizer, answer_fields=('correct_answer',))
    start, _ = items[0]['answer_span']
    assert tokenizer.decode(items[0]['input_ids'][:start]) == 'Hi'


def test_build_tokenized_inputs_batch_encoding_tokenizer_transformers5():
    tokenizer = BatchEncodingTokenizer()
    records = [make_record(prompt='What?', knowledge='Fact.', correct_answer='Yes.')]
    items = build_tokenized_inputs(records, tokenizer, answer_fields=('correct_answer',))
    item = items[0]
    start, end = item['answer_span']
    decoded = tokenizer.decode(item['input_ids'][start:end]).strip()
    assert decoded == 'Yes.'
    assert end > start


def test_build_tokenized_inputs_shifted_template_raises_mismatch():
    tokenizer = ShiftedFakeTokenizer()
    records = [make_record(prompt='Q', knowledge='K', correct_answer='Answer')]
    with pytest.raises(AnswerSpanMismatchError):
        build_tokenized_inputs(records, tokenizer, answer_fields=('correct_answer',))


def test_build_tokenized_inputs_prefix_alignment_honest_template_passes():
    tokenizer = FakeTokenizer()
    records = [make_record(prompt='What?', knowledge='Fact.', correct_answer='Yes.')]
    items = build_tokenized_inputs(records, tokenizer, answer_fields=('correct_answer',))
    item = items[0]
    start, end = item['answer_span']
    assert item['input_ids'][:start] == list(
        tokenizer.apply_chat_template(
            [{'role': 'user', 'content': 'Fact.\n\nWhat?'}],
            add_generation_prompt=True,
        )
    )
    assert item['n_pooled'] == end - start


def test_build_tokenized_inputs_prefix_drift_decode_blind_raises_mismatch():
    tokenizer = DriftedTemplateTokenizer()
    records = [make_record(prompt='Q', knowledge='K', correct_answer='Answer')]
    with pytest.raises(AnswerSpanMismatchError, match='не начинается с токенов префикса'):
        build_tokenized_inputs(records, tokenizer, answer_fields=('correct_answer',))


def test_build_tokenized_inputs_prefix_drift_after_truncation_raises_mismatch():
    tokenizer = TruncationDriftTokenizer()
    records = [make_record(prompt='Q', knowledge='K' * 60, correct_answer='Answer')]
    with pytest.raises(AnswerSpanMismatchError, match='не начинается с токенов префикса'):
        build_tokenized_inputs(records, tokenizer, answer_fields=('correct_answer',),
                               max_length=40)


@pytest.mark.parametrize('empty_answer', ['', '   '])
def test_build_tokenized_inputs_empty_or_blank_answer_raises(empty_answer):
    tokenizer = FakeTokenizer()
    records = [make_record(correct_answer=empty_answer)]
    with pytest.raises(EmptyAnswerSpanError):
        build_tokenized_inputs(records, tokenizer, answer_fields=('correct_answer',))


def test_build_tokenized_inputs_truncates_knowledge_and_keeps_answer():
    tokenizer = FakeTokenizer()
    records = [make_record(prompt='Q', knowledge='K' * 50, correct_answer='A')]
    items = build_tokenized_inputs(records, tokenizer, answer_fields=('correct_answer',), max_length=20)
    item = items[0]

    assert item['truncated'] is True
    assert len(item['input_ids']) <= 20
    assert item['knowledge_tokens_dropped'] > 0
    start, end = item['answer_span']
    decoded = tokenizer.decode(item['input_ids'][start:end]).strip()
    assert decoded == 'A'
    assert item['n_pooled'] == end - start
    assert item['n_pooled'] > 0


def test_build_tokenized_inputs_truncation_exhausted_raises():
    tokenizer = FakeTokenizer()
    records = [make_record(prompt='Q', knowledge='K' * 50, correct_answer='A')]
    with pytest.raises(EmptyAnswerSpanError):
        build_tokenized_inputs(records, tokenizer, answer_fields=('correct_answer',), max_length=3)


def test_build_tokenized_inputs_item_fields_present():
    tokenizer = FakeTokenizer()
    records = [make_record(prompt_id='pid-1')]
    items = build_tokenized_inputs(records, tokenizer, answer_fields=('correct_answer',))
    item = items[0]
    for key in ('prompt_id', 'answer_kind', 'input_ids', 'prefix_len', 'answer_span',
               'len_prompt_tokens', 'len_answer_tokens', 'n_pooled', 'truncated',
               'knowledge_tokens_dropped'):
        assert key in item
    assert item['prompt_id'] == 'pid-1'
    assert item['prefix_len'] == item['answer_span'][0]
    assert item['len_answer_tokens'] == item['n_pooled']


def test_build_tokenized_inputs_unknown_format_raises():
    tokenizer = FakeTokenizer()
    records = [make_record()]
    with pytest.raises(ValueError):
        build_tokenized_inputs(records, tokenizer, input_format='unknown_format')


def _build_items(tokenizer, prompt, knowledge, answer, prompt_id='p0'):
    records = [make_record(prompt_id=prompt_id, prompt=prompt, knowledge=knowledge, correct_answer=answer)]
    return build_tokenized_inputs(records, tokenizer, answer_fields=('correct_answer',))


def test_extract_embeddings_padding_does_not_affect_pooled_vector():
    tokenizer = FakeTokenizer()
    short_items = _build_items(tokenizer, 'Q', 'K', 'Hi', prompt_id='short')
    long_items = _build_items(tokenizer, 'Question much longer', 'Knowledge also longer here',
                             'Longer answer text', prompt_id='long')

    model = FakeModel(hidden_size=4, n_layers=2)
    layer_indices = [1, 2]
    layer_fractions = [0.5, 1.0]

    _, vectors_batch = extract_embeddings(
        short_items + long_items, model, layer_indices, layer_fractions,
        pooling='mean_answer', batch_size=2, device='cpu', dtype='float32',
    )
    _, vectors_solo = extract_embeddings(
        short_items, model, layer_indices, layer_fractions,
        pooling='mean_answer', batch_size=1, device='cpu', dtype='float32',
    )

    assert vectors_batch[0] == pytest.approx(vectors_solo[0])


def test_extract_embeddings_mean_pooling_exact():
    tokenizer = FakeTokenizer()
    items = _build_items(tokenizer, 'Question', 'Knowledge', 'Correct answer here')
    model = FakeModel(hidden_size=3, n_layers=4)
    layer_indices = [2, 4]
    layer_fractions = [0.5, 1.0]

    meta, vectors = extract_embeddings(
        items, model, layer_indices, layer_fractions,
        pooling='mean_answer', batch_size=8, device='cpu', dtype='float32',
    )

    start, end = items[0]['answer_span']
    span_ids = items[0]['input_ids'][start:end]
    for layer_pos, layer_index in enumerate(layer_indices):
        expected = expected_pooled_value(span_ids, layer_index, 'mean_answer')
        assert vectors[0, layer_pos] == pytest.approx([expected] * 3)

    assert vectors.shape == (1, 2, 3)
    assert set(meta['layer_index']) == {2, 4}
    assert set(meta['layer_fraction']) == {0.5, 1.0}
    assert meta.loc[meta['layer_index'] == 2, 'layer_fraction'].iloc[0] == 0.5


def test_extract_embeddings_last_answer_token_pooling():
    tokenizer = FakeTokenizer()
    items = _build_items(tokenizer, 'Q', 'K', 'Multi token answer')
    model = FakeModel(hidden_size=2, n_layers=1)
    layer_indices = [1]
    layer_fractions = [1.0]

    _, vectors = extract_embeddings(
        items, model, layer_indices, layer_fractions,
        pooling='last_answer_token', batch_size=4, device='cpu', dtype='float32',
    )

    start, end = items[0]['answer_span']
    last_id = items[0]['input_ids'][end - 1]
    expected = last_id * (layer_indices[0] + 1)
    assert vectors[0, 0] == pytest.approx([expected, expected])


def test_extract_embeddings_meta_row_count():
    tokenizer = FakeTokenizer()
    items = build_tokenized_inputs(
        [make_record(prompt_id='a'), make_record(prompt_id='b')],
        tokenizer, answer_fields=('correct_answer', 'hallucinated_answer'),
    )
    model = FakeModel(hidden_size=2, n_layers=3)
    layer_indices = [1, 2, 3]
    layer_fractions = [1 / 3, 2 / 3, 1.0]

    meta, vectors = extract_embeddings(
        items, model, layer_indices, layer_fractions,
        pooling='mean_answer', batch_size=3, device='cpu', dtype='float32',
    )

    assert len(meta) == len(items) * len(layer_indices)
    assert vectors.shape == (len(items), len(layer_indices), 2)


def test_extract_embeddings_invalid_pooling_raises():
    tokenizer = FakeTokenizer()
    items = _build_items(tokenizer, 'Q', 'K', 'Answer')
    model = FakeModel()
    with pytest.raises(ValueError):
        extract_embeddings(items, model, [1], [1.0], pooling='bad_pooling', device='cpu')


def test_extract_embeddings_mismatched_layer_lists_raises():
    tokenizer = FakeTokenizer()
    items = _build_items(tokenizer, 'Q', 'K', 'Answer')
    model = FakeModel()
    with pytest.raises(ValueError):
        extract_embeddings(items, model, [1, 2], [1.0], device='cpu')


def test_extract_embeddings_calls_model_eval():
    tokenizer = FakeTokenizer()
    items = _build_items(tokenizer, 'Q', 'K', 'Answer')
    model = FakeModel()
    extract_embeddings(items, model, [1], [1.0], device='cpu', dtype='float32')
    assert model.eval_called is True
    assert PAD_ID == 0 and PAD_ID not in SPECIAL_IDS


class LogitsToKeepModel(FakeModel):
    """Сигнатура forward как у CausalLM (logits_to_keep): extract обязан передать 1."""

    def forward(self, input_ids, attention_mask, output_hidden_states=True,
                use_cache=False, logits_to_keep=None):
        assert output_hidden_states is True
        assert use_cache is False
        self.logits_to_keep_seen = logits_to_keep
        values = input_ids.to(torch.float32)
        hidden_states = tuple(
            (values * (layer + 1)).unsqueeze(-1).expand(-1, -1, self.hidden_size)
            for layer in range(self.n_layers + 1)
        )
        return FakeOutputs(hidden_states)

    __call__ = forward


def test_extract_embeddings_passes_logits_to_keep_when_supported():
    tokenizer = FakeTokenizer()
    items = _build_items(tokenizer, 'Question', 'Knowledge', 'Correct answer here')
    model = LogitsToKeepModel(hidden_size=3, n_layers=4)
    layer_indices = [2, 4]
    layer_fractions = [0.5, 1.0]

    _, vectors = extract_embeddings(
        items, model, layer_indices, layer_fractions,
        pooling='mean_answer', batch_size=8, device='cpu', dtype='float32',
    )

    assert model.logits_to_keep_seen == 1
    start, end = items[0]['answer_span']
    span_ids = items[0]['input_ids'][start:end]
    for layer_pos, layer_index in enumerate(layer_indices):
        expected = expected_pooled_value(span_ids, layer_index, 'mean_answer')
        assert vectors[0, layer_pos] == pytest.approx([expected] * 3)


class OomOnBatchModel(FakeModel):
    """CUDA OOM на любом батче > 1 (эмуляция): extract обязан делить до singleton-батчей."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seen_batch_sizes = []

    def __call__(self, input_ids, attention_mask, output_hidden_states=True, use_cache=False):
        self.seen_batch_sizes.append(int(input_ids.shape[0]))
        if input_ids.shape[0] > 1:
            raise torch.cuda.OutOfMemoryError('тестовый OOM')
        return super().__call__(input_ids, attention_mask, output_hidden_states, use_cache)


def test_extract_embeddings_splits_batch_on_cuda_oom():
    tokenizer = FakeTokenizer()
    items = build_tokenized_inputs(
        [make_record(prompt_id='p0', correct_answer='Aa'),
         make_record(prompt_id='p1', correct_answer='Bb'),
         make_record(prompt_id='p2', correct_answer='Cc'),
         make_record(prompt_id='p3', correct_answer='Dd')],
        tokenizer, answer_fields=('correct_answer',),
    )
    model = OomOnBatchModel(hidden_size=2, n_layers=2)
    layer_indices = [1, 2]
    layer_fractions = [0.5, 1.0]

    meta, vectors = extract_embeddings(
        items, model, layer_indices, layer_fractions,
        pooling='mean_answer', batch_size=4, device='cpu', dtype='float32',
    )

    assert model.seen_batch_sizes == [4, 2, 1, 1, 2, 1, 1]
    assert vectors.shape == (4, 2, 2)
    assert list(meta['prompt_id']) == ['p0', 'p0', 'p1', 'p1', 'p2', 'p2', 'p3', 'p3']
    assert list(meta['layer_index']) == [1, 2] * 4
    for index, item in enumerate(items):
        start, end = item['answer_span']
        span_ids = item['input_ids'][start:end]
        for layer_pos, layer_index in enumerate(layer_indices):
            expected = expected_pooled_value(span_ids, layer_index, 'mean_answer')
            assert vectors[index, layer_pos] == pytest.approx([expected] * 2)
