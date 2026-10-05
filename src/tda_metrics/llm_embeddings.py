"""Извлечение скрытых представлений LLM с answer-masking (мастер-план §L1, ветка G Wave 2).

Формат входа зафиксирован решением оркестратора (D-006-усл.3):
``'chat_knowledge_v1'`` — знания включаются в промпт пользователя, полная
сериализация идёт через ``tokenizer.apply_chat_template`` модели. Сюда
входят три слоя: токенизация с teacher-forcing и проверкой маски ответа
(``build_tokenized_inputs``), forward pass с пулингом по слоям и диапазону
ответа (``extract_embeddings``), выбор индексов слоёв по долям глубины
(``resolve_layers``). Требует torch (CPU или CUDA) и tokenizer/model
с HF-совместимым интерфейсом (``apply_chat_template``, ``encode``,
``decode``, ``output_hidden_states=True``).
"""
import inspect

import numpy as np
import torch

__all__ = [
    'INPUT_FORMATS',
    'resolve_layers',
    'build_tokenized_inputs',
    'extract_embeddings',
    'AnswerMaskError',
    'EmptyAnswerSpanError',
    'AnswerSpanMismatchError',
]

PAD_ID = 0
POOLINGS = ('mean_answer', 'last_answer_token')
_NUMPY_DTYPES = {'float16': np.float16, 'float32': np.float32, 'bfloat16': np.float32}
_MAX_TRUNCATION_STEPS = 10_000


class AnswerMaskError(Exception):
    """Базовая ошибка построения маски ответа при токенизации."""


class EmptyAnswerSpanError(AnswerMaskError):
    """Диапазон ответа пуст либо не выживает после усечения знаний."""


class AnswerSpanMismatchError(AnswerMaskError):
    """Декодированный диапазон ответа не совпадает с исходным текстом ответа."""


def _format_chat_knowledge_v1(knowledge, prompt):
    text = (knowledge or '').strip()
    return f'{text}\n\n{prompt}' if text else prompt


INPUT_FORMATS = {'chat_knowledge_v1': _format_chat_knowledge_v1}


def resolve_layers(n_layers, fractions=(0.25, 0.5, 0.75, 1.0)):
    """Индексы hidden_states по долям глубины сети.

    hidden_states содержит ``n_layers + 1`` элементов (индекс 0 — слой
    эмбеддингов); индекс доли ``f`` — ``round(f * n_layers)``, доля 1.0 даёт
    последний слой. Разные доли могут указывать на один и тот же индекс
    (малое число слоёв) — дубликаты не схлопываются вызывающим кодом.
    """
    return {fraction: round(fraction * n_layers) for fraction in fractions}


def _chat_template_ids(tokenizer, messages, add_generation_prompt):
    """Идентификаторы chat-шаблона в list[int] — независимо от версии transformers.

    transformers 4.x возвращает list[int]; 5.x — BatchEncoding (dict с
    'input_ids' плоским списком либо батч-обёрткой [[ids]]); str — при
    tokenize=False. Проверено на Qwen2.5-3B + transformers 5.17: prefix
    (add_generation_prompt=True) — строгий токен-префикс полной сериализации.
    """
    result = tokenizer.apply_chat_template(
        messages, add_generation_prompt=add_generation_prompt)
    if isinstance(result, str):
        return tokenizer.encode(result, add_special_tokens=False)
    ids = result['input_ids'] if (
        hasattr(result, '__getitem__') and 'input_ids' in result) else result
    if hasattr(ids, 'tolist'):
        ids = ids.tolist()
    ids = list(ids)
    if ids and isinstance(ids[0], (list, tuple)):
        ids = list(ids[0])
    return ids


def _tokenize_messages(tokenizer, format_fn, knowledge, prompt, answer):
    user_message = {'role': 'user', 'content': format_fn(knowledge, prompt)}
    assistant_message = {'role': 'assistant', 'content': answer}
    prefix = _chat_template_ids(tokenizer, [user_message], add_generation_prompt=True)
    full = _chat_template_ids(tokenizer, [user_message, assistant_message],
                              add_generation_prompt=False)
    return prefix, full


def _validate_prefix_alignment(full, prefix, prompt_id):
    """Токен-префиксная проверка: full обязан начинаться с токенов prefix.

    Две сериализации apply_chat_template (префикс с add_generation_prompt=True
    и полный оборот с ответом) обязаны быть согласованы до токена: расхождение
    означает склейку BPE-границ или дрейф шаблона — тогда span, построенный
    от len(prefix), смещён относительно реального текста ответа, даже если
    decode-проверка со skip_special_tokens=True текст «не видит» (спец-токены
    вырезаются при decode). Ловится здесь.
    """
    if full[:len(prefix)] == prefix:
        return
    first = next(
        (pos for pos, (a, b) in enumerate(zip(full, prefix)) if a != b),
        min(len(full), len(prefix)),
    )
    raise AnswerSpanMismatchError(
        f'prompt_id={prompt_id}: полная сериализация не начинается с токенов '
        f'префикса (первое расхождение на позиции {first}: '
        f'prefix={prefix[first] if first < len(prefix) else None!r}, '
        f'full={full[first] if first < len(full) else None!r}; '
        f'len_prefix={len(prefix)}, len_full={len(full)}) — вероятно, '
        'склейка токенов на границе шаблона и ответа либо дрейф шаблона'
    )


def _validate_answer_span(tokenizer, full, prefix_len, answer, prompt_id):
    if len(full) <= prefix_len:
        raise EmptyAnswerSpanError(f'prompt_id={prompt_id}: диапазон ответа пуст')
    decoded = tokenizer.decode(full[prefix_len:], skip_special_tokens=True).strip()
    if decoded != answer.strip():
        raise AnswerSpanMismatchError(
            f'prompt_id={prompt_id}: декодированный диапазон {decoded!r} '
            f'не совпадает с ответом {answer.strip()!r}'
        )


def _build_item(tokenizer, format_fn, prompt_id, answer_kind, knowledge, prompt, answer, max_length):
    if answer is None or answer.strip() == '':
        raise EmptyAnswerSpanError(f'prompt_id={prompt_id}: ответ пуст')
    knowledge_text = knowledge or ''
    knowledge_ids = tokenizer.encode(knowledge_text, add_special_tokens=False) if knowledge_text else []
    prefix, full = _tokenize_messages(tokenizer, format_fn, knowledge_text, prompt, answer)
    prefix_len = len(prefix)
    _validate_prefix_alignment(full, prefix, prompt_id)
    _validate_answer_span(tokenizer, full, prefix_len, answer, prompt_id)

    truncated = False
    remaining = list(knowledge_ids)
    steps = 0
    while len(full) > max_length:
        steps += 1
        if steps > _MAX_TRUNCATION_STEPS:
            raise EmptyAnswerSpanError(f'prompt_id={prompt_id}: превышен лимит шагов усечения знаний')
        if not remaining:
            raise EmptyAnswerSpanError(
                f'prompt_id={prompt_id}: знания исчерпаны, длина {len(full)} всё ещё '
                f'превышает max_length={max_length}'
            )
        overflow = len(full) - max_length
        cut = min(len(remaining), max(overflow, 1))
        remaining = remaining[:len(remaining) - cut]
        new_knowledge_text = tokenizer.decode(remaining) if remaining else ''
        prefix, full = _tokenize_messages(tokenizer, format_fn, new_knowledge_text, prompt, answer)
        prefix_len = len(prefix)
        truncated = True

    if truncated:
        _validate_prefix_alignment(full, prefix, prompt_id)
        _validate_answer_span(tokenizer, full, prefix_len, answer, prompt_id)

    answer_span = (prefix_len, len(full))
    n_pooled = answer_span[1] - answer_span[0]
    return {
        'prompt_id': prompt_id,
        'answer_kind': answer_kind,
        'input_ids': list(full),
        'prefix_len': prefix_len,
        'answer_span': answer_span,
        'len_prompt_tokens': prefix_len,
        'len_answer_tokens': n_pooled,
        'n_pooled': n_pooled,
        'truncated': truncated,
        'knowledge_tokens_dropped': len(knowledge_ids) - len(remaining),
    }


def build_tokenized_inputs(records, tokenizer, answer_fields=('correct_answer', 'hallucinated_answer'),
                           max_length=1024, input_format='chat_knowledge_v1'):
    """Teacher-forced токенизация записей HaluEval QA с маской ответа.

    ``records`` — итерируемый dicts с ключами canonical-контракта
    (``prompt_id, prompt, knowledge`` + ``answer_fields``). Для каждой записи
    и каждого поля ответа строится ``prefix`` (промпт + шаблон генерации) и
    ``full`` (промпт + сериализованный ответ); диапазон ответа —
    ``(len(prefix), len(full))``, проверяется decode-сверкой с исходным
    текстом. При переполнении ``max_length`` знания урезаются с конца
    (вопрос, шаблон и ответ не трогаются); если ответ не выживает после
    полного исчерпания знаний — ``EmptyAnswerSpanError``. Порядок точек
    (запись × поле ответа, в порядке ``answer_fields``) сохраняется.
    """
    if input_format not in INPUT_FORMATS:
        raise ValueError(f'неизвестный input_format {input_format!r}; доступны {sorted(INPUT_FORMATS)}')
    format_fn = INPUT_FORMATS[input_format]
    items = []
    for record in records:
        prompt_id = record['prompt_id']
        prompt = record['prompt']
        knowledge = record.get('knowledge')
        for field in answer_fields:
            answer_kind = field[:-len('_answer')] if field.endswith('_answer') else field
            answer = record[field]
            items.append(_build_item(tokenizer, format_fn, prompt_id, answer_kind, knowledge, prompt, answer, max_length))
    return items


def _pad_batch(batch):
    max_len = max(len(item['input_ids']) for item in batch)
    input_ids = np.full((len(batch), max_len), PAD_ID, dtype=np.int64)
    attention_mask = np.zeros((len(batch), max_len), dtype=np.int64)
    for row, item in enumerate(batch):
        ids = item['input_ids']
        input_ids[row, :len(ids)] = ids
        attention_mask[row, :len(ids)] = 1
    return input_ids, attention_mask


def _pool_span(layer_hidden, start, end, pooling):
    span = layer_hidden[start:end]
    return span.mean(dim=0) if pooling == 'mean_answer' else span[-1]


def _logits_to_keep_kwargs(model):
    """Kwarg пропуска логитов lm_head (значение 1), если сигнатура его поддерживает.

    Полный логит-тензор ``batch x seq x vocab`` CausalLM — доминирующий расход
    GPU-памяти при ``output_hidden_states=True`` (repro: CUDA OOM 05.10,
    Qwen2.5-3B, 2.76 GiB на одном батче), а для пулинга hidden states логиты
    не нужны вовсе; слайс на hidden states не влияет. Имя параметра менялось
    между версиями transformers (``num_logits_to_keep`` / ``logits_to_keep``) —
    берётся первое поддерживаемое; модели без него (моки, старые сигнатуры)
    вызываются без kwarg.
    """
    for attribute in ('forward', '__call__'):
        method = getattr(model, attribute, None)
        if method is None:
            continue
        try:
            parameters = inspect.signature(method).parameters
        except (TypeError, ValueError):
            continue
        for name in ('num_logits_to_keep', 'logits_to_keep'):
            if name in parameters:
                return {name: 1}
    return {}


def _split_on_oom(batch, process):
    """Обработка батча с рекурсивным делением пополам при CUDA OOM.

    Пулинг поэлементный, поэтому результат не зависит от разбиения батча:
    длинные последовательности, не влезающие в память целиком, обрабатываются
    половинами вплоть до singleton-батчей; пункт, не влезающий в одиночку,
    пробрасывает исходную ``OutOfMemoryError``.
    """
    try:
        return process(batch)
    except torch.cuda.OutOfMemoryError:
        if len(batch) == 1:
            raise
        torch.cuda.empty_cache()
        mid = (len(batch) + 1) // 2
        print(f'CUDA OOM: батч {len(batch)} делится пополам ({mid}+{len(batch) - mid})')
        return (_split_on_oom(batch[:mid], process)
                + _split_on_oom(batch[mid:], process))


def extract_embeddings(inputs, model, layer_indices, layer_fractions, pooling='mean_answer',
                       batch_size=32, device='cuda', dtype='float16'):
    """Forward pass с пулингом скрытых состояний по диапазону ответа.

    ``inputs`` — результат ``build_tokenized_inputs``. Паддинг правый,
    ручной (``PAD_ID=0`` + attention-маска); порядок пунктов и батчей не
    меняется. Пулинг ('mean_answer' — среднее, 'last_answer_token' —
    последний токен) берётся по ``item['answer_span']`` поверх реального
    (непаддингового) диапазона — паддинг физически не может попасть в span.
    'last_answer_token' возвращает последний токен span, т.е. хвост шаблона
    assistant-оборота (для Qwen2.5 — токен после ``<|im_end|>``), а не
    последний контентный токен ответа; семантику режима обязательно
    оговаривать в таблицах результатов.
    Возвращает ``(meta, vectors)``: ``meta`` — DataFrame с одной строкой на
    пару (пункт, слой) в порядке ``inputs`` × ``layer_indices``; ``vectors``
    — ``np.ndarray`` формы ``(len(inputs), len(layer_indices), hidden)``.
    Логиты lm_head не вычисляются по всей последовательности: при поддержке
    сигнатурой передаётся ``num_logits_to_keep``/``logits_to_keep=1``
    (``_logits_to_keep_kwargs``); при ``torch.cuda.OutOfMemoryError`` батч
    рекурсивно делится пополам (``_split_on_oom``) — пулинг поэлементный,
    разбиение не меняет результат.
    """
    import pandas as pd

    if len(layer_indices) != len(layer_fractions):
        raise ValueError('layer_indices и layer_fractions должны быть одинаковой длины')
    if pooling not in POOLINGS:
        raise ValueError(f'неизвестный pooling {pooling!r}; доступны {POOLINGS}')

    numpy_dtype = _NUMPY_DTYPES.get(dtype, np.float32)
    model.eval()
    logits_kwargs = _logits_to_keep_kwargs(model)

    meta_rows = []
    vector_rows = []

    def process(batch):
        """Forward одного батча + пулинг; список (meta-строки, vectors) по пунктам."""
        input_ids_np, attention_mask_np = _pad_batch(batch)
        input_ids = torch.as_tensor(input_ids_np, dtype=torch.long, device=device)
        attention_mask = torch.as_tensor(attention_mask_np, dtype=torch.long, device=device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True, use_cache=False, **logits_kwargs)
        hidden_states = outputs.hidden_states

        per_item = []
        for item_index, item in enumerate(batch):
            span_start, span_end = item['answer_span']
            hidden_size = hidden_states[layer_indices[0]].shape[-1]
            item_vectors = np.zeros((len(layer_indices), hidden_size), dtype=numpy_dtype)
            item_meta = []
            for layer_pos, layer_index in enumerate(layer_indices):
                layer_hidden = hidden_states[layer_index][item_index]
                pooled = _pool_span(layer_hidden, span_start, span_end, pooling)
                item_vectors[layer_pos] = pooled.detach().to('cpu').float().numpy().astype(numpy_dtype)
                item_meta.append({
                    'prompt_id': item['prompt_id'],
                    'answer_kind': item['answer_kind'],
                    'layer_index': layer_index,
                    'layer_fraction': layer_fractions[layer_pos],
                    'pooling': pooling,
                    'input_len': len(item['input_ids']),
                    'n_pooled': item['n_pooled'],
                    'truncated': item['truncated'],
                })
            per_item.append((item_meta, item_vectors))
        return per_item

    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start:start + batch_size]
            for item_meta, item_vectors in _split_on_oom(batch, process):
                meta_rows.extend(item_meta)
                vector_rows.append(item_vectors)

    vectors = np.stack(vector_rows, axis=0) if vector_rows else np.zeros((0, len(layer_indices), 0), dtype=numpy_dtype)
    meta = pd.DataFrame(meta_rows)
    return meta, vectors
