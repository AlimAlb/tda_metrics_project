"""Скоринг уверенности модели в ответах: NLL, perplexity, энтропия, длина (мастер-план §L9).

Скоринг-проход поверх пунктов ``build_tokenized_inputs`` (teacher forcing):
для каждого пункта выполняется forward полного ``input_ids``, логиты позиций
``[span_start-1 .. span_end-2]`` предсказывают токены ответа
``[span_start .. span_end-1]``. Полный логит-тензор ``batch x seq x vocab``
CausalLM — доминирующий расход GPU-памяти; в отличие от
``extract_embeddings`` (``logits_to_keep=1``), здесь нужны логиты по всем
позициям ответа, поэтому при ``torch.cuda.OutOfMemoryError`` батч
рекурсивно делится пополам (скоринг поэлементный — разбиение не меняет
результат). Требует torch и модель с HF-совместимым ``__call__``
(``input_ids``/``attention_mask``/``use_cache`` -> ``outputs.logits``).
"""
import numpy as np
import torch

__all__ = ['score_answers', 'run_with_oom_split']


def run_with_oom_split(batch, process):
    """Публичная обёртка над ``llm_embeddings._split_on_oom``.

    ``process(batch)`` выполняется с обработкой CUDA OOM: батч делится
    пополам рекурсивно вплоть до singleton-батчей (печатает сообщение при
    каждом делении); пункт, не влезающий в одиночку, пробрасывает
    исходную ``OutOfMemoryError``.
    """
    from tda_metrics.llm_embeddings import _split_on_oom
    return _split_on_oom(batch, process)


def _validated_span(item):
    """Диапазон ответа: span_start >= 1 (есть префикс), span непуст и внутри input_ids."""
    span_start, span_end = item['answer_span']
    ids = item['input_ids']
    if span_start < 1:
        raise ValueError(
            f'prompt_id={item.get("prompt_id")!r}: span_start={span_start} < 1 — '
            'нет токена-предшественника для первого токена ответа (префикс отсутствует)'
        )
    if span_end <= span_start:
        raise ValueError(
            f'prompt_id={item.get("prompt_id")!r}: пустой диапазон ответа '
            f'({span_start}, {span_end})'
        )
    if span_end > len(ids):
        raise ValueError(
            f'prompt_id={item.get("prompt_id")!r}: span_end={span_end} превышает '
            f'длину input_ids={len(ids)}'
        )
    return span_start, span_end


def score_answers(inputs, model, batch_size=8, device='cuda', dtype='float16'):
    """Пер-пунктный скоринг уверенности: NLL, perplexity, энтропия, длина ответа.

    Для каждого пункта ``inputs`` (формат ``build_tokenized_inputs``:
    ``input_ids``, ``answer_span=(start, end)``, ``prompt_id``,
    ``answer_kind``) выполняется forward полного ``input_ids`` с правым
    паддингом (``llm_embeddings._pad_batch``; паддинг в диапазоны ответа
    попасть не может — они внутри реальной длины) и ``use_cache=False``.
    NLL — минус среднее ``log p(target)`` по позициям ответа (позиция
    ``t`` предсказывает токен ``t+1``); ``ppl = exp(NLL)``; ``entropy`` —
    среднее по позициям энтропии распределения следующего токена
    ``-sum p log p`` (log_softmax и энтропия — в float32); ``answer_length``
    — число токенов ответа (``span_end - span_start``). ``logits_to_keep``
    не передаётся: нужны логиты по всем позициям ответа; полный
    логит-тензор ``batch x seq x vocab`` — доминирующий расход памяти,
    при ``torch.cuda.OutOfMemoryError`` батч делится пополам
    (``run_with_oom_split``). Результат зависит от dtype модели: fp16-
    логиты дают другой NLL/entropy, чем float32 — к float32 приводится
    только срез логитов после forward. ``dtype`` оставлен для симметрии
    сигнатуры с ``extract_embeddings`` и на расчёт не влияет. Возвращает
    pandas.DataFrame со строками в порядке ``inputs`` и колонками
    ``prompt_id, answer_kind, nll, ppl, entropy, answer_length``;
    невалидный диапазон ответа — ValueError до первого forward.
    """
    import pandas as pd
    from tda_metrics.llm_embeddings import _pad_batch

    model.eval()
    for item in inputs:
        _validated_span(item)

    columns = ['prompt_id', 'answer_kind', 'nll', 'ppl', 'entropy', 'answer_length']
    rows = []

    def process(batch):
        """Forward батча + gather логитов по позициям ответа; список строк по пунктам."""
        input_ids_np, attention_mask_np = _pad_batch(batch)
        input_ids = torch.as_tensor(input_ids_np, dtype=torch.long, device=device)
        attention_mask = torch.as_tensor(attention_mask_np, dtype=torch.long, device=device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits
        per_item = []
        for item_pos, item in enumerate(batch):
            span_start, span_end = item['answer_span']
            token_logits = logits[item_pos, span_start - 1:span_end - 1, :].to(torch.float32)
            log_probs = torch.log_softmax(token_logits, dim=-1)
            targets = input_ids[item_pos, span_start:span_end].unsqueeze(1)
            token_log_probs = log_probs.gather(1, targets).squeeze(1)
            nll = float(-token_log_probs.mean())
            entropies = -(log_probs.exp() * log_probs).sum(dim=-1)
            per_item.append({
                'prompt_id': item['prompt_id'],
                'answer_kind': item['answer_kind'],
                'nll': nll,
                'ppl': float(np.exp(nll)),
                'entropy': float(entropies.mean()),
                'answer_length': int(span_end - span_start),
            })
        return per_item

    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start:start + batch_size]
            rows.extend(run_with_oom_split(batch, process))
    return pd.DataFrame(rows, columns=columns)
