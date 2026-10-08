"""Извлечение скрытых представлений LLM на HaluEval QA.

Запускается на Colab VM (heavy-стек: torch + transformers). Поток: canonical
HaluEval QA (тонкий адаптер nlp_datasets) -> teacher-forced токенизация с
answer-masking (llm_embeddings.build_tokenized_inputs) -> forward pass и
пулинг по слоям (llm_embeddings.extract_embeddings) -> атомарный дисковый
кэш с манифестом (embedding_cache.save_cache). Формат входа фиксирован
протоколом эксперимента: knowledge включается в промпт,
сериализация через chat-шаблон модели, идентификатор 'chat_knowledge_v1'.

Прогресс печатается чанками по ~200 пунктов (пункт = одна пара
prompt x answer_kind); чанкинг также держит пиковую память извлечения под
контролем на длинных прогонах. Против смерти сессий Colab: токенизированные
inputs сохраняются в {out_dir}/extract_inputs.pkl, каждые --checkpoint-every
чанков атомарно пишется {out_dir}/partial.pkl (vectors/meta/done_count +
параметры прогона); при рестарте совместимый partial подхватывается и
извлечение продолжается с done_count. SystemExit в логике прогона не
поднимается (только штатное завершение интерпретатора); разбор аргументов
командной строки — argparse, его собственное поведение при --help/ошибках
использования не затрагивается этим требованием (как и в
run_power_analysis.py).
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import torch as _torch

if _torch.cuda.is_available():
    _torch.cuda.init()

from tda_metrics.embedding_cache import prompt_ids_hash, save_cache
from tda_metrics.llm_embeddings import (
    INPUT_FORMATS,
    build_tokenized_inputs,
    extract_embeddings,
    resolve_layers,
)
from tda_metrics.nlp_datasets import HALUEVAL_QA_SHA256, load_halueval_qa_canonical

PROGRESS_CHUNK = 200

TEMPLATE_NOTES = (
    "chat_knowledge_v1: user-сообщение = knowledge + '\\n\\n' + prompt "
    '(knowledge пустой -> только prompt), assistant-сообщение = ответ; '
    'сериализация через tokenizer.apply_chat_template'
)

_DTYPE_MAP = {
    'float16': 'float16',
    'bfloat16': 'bfloat16',
    'float32': 'float32',
}


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Извлечение hidden states Qwen2.5 на HaluEval QA с answer-masking и пулингом.'
    )
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-3B-Instruct')
    parser.add_argument('--limit', type=int, default=None, help='ограничить число записей (по умолчанию все)')
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-length', type=int, default=1024)
    parser.add_argument('--input-format', type=str, default='chat_knowledge_v1')
    parser.add_argument('--pooling', type=str, default='mean_answer')
    parser.add_argument('--data-path', type=str, default=None,
                       help='локальный кэш qa_data.json; при отсутствии — скачивание с sha256-проверкой')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dtype', type=str, default='float16', choices=sorted(_DTYPE_MAP))
    parser.add_argument('--checkpoint-every', type=int, default=5,
                        help='чекпоинт partial.pkl каждые N чанков (200 пунктов в чанке)')
    return parser


def _atomic_pickle(path, payload):
    """Атомарная pickle-запись: tmp + fsync + replace (частичная запись исключена)."""
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _percentile_summary(values):
    array = np.asarray(values, dtype=float)
    percentiles = np.percentile(array, [10, 25, 50, 75, 90]) if len(array) else [float('nan')] * 5
    return {
        'mean': float(array.mean()) if len(array) else float('nan'),
        'min': float(array.min()) if len(array) else float('nan'),
        'p10': float(percentiles[0]),
        'p25': float(percentiles[1]),
        'p50': float(percentiles[2]),
        'p75': float(percentiles[3]),
        'p90': float(percentiles[4]),
        'max': float(array.max()) if len(array) else float('nan'),
    }


def _resolve_torch_dtype(dtype_name):
    import torch
    return {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }[dtype_name]


def _load_model_and_tokenizer(model_name, device, dtype_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=_resolve_torch_dtype(dtype_name))
    model = model.to(device)
    model.eval()
    return tokenizer, model


def _model_revision(model):
    return getattr(model.config, '_commit_hash', None)


def run(args):
    if args.input_format not in INPUT_FORMATS:
        raise ValueError(
            f'неизвестный input_format {args.input_format!r}; доступны {sorted(INPUT_FORMATS)}'
        )

    print(f'=== загрузка HaluEval QA (canonical), data_path={args.data_path} ===')
    canonical = load_halueval_qa_canonical(path=args.data_path, expected_sha256=HALUEVAL_QA_SHA256)
    if args.limit is not None:
        if args.limit <= 0:
            print(f'--limit={args.limit}: записей для извлечения нет — штатное завершение')
            return
        canonical = canonical.iloc[:args.limit].reset_index(drop=True)
    n_records = len(canonical)
    if n_records == 0:
        print('датасет пуст: записей для извлечения нет — штатное завершение')
        return
    print(f'записей после --limit: {n_records}')

    print(f'=== загрузка модели {args.model} (device={args.device}, dtype={args.dtype}) ===')
    tokenizer, model = _load_model_and_tokenizer(args.model, args.device, args.dtype)

    n_layers = getattr(model.config, 'num_hidden_layers', None)
    if n_layers is None:
        raise ValueError('model.config.num_hidden_layers отсутствует: не удаётся определить число слоёв')
    fraction_to_index = resolve_layers(n_layers)
    ordered = sorted(fraction_to_index.items())
    layer_fractions = [fraction for fraction, _ in ordered]
    layer_indices = [index for _, index in ordered]
    print(f'слоёв в модели: {n_layers}; извлекаемые индексы: {dict(ordered)}')

    print('=== токенизация (answer-masking, teacher-forcing) ===')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_params = {
        'model': args.model,
        'input_format': args.input_format,
        'pooling': args.pooling,
        'max_length': args.max_length,
        'dtype': args.dtype,
        'layer_indices': layer_indices,
        'batch_size': args.batch_size,
        'chunk': PROGRESS_CHUNK,
        'n_records': n_records,
        'limit': args.limit,
    }
    records = canonical.to_dict('records')
    inputs = build_tokenized_inputs(
        records, tokenizer,
        max_length=args.max_length,
        input_format=args.input_format,
    )
    n_items = len(inputs)
    print(f'пунктов (prompt x answer_kind): {n_items}')
    inputs_path = out_dir / 'extract_inputs.pkl'
    _atomic_pickle(inputs_path, {'params': run_params, 'n_items': n_items, 'inputs': inputs})
    print(f'inputs сохранены (токенизация не будет повторяться): {inputs_path}')

    partial_path = out_dir / 'partial.pkl'
    base_meta = None
    base_vectors = None
    done_count = 0
    if partial_path.exists():
        try:
            with open(partial_path, 'rb') as f:
                partial = pickle.load(f)
            if partial.get('params') == run_params and 0 < partial.get('done_count', 0) <= n_items:
                base_meta = partial['meta']
                base_vectors = partial['vectors']
                done_count = int(partial['done_count'])
                print(f'resume: partial.pkl подхвачен, продолжение с {done_count}/{n_items}')
            else:
                print('partial.pkl несовместим с текущими параметрами — извлечение с нуля')
        except Exception as exc:
            print(f'partial.pkl не читается ({type(exc).__name__}: {exc}) — извлечение с нуля')

    meta_chunks = []
    vector_chunks = []
    processed = done_count
    chunks_since_checkpoint = 0
    started = time.perf_counter()
    print(f'=== извлечение эмбеддингов (с {done_count}/{n_items}) ===')
    for offset in range(done_count, n_items, PROGRESS_CHUNK):
        chunk = inputs[offset:offset + PROGRESS_CHUNK]
        chunk_meta, chunk_vectors = extract_embeddings(
            chunk, model, layer_indices, layer_fractions,
            pooling=args.pooling, batch_size=args.batch_size,
            device=args.device, dtype=args.dtype,
        )
        meta_chunks.append(chunk_meta)
        vector_chunks.append(chunk_vectors)
        processed += len(chunk)
        chunks_since_checkpoint += 1
        elapsed = time.perf_counter() - started
        speed = (processed - done_count) / max(elapsed, 1e-9)
        eta_minutes = (n_items - processed) / speed / 60 if speed > 0 else float('inf')
        if chunks_since_checkpoint >= args.checkpoint_every or processed >= n_items:
            _atomic_pickle(partial_path, {
                'params': run_params,
                'done_count': processed,
                'vectors': np.concatenate(
                    ([base_vectors] if base_vectors is not None else []) + vector_chunks,
                    axis=0),
                'meta': pd.concat(
                    ([base_meta] if base_meta is not None else []) + meta_chunks,
                    ignore_index=True),
            })
            chunks_since_checkpoint = 0
            print(f'  чекпоинт: {processed}/{n_items} пунктов '
                  f'({speed:.1f} п/с, ETA {eta_minutes:.1f} мин)')
        else:
            print(f'  обработано {processed}/{n_items} '
                  f'({speed:.1f} п/с, ETA {eta_minutes:.1f} мин)')

    meta = pd.concat(
        ([base_meta] if base_meta is not None else []) + meta_chunks, ignore_index=True)
    vectors = np.concatenate(
        ([base_vectors] if base_vectors is not None else []) + vector_chunks, axis=0)
    assert len(meta) == n_items * len(layer_indices)
    assert vectors.shape[0] == n_items

    answer_kinds_counts = {}
    for item in inputs:
        kind = item['answer_kind']
        answer_kinds_counts[kind] = answer_kinds_counts.get(kind, 0) + 1

    truncated_flags = [bool(item['truncated']) for item in inputs]
    truncation_share = float(np.mean(truncated_flags)) if truncated_flags else 0.0

    lens_summary = {
        'prompt': _percentile_summary([item['len_prompt_tokens'] for item in inputs]),
        'answer': _percentile_summary([item['len_answer_tokens'] for item in inputs]),
        'pooled': _percentile_summary([item['n_pooled'] for item in inputs]),
    }

    manifest = {
        'model': args.model,
        'model_revision': _model_revision(model),
        'input_format': args.input_format,
        'template_notes': TEMPLATE_NOTES,
        'dataset_sha256': HALUEVAL_QA_SHA256,
        'n_records': n_records,
        'n_items': n_items,
        'answer_kinds': answer_kinds_counts,
        'layer_fractions': layer_fractions,
        'layer_indices': layer_indices,
        'pooling': args.pooling,
        'dtype': args.dtype,
        'max_length': args.max_length,
        'truncation_share': truncation_share,
        'prompt_ids_hash': prompt_ids_hash(canonical),
        'lens': lens_summary,
    }

    print(f'=== сохранение кэша в {args.out_dir} ===')
    save_cache(args.out_dir, meta, vectors, manifest)

    print('\n=== ИТОГ ===')
    print(f'n_records={n_records} n_items={n_items} answer_kinds={answer_kinds_counts}')
    print(f'truncation_share={truncation_share:.4f}')
    print(f'vectors.shape={vectors.shape} dtype={args.dtype}')
    for field, stats in lens_summary.items():
        print(f'lens[{field}]: mean={stats["mean"]:.1f} p50={stats["p50"]:.1f} '
              f'p90={stats["p90"]:.1f} max={stats["max"]:.0f}')
    print(f'кэш сохранён: {args.out_dir}')


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
