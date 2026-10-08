"""Генерация ответов Wave ST: correct, локально-ложная правка (wrong), парафраз.

Оба источника (A = Qwen2.5-3B-Instruct, B = Phi-3.5-mini) дают на каждый промпт
три ответа: ``correct`` — ответ строго по knowledge; ``wrong`` — минимальная
правка correct (подмена одной сущности/даты/числа либо неподтверждённое
утверждение) с сохранением длины и стиля; ``paraphrase`` — парафраз correct с
сохранением смысла и длины (негативный контроль стиля). Декодирование greedy
(do_sample=False) — полная детерминированность без сэмплов; все тексты
промптов фиксируются протоколом дословно. Автоверификатор — тот же читатель
(Qwen2.5-3B): «поддерживается/противоречит knowledge» — фильтр кандидатов
при генерации, НЕ ground truth: финальные метки даёт ручная разметка
(пользователь 100% + второй разметчик 10% по рубрике).
"""
import re

__all__ = [
    'ANSWER_SYSTEM_PROMPT',
    'VERIFY_SYSTEM_PROMPT',
    'correct_user_prompt',
    'wrong_edit_user_prompt',
    'paraphrase_user_prompt',
    'verify_user_prompt',
    'build_messages',
    'greedy_generate',
    'parse_verdict',
    'verify_supported',
    'verify_contradicted',
    'length_stats',
    'length_compatible',
    'changed_word_fraction',
    'edit_is_minimal',
]

ANSWER_SYSTEM_PROMPT = (
    'You are a precise reading assistant. Answer the question using ONLY the '
    'provided knowledge. Reply with one or two short sentences. Do not add '
    'anything else.'
)
VERIFY_SYSTEM_PROMPT = (
    'You are a strict fact checker. Reply with exactly one word.'
)
_MAX_ANSWER_WORDS = 60
_LENGTH_MIN_RATIO = 0.6
_LENGTH_MAX_RATIO = 1.6
_MAX_WORD_DIFF = 12
_MAX_EDIT_FRACTION = 0.4


def correct_user_prompt(question, knowledge):
    return f'Knowledge: {knowledge}\nQuestion: {question}'


def wrong_edit_user_prompt(question, knowledge, correct_answer):
    return (
        f'Knowledge: {knowledge}\nQuestion: {question}\n'
        f'Your answer: {correct_answer}\n\n'
        'Task: minimally edit your answer so that it becomes factually WRONG '
        'with respect to the knowledge: change exactly one entity, date, or '
        'number, or state one claim the knowledge does not support. Keep the '
        'same length, wording and style. Reply only with the edited answer.'
    )


def paraphrase_user_prompt(question, knowledge, correct_answer):
    return (
        f'Knowledge: {knowledge}\nQuestion: {question}\n'
        f'Your answer: {correct_answer}\n\n'
        'Task: rewrite your answer with different wording while keeping '
        'exactly the same meaning and approximately the same length. Reply '
        'only with the rewritten answer.'
    )


def verify_user_prompt(knowledge, answer, mode):
    """Формулировка вердикта: mode='support' — подтверждён ли ответ knowledge;
    mode='contradict' — противоречит/не подтверждён ли ответ knowledge."""
    if mode not in ('support', 'contradict'):
        raise ValueError(f'неизвестный mode верификатора: {mode!r}')
    question = (
        'Does the knowledge fully support this answer?'
        if mode == 'support' else
        'Is this answer factually contradicted by or unsupported in the knowledge?'
    )
    return f'Knowledge: {knowledge}\nAnswer: {answer}\n{question}'


def build_messages(system, user):
    """Чат-сообщения для apply_chat_template."""
    return [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': user},
    ]


def greedy_generate(model, tokenizer, messages, max_new_tokens, device):
    """Greedy-генерация (do_sample=False) с chat-шаблоном; детерминирована.

    Возвращает декодированный текст без спец-токенов, stripped. Требует
    torch и HF-совместимый интерфейс (apply_chat_template, generate).
    """
    import torch

    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    generated = output[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def parse_verdict(text):
    """Первое yes/no/unsure в ответе верификатора (регистронезависимо)."""
    match = re.search(r'\b(yes|no|unsure)\b', str(text).lower())
    return match.group(1) if match else 'unsure'


def verify_supported(model, tokenizer, knowledge, answer, device):
    """Вердикт «поддерживается ли ответ knowledge»: yes/no/unsure."""
    messages = build_messages(
        VERIFY_SYSTEM_PROMPT,
        verify_user_prompt(knowledge, answer, 'support'))
    return parse_verdict(greedy_generate(model, tokenizer, messages, 8, device))


def verify_contradicted(model, tokenizer, knowledge, answer, device):
    """Вердикт «противоречит/не подтверждён ли ответ knowledge»: yes/no/unsure."""
    messages = build_messages(
        VERIFY_SYSTEM_PROMPT,
        verify_user_prompt(knowledge, answer, 'contradict'))
    return parse_verdict(greedy_generate(model, tokenizer, messages, 8, device))


def length_stats(correct, edited):
    """Словарь длин (слова) пары correct/edited и их отношения."""
    n_correct = len(str(correct).split())
    n_edited = len(str(edited).split())
    ratio = n_edited / n_correct if n_correct else float('inf')
    return {
        'n_words_correct': n_correct,
        'n_words_edited': n_edited,
        'word_diff': abs(n_edited - n_correct),
        'ratio': ratio,
    }


def length_compatible(correct, edited):
    """Проверка сопоставимости длин правки/парафраза с correct (протокол)."""
    stats = length_stats(correct, edited)
    if stats['n_words_correct'] == 0 or stats['n_words_edited'] == 0:
        return False, stats
    ok = (stats['ratio'] >= _LENGTH_MIN_RATIO and stats['ratio'] <= _LENGTH_MAX_RATIO
          and stats['word_diff'] <= _MAX_WORD_DIFF)
    return ok, stats


def changed_word_fraction(correct, edited):
    """Доля изменённых слов (bag-of-words) между correct и edited."""
    correct_words = str(correct).lower().split()
    edited_words = str(edited).lower().split()
    if not correct_words or not edited_words:
        return 1.0
    from collections import Counter
    bag_correct = Counter(correct_words)
    bag_edited = Counter(edited_words)
    changed = sum((bag_correct - bag_edited).values()) + \
        sum((bag_edited - bag_correct).values())
    return min(1.0, changed / max(len(correct_words), len(edited_words)))


def edit_is_minimal(correct, edited):
    """Правка минимальна: изменено не более 40% слов (bag-of-words)."""
    return changed_word_fraction(correct, edited) <= _MAX_EDIT_FRACTION
