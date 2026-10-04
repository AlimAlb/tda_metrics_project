"""Аудит парного датасета галлюцинаций HaluEval QA (ветка B, nlp-audit).

Self-contained скрипт для Colab VM (CPU): скачать датасет с пиннанным revision,
вывести фактическую schema, 5 примеров, аудит пар correct/hallucinated
(пустые, дубликаты, длины), проверку схемы prompt_id, кросс-чек HF-зеркала.
Артефакты — в /content/nlp_audit/ (JSON/CSV), лог дублируется Tee-выводом.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

import pandas as pd

LOG_PATH = '/content/nlp_audit_log.txt'
AUDIT_DIR = '/content/nlp_audit'
DATASET_PATH = os.path.join(AUDIT_DIR, 'qa_data.json')

GITHUB_RAW_URL = ('https://raw.githubusercontent.com/RUCAIBox/HaluEval/'
                  'b7253db3cdaa0ab2c382f92b26b390109174f77e/data/qa_data.json')
GITHUB_COMMIT = 'b7253db3cdaa0ab2c382f92b26b390109174f77e'
GITHUB_BLOB_SHA = '981d8944c2a74937a9556d50dd12ffb493d95c27'
HF_MIRROR = 'pminervini/HaluEval'
HF_REVISION = '12a856119f03975a94509091e8cada3e6be6ead7'
HF_PARQUET_URL = (f'https://huggingface.co/datasets/{HF_MIRROR}/resolve/'
                  f'{HF_REVISION}/qa/data-00000-of-00001.parquet')


class Tee:
    def __init__(self, path):
        self.file = open(path, 'w', encoding='utf-8')

    def write(self, text):
        sys.__stdout__.write(text)
        self.file.write(text)

    def flush(self):
        sys.__stdout__.flush()
        self.file.flush()


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def text(x):
    return '' if x is None else str(x)


def snippet(s, n=200):
    s = text(s)
    return s if len(s) <= n else s[:n] + f'...[+{len(s) - n} chars]'


def summary_stats(values):
    s = pd.Series(values)
    return {
        'count': int(s.count()),
        'mean': float(s.mean()),
        'std': float(s.std()),
        'min': float(s.min()),
        'q25': float(s.quantile(0.25)),
        'median': float(s.median()),
        'q75': float(s.quantile(0.75)),
        'max': float(s.max()),
    }


sys.stdout = Tee(LOG_PATH)
os.makedirs(AUDIT_DIR, exist_ok=True)

results = {}
records = None

# ============ Секция 1: окружение ============
try:
    print('=== [1] ENV ===')
    print('python:', sys.version.split()[0])
    print('pandas:', pd.__version__)
    r = run([sys.executable, '-c', 'import numpy; print(numpy.__version__)'])
    print('numpy:', r.stdout.strip() if r.returncode == 0 else 'FAIL')
    r = run([sys.executable, '-c', 'import pyarrow; print(pyarrow.__version__)'])
    print('pyarrow:', r.stdout.strip() if r.returncode == 0 else 'absent')
    results['env'] = 'OK'
except Exception:
    results['env'] = 'FAIL'
    traceback.print_exc(file=sys.stdout)

# ============ Секция 2: скачивание оригинала (GitHub, pinned commit) ============
try:
    print('\n=== [2] DOWNLOAD (GitHub RUCAIBox/HaluEval @ pinned commit) ===')
    t0 = time.time()
    r = run(['curl', '-sSL', '--retry', '2', '-o', DATASET_PATH, GITHUB_RAW_URL])
    print('curl rc:', r.returncode, (r.stderr or '')[:200])
    if r.returncode != 0:
        raise RuntimeError(f'curl failed rc={r.returncode}')
    size = os.path.getsize(DATASET_PATH)
    with open(DATASET_PATH, 'rb') as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    print(f'file: {DATASET_PATH}')
    print(f'size: {size} bytes ({size / 1e6:.2f} MB), sha256: {digest}')
    print(f'source: {GITHUB_RAW_URL}')
    print(f'expected github blob sha: {GITHUB_BLOB_SHA}')
    print(f'download time: {time.time() - t0:.1f} s')
    results['download'] = {'size_bytes': size, 'sha256': digest, 'rc': r.returncode}
except Exception:
    results['download'] = 'FAIL'
    traceback.print_exc(file=sys.stdout)

# ============ Секция 3: schema ============
try:
    print('\n=== [3] SCHEMA ===')
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    try:
        records = json.loads(raw_text)
        records_format = 'json-array'
    except json.JSONDecodeError:
        records = []
        for line in raw_text.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        records_format = 'jsonl (одна JSON-запись на строку; расширение .json вводит в заблуждение)'
    del raw_text
    print('формат файла:', records_format, '| n_records:', len(records))
    keys_union = []
    for rec in records:
        for k in rec:
            if k not in keys_union:
                keys_union.append(k)
    print('union of keys across all records:', keys_union)
    print('keys of record[0]:', list(records[0].keys()))
    df = pd.DataFrame(records)
    print('columns:', list(df.columns))
    print('dtypes:\n', df.dtypes.to_string())
    print('shape:', df.shape)
    schema = {
        'source': GITHUB_RAW_URL,
        'github_commit': GITHUB_COMMIT,
        'github_blob_sha': GITHUB_BLOB_SHA,
        'file_format': records_format,
        'n_records': len(records),
        'keys_union': keys_union,
        'columns': list(df.columns),
        'dtypes': {c: str(df[c].dtype) for c in df.columns},
    }
    with open(os.path.join(AUDIT_DIR, 'schema.json'), 'w', encoding='utf-8') as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)
    print('5 примеров (поля усечены до 200 символов):')
    examples = []
    for i in range(5):
        rec = records[i]
        examples.append(rec)
        print(f'--- пример {i} ---')
        for k, v in rec.items():
            print(f'  {k}: {snippet(v)}')
    with open(os.path.join(AUDIT_DIR, 'examples.json'), 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    results['schema'] = 'OK'
except Exception:
    results['schema'] = 'FAIL'
    traceback.print_exc(file=sys.stdout)

# ============ Секция 4: аудит пар ============
try:
    print('\n=== [4] PAIRS AUDIT ===')
    n = len(records)
    fields = keys_union if records else []

    def field_of(rec, name):
        return text(rec.get(name, '')).strip()

    empty_counts = {k: sum(1 for rec in records if field_of(rec, k) == '') for k in fields}
    none_counts = {k: sum(1 for rec in records if rec.get(k) is None) for k in fields}
    print('пустые (strip == "") по полям:', empty_counts)
    print('None по полям:', none_counts)

    has_correct = [i for i, rec in enumerate(records) if field_of(rec, 'right_answer') != '']
    has_hallucinated = [i for i, rec in enumerate(records) if field_of(rec, 'hallucinated_answer') != '']
    has_both = [i for i, rec in enumerate(records)
                if field_of(rec, 'right_answer') != '' and field_of(rec, 'hallucinated_answer') != '']
    has_question = [i for i, rec in enumerate(records) if field_of(rec, 'question') != '']
    print(f'строк с непустым question: {len(has_question)}/{n}')
    print(f'строк с непустым right_answer: {len(has_correct)}/{n}')
    print(f'строк с непустым hallucinated_answer: {len(has_hallucinated)}/{n}')
    print(f'строк с ОБЕИМИ ответами (пар): {len(has_both)}/{n} = {len(has_both) / n:.4%}')

    same_text = [i for i in range(n)
                if field_of(records[i], 'right_answer') == field_of(records[i], 'hallucinated_answer')
                and field_of(records[i], 'right_answer') != '']
    print(f'строк, где right_answer == hallucinated_answer (непустой): {len(same_text)}')

    def dup_stats(get_key):
        seen, dups = {}, {}
        for i in range(n):
            key = get_key(records[i])
            if key in seen:
                dups.setdefault(key, [seen[key]]).append(i)
            else:
                seen[key] = i
        return dups

    dup_full = dup_stats(lambda r: tuple(field_of(r, k) for k in fields))
    dup_prompt = dup_stats(lambda r: (field_of(r, 'knowledge'), field_of(r, 'question')))
    dup_question = dup_stats(lambda r: field_of(r, 'question'))
    dup_right = dup_stats(lambda r: field_of(r, 'right_answer'))
    dup_hall = dup_stats(lambda r: field_of(r, 'hallucinated_answer'))

    print(f'полных дублирующихся строк (все поля): {len(dup_full)} ключей, '
          f'{sum(len(v) for v in dup_full.values())} лишних вхождений')
    print(f'дубликаты пары (knowledge, question): {len(dup_prompt)} ключей, '
          f'{sum(len(v) for v in dup_prompt.values())} лишних вхождений')
    print(f'дубликаты только question: {len(dup_question)} ключей')
    print(f'дубликаты right_answer: {len(dup_right)} ключей')
    print(f'дубликаты hallucinated_answer: {len(dup_hall)} ключей')

    top_prompt_dups = sorted(dup_prompt.items(), key=lambda kv: -len(kv[1]))[:10]
    print('топ дубликатов (knowledge, question):')
    for (k, q), idxs in top_prompt_dups:
        print(f'  x{len(idxs)} | question: {snippet(q, 100)}')
    top_right_dups = sorted(dup_right.items(), key=lambda kv: -len(kv[1]))[:5]
    print('топ дубликатов right_answer:')
    for ans, idxs in top_right_dups:
        print(f'  x{len(idxs)} | {snippet(ans, 100)}')

    pairs_audit = {
        'n_records': n,
        'empty_counts': empty_counts,
        'none_counts': none_counts,
        'n_with_question': len(has_question),
        'n_with_correct': len(has_correct),
        'n_with_hallucinated': len(has_hallucinated),
        'n_with_both_answers': len(has_both),
        'share_with_both_answers': len(has_both) / n,
        'n_right_equals_hallucinated_nonempty': len(same_text),
        'n_duplicate_full_rows_keys': len(dup_full),
        'n_duplicate_extra_full_rows': sum(len(v) - 1 for v in dup_full.values()),
        'n_duplicate_prompt_pairs_keys': len(dup_prompt),
        'n_duplicate_prompt_extra_rows': sum(len(v) - 1 for v in dup_prompt.values()),
        'n_duplicate_question_keys': len(dup_question),
        'n_duplicate_right_answer_keys': len(dup_right),
        'n_duplicate_hallucinated_answer_keys': len(dup_hall),
        'top_prompt_duplicates': [
            {'knowledge': k, 'question': q, 'row_indices': v[:20], 'n_occurrences': len(v)}
            for (k, q), v in top_prompt_dups
        ],
        'top_right_answer_duplicates': [
            {'answer': a, 'row_indices': v[:20], 'n_occurrences': len(v)}
            for a, v in top_right_dups
        ],
    }
    with open(os.path.join(AUDIT_DIR, 'pairs_audit.json'), 'w', encoding='utf-8') as f:
        json.dump(pairs_audit, f, indent=2, ensure_ascii=False)
    results['pairs_audit'] = 'OK'
except Exception:
    results['pairs_audit'] = 'FAIL'
    traceback.print_exc(file=sys.stdout)

# ============ Секция 5: длины correct vs hallucinated ============
try:
    print('\n=== [5] LENGTHS (chars / words) ===')
    lens = {
        'question_chars': [len(field_of(r, 'question')) for r in records],
        'question_words': [len(field_of(r, 'question').split()) for r in records],
        'knowledge_chars': [len(field_of(r, 'knowledge')) for r in records],
        'knowledge_words': [len(field_of(r, 'knowledge').split()) for r in records],
        'correct_chars': [len(field_of(r, 'right_answer')) for r in records],
        'correct_words': [len(field_of(r, 'right_answer').split()) for r in records],
        'hallucinated_chars': [len(field_of(r, 'hallucinated_answer')) for r in records],
        'hallucinated_words': [len(field_of(r, 'hallucinated_answer').split()) for r in records],
    }
    for name, values in lens.items():
        st = summary_stats(values)
        print(f'{name}: mean={st["mean"]:.2f} std={st["std"]:.2f} median={st["median"]:.1f} '
              f'q25={st["q25"]:.1f} q75={st["q75"]:.1f} min={st["min"]:.0f} max={st["max"]:.0f}')

    diff_chars = [h - c for h, c in zip(lens['hallucinated_chars'], lens['correct_chars'])]
    diff_words = [h - c for h, c in zip(lens['hallucinated_words'], lens['correct_words'])]
    st_dc, st_dw = summary_stats(diff_chars), summary_stats(diff_words)
    print(f'diff chars (hallucinated - correct): mean={st_dc["mean"]:.2f} std={st_dc["std"]:.2f} '
          f'median={st_dc["median"]:.2f} q25={st_dc["q25"]:.2f} q75={st_dc["q75"]:.2f}')
    print(f'diff words (hallucinated - correct): mean={st_dw["mean"]:.2f} std={st_dw["std"]:.2f} '
          f'median={st_dw["median"]:.2f} q25={st_dw["q25"]:.2f} q75={st_dw["q75"]:.2f}')
    n_shorter_h = sum(1 for d in diff_chars if d < 0)
    n_longer_h = sum(1 for d in diff_chars if d > 0)
    n_equal_h = sum(1 for d in diff_chars if d == 0)
    print(f'hallucinated короче/длиннее/равно по символам: {n_shorter_h}/{n_longer_h}/{n_equal_h}')
    big_diff = sum(1 for d in diff_chars if abs(d) > 50)
    print(f'пар с |diff chars| > 50: {big_diff} ({big_diff / n:.2%})')

    lengths_summary = {name: summary_stats(values) for name, values in lens.items()}
    lengths_summary['diff_chars'] = st_dc
    lengths_summary['diff_words'] = st_dw
    lengths_summary['n_hallucinated_shorter_chars'] = n_shorter_h
    lengths_summary['n_hallucinated_longer_chars'] = n_longer_h
    lengths_summary['n_equal_chars'] = n_equal_h
    lengths_summary['n_abs_diff_chars_gt_50'] = big_diff
    with open(os.path.join(AUDIT_DIR, 'lengths_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(lengths_summary, f, indent=2, ensure_ascii=False)
    pd.DataFrame({
        'question_chars': lens['question_chars'],
        'question_words': lens['question_words'],
        'knowledge_chars': lens['knowledge_chars'],
        'knowledge_words': lens['knowledge_words'],
        'correct_chars': lens['correct_chars'],
        'correct_words': lens['correct_words'],
        'hallucinated_chars': lens['hallucinated_chars'],
        'hallucinated_words': lens['hallucinated_words'],
    }).to_csv(os.path.join(AUDIT_DIR, 'lengths_per_row.csv'), index=False)
    print('per-row длины сохранены: lengths_per_row.csv')
    results['lengths'] = 'OK'
except Exception:
    results['lengths'] = 'FAIL'
    traceback.print_exc(file=sys.stdout)

# ============ Секция 6: категории ============
try:
    print('\n=== [6] CATEGORIES ===')
    expected = {'knowledge', 'question', 'right_answer', 'hallucinated_answer'}
    extra_fields = [k for k in keys_union if k not in expected]
    print('полей сверх ожидаемых четырёх:', extra_fields or 'нет')
    category_like = [k for k in keys_union if 'categ' in k.lower() or 'type' in k.lower() or 'label' in k.lower()]
    print('похожие на категорию поля:', category_like or 'нет')
    results['categories'] = {
        'extra_fields': extra_fields,
        'category_like_fields': category_like,
        'verdict': 'колонки категории в HaluEval qa_data нет; в контракте category = константа "qa"',
    }
    print('вывод: колонки категории нет; в контракте category = константа "qa"')
except Exception:
    results['categories'] = 'FAIL'
    traceback.print_exc(file=sys.stdout)

# ============ Секция 7: предложение prompt_id ============
try:
    print('\n=== [7] PROMPT_ID PROPOSAL ===')
    def make_pid(rec):
        payload = field_of(rec, 'knowledge') + '\x1f' + field_of(rec, 'question')
        return hashlib.sha1(payload.encode('utf-8')).hexdigest()

    pids = [make_pid(r) for r in records]
    n_unique_pids = len(set(pids))
    n_unique_prompts = len(set((field_of(r, 'knowledge'), field_of(r, 'question')) for r in records))
    n_unique_questions = len(set(field_of(r, 'question') for r in records))
    print(f'уникальных question: {n_unique_questions}/{n}')
    print(f'уникальных пар (knowledge, question): {n_unique_prompts}/{n}')
    print(f'уникальных sha1-prompt_id: {n_unique_pids}/{n}')
    print(f'консистентность: prompt_id уникален <=> пара (knowledge, question) уникальна: '
          f'{n_unique_pids == n_unique_prompts}')
    print('примеры prompt_id:', pids[:3])
    proposal = {
        'scheme': 'sha1(knowledge + "\\x1f" + question) hex',
        'n_unique_questions': n_unique_questions,
        'n_unique_prompt_pairs': n_unique_prompts,
        'n_unique_prompt_ids': n_unique_pids,
        'consistent': n_unique_pids == n_unique_prompts,
        'note': ('индекс строки не годится как prompt_id при наличии дубликатов prompt-объектов; '
                 'hash от (knowledge, question) идентифицирует один и тот же prompt между строками'),
    }
    with open(os.path.join(AUDIT_DIR, 'prompt_id_check.json'), 'w', encoding='utf-8') as f:
        json.dump(proposal, f, indent=2, ensure_ascii=False)
    results['prompt_id'] = 'OK'
except Exception:
    results['prompt_id'] = 'FAIL'
    traceback.print_exc(file=sys.stdout)

# ============ Секция 8: кросс-чек HF-зеркала pminervini/HaluEval ============
try:
    print('\n=== [8] HF MIRROR CROSS-CHECK ===')
    mirror = pd.read_parquet(HF_PARQUET_URL)
    print('зеркало загружено:', HF_MIRROR, '@', HF_REVISION)
    print('shape зеркала:', mirror.shape, '| колонки:', list(mirror.columns))
    gh_frame = pd.DataFrame(records) if records is not None else None
    same_shape = gh_frame is not None and mirror.shape == gh_frame.shape
    same_cols = gh_frame is not None and list(mirror.columns) == list(gh_frame.columns)
    print('shape совпадает:', same_shape, '| колонки совпадают:', same_cols)
    content_equal = False
    if same_shape and same_cols:
        a = gh_frame.sort_values(list(gh_frame.columns)).reset_index(drop=True)
        b = mirror.sort_values(list(mirror.columns)).reset_index(drop=True)
        content_equal = a.equals(b)
        if not content_equal:
            neq = (a.astype(str) != b.astype(str)).any(axis=1).sum()
            print(f'расхождений строк: {neq}')
    print('контент GitHub vs HF-зеркало полностью равен:', content_equal)
    crosscheck = {
        'mirror': HF_MIRROR,
        'revision': HF_REVISION,
        'parquet_url': HF_PARQUET_URL,
        'mirror_shape': list(mirror.shape),
        'same_shape': same_shape,
        'same_columns': same_cols,
        'content_equal': bool(content_equal),
    }
    with open(os.path.join(AUDIT_DIR, 'mirror_crosscheck.json'), 'w', encoding='utf-8') as f:
        json.dump(crosscheck, f, indent=2, ensure_ascii=False)
    results['mirror_crosscheck'] = 'OK'
except Exception:
    results['mirror_crosscheck'] = 'FAIL'
    traceback.print_exc(file=sys.stdout)

# ============ Секция 9: итог ============
try:
    print('\n=== [9] ARTIFACTS ===')
    r = run(['ls', '-la', AUDIT_DIR])
    print(r.stdout)
    results['artifacts'] = 'OK'
except Exception:
    results['artifacts'] = 'FAIL'
    traceback.print_exc(file=sys.stdout)

print('\n=== ИТОГ АУДИТА ===')
for key, value in results.items():
    print(f'  {key}: {value}')
sys.stdout.flush()
