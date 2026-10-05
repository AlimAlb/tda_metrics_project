"""Supervisor ночного/дневного цикла прогонов синтетики (оркестраторский инструмент).

Причина (appendix §23.1 — конкретная повторяющаяся проблема): сессии Colab
умирают каждые ~40-45 минут (принудительный таймаут GPU free-tier при
интенсивном использовании) и 4 раза теряли диск при живом BUSY-ядре.
Supervisor выполняет pre-emptive миграцию ДО таймаута (чекпоинт -> stop ->
new -> lite-setup -> restore -> resume) и реактивное восстановление при
ранней смерти, пока счётчик строк store не достигнет цели.

Работает ЛОКАЛЬНО (subprocess вокруг colab CLI), лог: /tmp/opencode/supervisor_state.log.
Останавливается сам при достижении цели (финальные артефакты скачивает).
"""
import json
import os
import subprocess
import sys
import time

SSH_SOCK = os.popen("ls ~/.ssh/agent/s.* 2>/dev/null | head -1").read().strip() or '/run/user/1000/openssh_agent'
ENV = dict(os.environ, SSH_AUTH_SOCK=SSH_SOCK)
COLAB = os.path.expanduser('~/.local/bin/colab')

TARGET_ROWS = 625
STORE_PARQUET = '/content/results/raw/synthetic/results.parquet'
STORE_MANIFEST = '/content/results/raw/synthetic/manifest.json'
LOCAL_STATE_DIR = '/tmp/opencode/supervisor_state'
LOG_PATH = '/tmp/opencode/supervisor_state.log'

SESSION_LIFETIME_MIN = 240
POLL_INTERVAL_MIN = 5
SNAPSHOT_EVERY_POLLS = 2
MAX_RETRIES_PER_STEP = 3
STALL_MIN = 25

os.makedirs(LOCAL_STATE_DIR, exist_ok=True)


def log(message):
    line = f'[{time.strftime("%H:%M:%S")}] {message}'
    print(line, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')


def colab(args, timeout=120, check=False):
    cmd = [COLAB, '--auth=oauth2'] + args
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=ENV)
    except subprocess.TimeoutExpired:
        return None
    if check and (r is None or r.returncode != 0):
        raise RuntimeError(f'colab {" ".join(args)} -> rc={r.returncode if r else "TIMEOUT"}\n{r.stderr if r else ""}')
    return r


def session_alive():
    r = colab(['ls', '/content'], timeout=60)
    if r is None:
        return False
    return 'not found' not in (r.stdout + r.stderr).lower() and r.returncode == 0


def manifest_rows():
    r = colab(['download', STORE_MANIFEST, f'{LOCAL_STATE_DIR}/manifest_probe.json'], timeout=90)
    if r is None or r.returncode != 0:
        return None
    try:
        return json.load(open(f'{LOCAL_STATE_DIR}/manifest_probe.json'))['n_rows']
    except Exception:
        return None


def download_snapshot(tag):
    colab(['download', STORE_PARQUET, f'{LOCAL_STATE_DIR}/snap_{tag}.parquet'], timeout=180)
    colab(['download', STORE_MANIFEST, f'{LOCAL_STATE_DIR}/snap_{tag}_manifest.json'], timeout=90)
    log(f'snapshot {tag} скачан')


def terminate_all_sessions():
    r = colab(['sessions'], timeout=60)
    names = []
    if r:
        for line in (r.stdout or '').splitlines():
            line = line.strip()
            if line.startswith('[') and ']' in line:
                names.append(line[1:line.index(']')])
    for name in names:
        colab(['stop', '--session', name], timeout=90)
        log(f'сессия {name} остановлена')


def create_session():
    for attempt in range(MAX_RETRIES_PER_STEP):
        r = colab(['new', '--gpu', 'L4'], timeout=180)
        if r is not None and r.returncode == 0:
            log('сессия создана')
            return True
        time.sleep(60)
    return False


def upload_files():
    for attempt in range(MAX_RETRIES_PER_STEP):
        ok = True
        for src, dst in (
            ('/tmp/opencode/tda_metrics_src.zip', '/content/tda_metrics_src.zip'),
            (f'{LOCAL_STATE_DIR}/latest_snapshot.parquet', '/content/snap.parquet'),
        ):
            r = colab(['upload', src, dst], timeout=300)
            if r is None or r.returncode != 0:
                ok = False
        if ok:
            log('upload zip + snapshot OK')
            return True
        time.sleep(45)
    return False


def run_exec(script_local, timeout_sec, label):
    for attempt in range(MAX_RETRIES_PER_STEP):
        r = colab(['exec', '-f', script_local, '--timeout', str(timeout_sec)], timeout=timeout_sec + 300)
        if r is not None and r.returncode == 0 and 'Connection was lost' not in (r.stderr or ''):
            log(f'exec {label}: клиент завершился штатно')
            return True
        log(f'exec {label}: сбой клиента (попытка {attempt + 1}); restart-kernel + retry')
        colab(['restart-kernel'], timeout=120)
        time.sleep(30)
    return False


def spawn_exec(script_local, timeout_sec, label):
    cmd = [COLAB, '--auth=oauth2', 'exec', '-f', script_local, '--timeout', str(timeout_sec)]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=ENV)
    log(f'exec {label} запущен неблокирующе (pid {proc.pid})')
    return proc


def wait_for_idle(timeout_min=40):
    """Ждать освобождения ядра (BUSY- exec в очереди), опрос каждые 60 с."""
    started = time.time()
    while (time.time() - started) / 60 < timeout_min:
        r = colab(['status'], timeout=60)
        text = (r.stdout + r.stderr) if r else ''
        if 'IDLE' in text:
            return True
        time.sleep(60)
    log(f'ядро не освободилось за {timeout_min} мин')
    return False


def main():
    log('=== SUPERVISOR: старт ===')
    poll_count = 0
    session_started = None
    migration_number = 0
    synth_proc = None
    last_rows = None
    last_change = time.time()
    deadline = time.time() + 6 * 3600

    latest = f'{LOCAL_STATE_DIR}/latest_snapshot.parquet'
    if not os.path.exists(latest):
        subprocess.run(['cp', '/tmp/opencode/syn_busy10.parquet', latest], check=True)
        log('стартовый snapshot: syn_busy10 (444 строки)')

    def start_fresh_cycle(reason):
        nonlocal migration_number, session_started, poll_count, synth_proc
        log(f'--- новый цикл: {reason} ---')
        if synth_proc is not None and synth_proc.poll() is None:
            synth_proc.kill()
            log('старый synth-run клиент убит')
        synth_proc = None
        terminate_all_sessions()
        if not create_session():
            log('НЕ УДАЛОСЬ СОЗДАТЬ СЕССИЮ — пауза 10 мин')
            time.sleep(600)
            return False
        if not upload_files():
            log('upload не прошёл — пауза 5 мин')
            time.sleep(300)
            return False
        if not run_exec('logs/orchestration_wave1/wave1b_setup_lite.py', 1500, 'lite-setup'):
            log('lite-setup не прошёл — пауза 5 мин')
            time.sleep(300)
            return False
        if not run_exec('/tmp/opencode/wave1b_restore2.py', 180, 'restore'):
            log('restore не прошёл — пауза 5 мин')
            time.sleep(300)
            return False
        colab(['upload', '/tmp/opencode/wave1b_sections.txt', '/content/wave1b_sections.txt'], timeout=90)
        synth_proc = spawn_exec('logs/orchestration_wave1/wave1b_synth_run.py', 5400, 'synth-run')
        migration_number += 1
        session_started = time.time()
        poll_count = 0
        return True

    while time.time() < deadline:
        rows = manifest_rows()
        log(f'строк в store: {rows}')
        if rows is not None and rows >= TARGET_ROWS:
            log('ЦЕЛЬ ДОСТИГНУТА — финальные артефакты')
            download_snapshot('final')
            colab(['download', '/content/wave1b_synth_log.txt', f'{LOCAL_STATE_DIR}/final_synth_log.txt'], timeout=180)
            colab(['download', '/content/results/raw/synthetic/synthetic_summary.csv', f'{LOCAL_STATE_DIR}/synthetic_summary.csv'], timeout=180)
            terminate_all_sessions()
            log('=== SUPERVISOR: ЗАВЕРШЁН ===')
            return

        session_ok = session_alive()
        if not session_ok:
            log('сессия мертва/недоступна')
            start_fresh_cycle('смерть сессии')
            continue

        if session_started is None:
            session_started = time.time()

        if synth_proc is None and (rows is None):
            log('сессия жива, но store/прогон не поднят — ожидаю ядро и поднимаю')
            if not wait_for_idle(timeout_min=40):
                start_fresh_cycle('ядро не освободилось')
                continue
            if run_exec('/tmp/opencode/wave1b_restore2.py', 180, 'restore'):
                if manifest_rows() is not None:
                    colab(['upload', '/tmp/opencode/wave1b_sections.txt', '/content/wave1b_sections.txt'], timeout=90)
                    synth_proc = spawn_exec('logs/orchestration_wave1/wave1b_synth_run.py', 5400, 'synth-run')
                    session_started = time.time()
                    last_rows = manifest_rows()
                    last_change = time.time()
                    poll_count = 0
                    continue
            log('restore/store не поднялся — полный цикл (setup внутри)')
            start_fresh_cycle('store не поднялся')
            continue

        if synth_proc is None and rows is not None and rows < TARGET_ROWS:
            if rows != last_rows:
                last_rows = rows
                last_change = time.time()
            elif (time.time() - last_change) / 60 >= STALL_MIN:
                log(f'застоя {STALL_MIN} мин на {rows} строках — resume-перезапуск прогона')
                if not wait_for_idle(timeout_min=15):
                    start_fresh_cycle('ядро не освободилось после застоя')
                    continue
                colab(['upload', '/tmp/opencode/wave1b_sections.txt', '/content/wave1b_sections.txt'], timeout=90)
                synth_proc = spawn_exec('logs/orchestration_wave1/wave1b_synth_run.py', 5400, 'synth-run-resume')
                last_change = time.time()
                session_started = time.time()
                continue

        age_min = (time.time() - session_started) / 60
        if age_min >= SESSION_LIFETIME_MIN:
            log(f'возраст сессии {age_min:.0f} мин — pre-emptive миграция')
            tag = f'pre{migration_number}'
            download_snapshot(tag)
            subprocess.run(['cp', f'{LOCAL_STATE_DIR}/snap_{tag}.parquet', latest], check=True)
            start_fresh_cycle('pre-emptive 38 мин')
            continue

        poll_count += 1
        if poll_count % SNAPSHOT_EVERY_POLLS == 0:
            tag = f'p{migration_number}_{poll_count}'
            colab(['download', STORE_PARQUET, f'{LOCAL_STATE_DIR}/snap_{tag}.parquet'], timeout=180)
            manifest_rows()
            subprocess.run(['cp', f'{LOCAL_STATE_DIR}/snap_{tag}.parquet', latest], check=True)
            log(f'обновлён latest_snapshot ({tag})')
        time.sleep(POLL_INTERVAL_MIN * 60)

    log('БЮДЖЕТ SUPERVISOR (6 ч) ИСЧИХАН — скачиваю что есть')
    download_snapshot('budget_end')
    terminate_all_sessions()
    log('=== SUPERVISOR: СТОП ПО БЮДЖЕТУ ===')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        log('SUPERVISOR УПАЛ:\n' + traceback.format_exc())
        sys.exit(1)