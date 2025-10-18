"""
S15 瑞士轮后期模拟器（通用版）
- 任意 N_SIM、任意 N_THREAD（含 1）
- 线程安全计数
- Bo3 只记最终胜负
- 已晋级/已淘汰不再出战
"""
import random, math, threading, time, logging
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

random.seed(42)
N_SIM = 50000 # 模拟轮数
N_THREAD = 4  # 线程数
LOG_FILE = 'swiss_sim.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')]
)
logger = logging.getLogger()

# ---------- 初始数据 ----------
RECORD = {
    'GEN': (2, 1), 'KT': (3, 0), 'AL': (3, 0), 'TES': (2, 1),
    'HLE': (2, 1), 'CFO': (2, 1), 'BLG': (1, 2), 'T1': (1, 2),
    '100T': (1, 2), 'TSW': (1, 2), 'FNC': (0, 2), 'MKOI': (0, 2),
    'PSG': (0, 2), 'VKS': (0, 2), 'FLY': (2, 1), 'G2': (2, 1)
}

POWER = {
    'GEN': 90, 'BLG': 87, 'TES': 83, 'T1': 89, 'HLE': 87,
    'AL': 86, 'KT': 84, 'CFO': 85, 'G2': 82, 'FLY': 80,
    '100T': 72, 'FNC': 75, 'TSW': 69, 'VKS': 66, 'PSG': 65, 'MKOI': 62
}

PLAYED = {
    ('tsw', 'vks'), ('cfo', 'fnc'), ('kt', 'mkoi'), ('blg', '100t'),
    ('fly', 'hle'), ('al', 'g2'), ('tes', 'gen'), ('gen', 'psg'),
    ('vks', 'fly'), ('kt', 'tsw'), ('mkoi', 'g2'), ('tes', '100t'),
    ('cfo', 'al'), ('gen', 'al'), ('blg', 'fnc'), ('hle', 'psg'),
    ('kt', 'tes'), ('tsw', 'fly'), ('gen', 't1'), ('g2', 'blg'), ('100t', 'hle')
}

ALL_TeAMS = list(RECORD.keys())

# ---------- 工具 ----------
def pwin(a, b, bo3=False):
    diff = POWER[b] - POWER[a]
    scale = 600 if bo3 else 400
    return 1 / (1 + 10 ** (diff / scale))

def can_play(t1, t2, played_set):
    return (t1.lower(), t2.lower()) not in played_set and (t2.lower(), t1.lower()) not in played_set

def draw_pairs(teams, played_set):
    teams = teams.copy()
    random.shuffle(teams)
    pairs, used = [], set()
    for t in teams:
        if t in used:
            continue
        for opp in teams:
            if opp == t or opp in used:
                continue
            if can_play(t, opp, played_set):
                pairs.append((t, opp))
                used.add(t)
                used.add(opp)
                break
        else:
            for opp in teams:
                if opp != t and opp not in used:
                    pairs.append((t, opp))
                    used.add(t)
                    used.add(opp)
                    break
    return pairs

# ---------- 单轮模拟 ----------
def simulate_one(sim_id):
    rec = {t: list(RECORD[t]) for t in ALL_TeAMS}
    played_set = PLAYED.copy()
    round_cnt = 3

    while True:
        alive = {t for t in ALL_TeAMS if rec[t][0] < 3 and rec[t][1] < 3}
        max_g = max(w + l for w, l in rec.values())
        incomplete = [t for t in alive if sum(rec[t]) < max_g]

        if incomplete:
            buckets = defaultdict(list)
            for t in incomplete:
                w, l = rec[t]
                buckets[w, l].append(t)
            for key in buckets:
                is_bo3 = key in [(0, 2), (1, 2), (2, 1), (2, 2)]
                pairs = draw_pairs(buckets[key], played_set)
                for a, b in pairs:
                    pa = pwin(a, b, bo3=is_bo3)
                    if random.random() < pa:
                        rec[a][0] += 1
                        rec[b][1] += 1
                        logger.info(f"[Sim {sim_id}] [Round {round_cnt}] [Bo3] {a} > {b}  -> {a}({rec[a][0]}-{rec[a][1]}) {b}({rec[b][0]}-{rec[b][1]})")
                    else:
                        rec[a][1] += 1
                        rec[b][0] += 1
                        logger.info(f"[Sim {sim_id}] [Round {round_cnt}] [Bo3] {b} > {a}  -> {a}({rec[a][0]}-{rec[a][1]}) {b}({rec[b][0]}-{rec[b][1]})")
                    played_set.add((a.lower(), b.lower()))
            continue

        advanced = {t for t in ALL_TeAMS if rec[t][0] == 3}
        eliminated = {t for t in ALL_TeAMS if rec[t][1] == 3}
        if len(advanced) >= 8 or (len(advanced) + len({t for t in alive if rec[t] == (2, 2)}) == 8):
            return advanced | {t for t in alive if rec[t][0] == 3}

        buckets = defaultdict(list)
        for t in alive:
            w, l = rec[t]
            buckets[w, l].append(t)
        round_cnt += 1

        next_matches = []
        for key in buckets:
            is_bo3 = key in [(0, 2), (1, 2), (2, 1), (2, 2)]
            pairs = draw_pairs(buckets[key], played_set)
            for a, b in pairs:
                next_matches.append((a, b, is_bo3))

        for a, b, is_bo3 in next_matches:
            pa = pwin(a, b, bo3=is_bo3)
            if random.random() < pa:
                rec[a][0] += 1
                rec[b][1] += 1
                logger.info(f"[Sim {sim_id}] [Round {round_cnt}] [Bo3] {a} > {b}  -> {a}({rec[a][0]}-{rec[a][1]}) {b}({rec[b][0]}-{rec[b][1]})")
            else:
                rec[a][1] += 1
                rec[b][0] += 1
                logger.info(f"[Sim {sim_id}] [Round {round_cnt}] [Bo3] {b} > {a}  -> {a}({rec[a][0]}-{rec[a][1]}) {b}({rec[b][0]}-{rec[b][1]})")
            played_set.add((a.lower(), b.lower()))

# ---------- 线程安全计数 ----------
counter = defaultdict(int)
blg_count = 0
lock = threading.Lock()

def worker(sim_id, count):
    local_counter = defaultdict(int)
    local_blg = 0
    for i in range(1, count + 1):
        adv = simulate_one(sim_id)
        for t in adv:
            local_counter[t] += 1
        if 'BLG' in adv:
            local_blg += 1
        # 每 1000 次回显一次（单核也安全）
        if i % 1000 == 0:
            print(f"Sim {sim_id}: {i}/{count} done")
    with lock:
        for t, c in local_counter.items():
            counter[t] += c
        global blg_count
        blg_count += local_blg

def main():
    # 份数计算（兼容任意 N_SIM / N_THREAD）
    per_thread = N_SIM // N_THREAD
    remainder = N_SIM % N_THREAD
    counts = [per_thread + 1 if i < remainder else per_thread for i in range(N_THREAD)]
    print(f"Start simulation | total={N_SIM} thread={N_THREAD}")
    start = time.time()
    with ThreadPoolExecutor(max_workers=N_THREAD) as pool:
        pool.map(worker, range(N_THREAD), counts)
    print(f"Done in {time.time() - start:.1f}s")

    print("\nEstimated 晋级概率（Top 8 晋级）：")
    for t, c in sorted(counter.items(), key=lambda x: -x[1]):
        print(f"{t:4s}: {c / N_SIM * 100:.2f}%")
    print(f"\nBLG 最终晋级概率：{blg_count / N_SIM * 100:.2f}%")

if __name__ == '__main__':
    main()