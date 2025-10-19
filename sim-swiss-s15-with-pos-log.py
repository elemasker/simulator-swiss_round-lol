"""
S15 瑞士轮模拟器
- 任意 N_SIM / N_THREAD 兼容
- 按 主观战力指数（比例）概率 预测输赢
- 任意 对局进度 兼容：修改初始战绩和对局进度
- 新增dominate_rate（虐菜率）：强队对弱队的稳定性
- 新增comeback_rate（翻盘率）：弱队对强队的爆冷概率
- 按队伍特性配置参数，调节盘中输赢概率
"""
import random, math, threading, time, logging
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

random.seed(42)
N_SIM = 50000  # 模拟次数
N_THREAD = 4   # 线程数
LOG_FILE = 'swiss_sim.log'

# 配置日志：输出到文件和控制台（按需求保留）
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
        # logging.StreamHandler()  # 新增控制台输出
    ]
)
logger = logging.getLogger()

# ---------- 核心数据（含新增参数）----------
# 1. 初始战绩 (胜, 负)
RECORD = {
    'GEN': (2, 1), 'KT': (3, 0), 'AL': (3, 0), 'TES': (2, 1),
    'HLE': (2, 1), 'CFO': (2, 1), 'BLG': (1, 2), 'T1': (1, 2),
    '100T': (1, 2), 'TSW': (1, 2), 'FNC': (0, 2), 'MKOI': (0, 2),
    'PSG': (0, 2), 'VKS': (0, 2), 'FLY': (2, 1), 'G2': (2, 1)
}

# RECORD = {
#     'GEN': (0, 0), 'KT': (0, 0), 'AL': (0, 0), 'TES': (0, 0),
#     'HLE': (0, 0), 'CFO': (0, 0), 'BLG': (0, 0), 'T1': (0, 0),
#     '100T': (0, 0), 'TSW': (0, 0), 'FNC': (0, 0), 'MKOI': (0, 0),
#     'PSG': (0, 0), 'VKS': (0, 0), 'FLY': (0, 0), 'G2': (0, 0)
# }

# 2. 基础实力值
POWER = {
    'GEN': 100, 'HLE': 91, 'T1': 87, 'KT': 85, 
    'BLG': 91, 'AL': 95, 'TES': 83,
    'CFO': 84, 'TSW': 70, 'PSG': 65,
    'G2': 81, 'FNC': 74, 'MKOI': 70,
    'FLY': 79,'100T': 72, 'VKS': 66
}

# 3. 虐菜率（dominate_rate）：面对弱队时的稳定性（0~1，越高越稳）
DOMINATE_RATE = {
    'GEN': 0.8, 'KT': 0.6, 'T1': 0.8, 'HLE': 0.7,
    'BLG': 0.4, 'AL': 0.8, 'TES': 0.6,
    'CFO': 0.5, 'TSW': 0.4, 'PSG': 0.3,
    'G2': 0.7, 'FNC': 0.6, 'MKOI': 0.5,
    'VKS': 0.3, 'FLY': 0.6, '100T': 0.4
}

# 4. 翻盘率（comeback_rate）：面对强队时的爆冷概率（0~1，越高越易翻盘）
COMEBACK_RATE = {
    'GEN': 0.2, 'KT': 0.4, 'T1': 0.8, 'HLE': 0.5,
    'BLG': 0.6, 'AL': 0.5, 'TES': 0.4,
    'CFO': 0.5, 'TSW': 0.3, 'PSG': 0.3,
    'G2': 0.5, 'FNC': 0.3, 'MKOI': 0.3,
    'VKS': 0.3, 'FLY': 0.5, '100T': 0.4
}

# 5. 赛区归属
REGION = {
    'GEN': 'LCK', 'KT': 'LCK', 'T1': 'LCK', 'HLE': 'LCK',
    'BLG': 'LPL', 'TES': 'LPL', 'AL': 'LPL',
    'CFO': 'LCP', 'TSW': 'LCP', 'PSG': 'LCP',
    'G2': 'LEC', 'FNC': 'LEC', 'MKOI': 'LEC',
    'VKS': 'LTA', 'FLY': 'LTA', '100T': 'LTA',
}

ALL_TEAMS = list(RECORD.keys())
ALL_REGIONS = sorted(list(set(REGION.values())))
LPL_TEAMS = [t for t in ALL_TEAMS if REGION[t] == 'LPL' and t != 'BLG']
LCK_TEAMS = [t for t in ALL_TEAMS if REGION[t] == 'LCK']

# 已对战记录
PLAYED = {
    ('tsw', 'vks'), ('cfo', 'fnc'), ('kt', 'mkoi'), ('blg', '100t'),
    ('fly', 'hle'), ('al', 'g2'), ('tes', 'gen'), ('gen', 'psg'),
    ('vks', 'fly'), ('kt', 'tsw'), ('mkoi', 'g2'), ('tes', '100t'),
    ('cfo', 'al'), ('gen', 'al'), ('blg', 'fnc'), ('hle', 'psg'),
    ('kt', 'tes'), ('tsw', 'fly'), ('gen', 't1'), ('g2', 'blg'), ('100t', 'hle')
}

# PLAYED = {}          # 已对战集合，元素为小写元组 (a, b)

CURRENT_ROUND = max(w + l for w, l in RECORD.values())  # 当前轮次：0（因初始战绩全0-0）


# ---------- 核心函数（含参数调节）----------
def pwin(a, b, bo3=False):
    """计算a战胜b的概率，结合翻盘率和虐菜率调节"""
    # 基础胜率（基于实力差）
    diff = POWER[a] - POWER[b]
    k = 0.25 if bo3 else 0.18  # BO3系数更高
    base_p = 1 / (1 + math.exp(-k * diff))  # 基础Sigmoid曲线
    
    # 调节系数（控制翻盘/虐菜的影响幅度）
    adjust_k = 0.3 if bo3 else 0.25
    
    if diff > 0:
        # a是强队（实力高于b）：a的虐菜率提升胜率，b的翻盘率提升胜率
        a_adj = base_p + (1 - base_p) * DOMINATE_RATE[a] * adjust_k
        b_adj = (1 - base_p) + base_p * COMEBACK_RATE[b] * adjust_k
        a_p = a_adj / (a_adj + b_adj)  # 重新归一化，确保a_p + b_p = 1
    elif diff < 0:
        # a是弱队（实力低于b）：a的翻盘率提升胜率，b的虐菜率提升胜率
        a_adj = base_p + (1 - base_p) * COMEBACK_RATE[a] * adjust_k
        b_adj = (1 - base_p) + base_p * DOMINATE_RATE[b] * adjust_k
        a_p = a_adj / (a_adj + b_adj)  # 重新归一化
    else:
        # 实力相等：直接用基础概率
        a_p = base_p
    
    # 限制概率范围（避免极端值）
    return min(0.995, max(0.005, a_p))

def can_play(t1, t2, played_set):
    """判断是否可对战（不同队伍+未交手）"""
    if t1 == t2:
        return False
    t1_low, t2_low = t1.lower(), t2.lower()
    return (t1_low, t2_low) not in played_set and (t2_low, t1_low) not in played_set

def draw_pairs(teams, played_set):
    """普通抽签（仅避免重复对战）"""
    teams = teams.copy()
    random.shuffle(teams)
    pairs, used = [], set()
    for t in teams:
        if t in used:
            continue
        for opp in teams:
            if opp in used or not can_play(t, opp, played_set):
                continue
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

def draw_pairs_avoid(teams, played_set, region_dict):
    """第一轮抽签（优先异赛区）"""
    teams = teams.copy()
    random.shuffle(teams)
    pairs, used = [], set()
    for t in teams:
        if t in used:
            continue
        # 优先异赛区+未对战
        for opp in teams:
            if opp in used or region_dict[t] == region_dict[opp] or not can_play(t, opp, played_set):
                continue
            pairs.append((t, opp))
            used.add(t)
            used.add(opp)
            break
        else:
            # 次选同赛区未对战
            for opp in teams:
                if opp not in used and can_play(t, opp, played_set):
                    pairs.append((t, opp))
                    used.add(t)
                    used.add(opp)
                    break
            else:
                # 兜底
                for opp in teams:
                    if opp != t and opp not in used:
                        pairs.append((t, opp))
                        used.add(t)
                        used.add(opp)
                        break
    return pairs


def simulate_one(sim_id):
    """单次模拟（含翻盘率/虐菜率影响及日志输出）"""
    rec = {t: [w, l] for t, (w, l) in RECORD.items()}  # 战绩深拷贝
    played_set = set(PLAYED)  # 已对战记录

    # 概率记录结构
    blg_region_prob = [defaultdict(float) for _ in range(5 - CURRENT_ROUND)]
    blg_lpl_teams_prob = [defaultdict(float) for _ in range(5 - CURRENT_ROUND)]
    blg_lck_teams_prob = [defaultdict(float) for _ in range(5 - CURRENT_ROUND)]

    while True:
        # 存活队伍：未晋级（<3胜）且未淘汰（<3负）
        alive = {t for t in ALL_TEAMS if rec[t][0] < 3 and rec[t][1] < 3}
        # 当前最大总场次（已完成轮次）
        max_games = max(w + l for w, l in rec.values())
        # 本轮未完成比赛的队伍（总场次不足）
        incomplete = [t for t in alive if (rec[t][0] + rec[t][1]) < max_games]

        # 1. 补赛逻辑（0-2队伍获胜后可进入1-2桶）
        if incomplete:
            buckets = defaultdict(list)
            for t in incomplete:
                buckets[(rec[t][0], rec[t][1])].append(t)  # 按当前战绩分桶
            
            for (w, l), teams in buckets.items():
                is_bo3 = (w, l) in [(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]
                bo_type = "Bo3" if is_bo3 else "Bo1"
                pairs = draw_pairs(teams, played_set)
                for a, b in pairs:
                    # 记录赛前战绩（用于日志对比）
                    a_pre = (rec[a][0], rec[a][1])
                    b_pre = (rec[b][0], rec[b][1])
                    # 计算胜率并判断胜负
                    pa = pwin(a, b, bo3=is_bo3)
                    if random.random() < pa:
                        # a获胜
                        rec[a][0] += 1
                        rec[b][1] += 1
                        winner, loser = a, b
                    else:
                        # b获胜
                        rec[a][1] += 1
                        rec[b][0] += 1
                        winner, loser = b, a
                    # 输出补赛日志（补赛属于当前max_games轮次）
                    logger.info(
                        f"[Sim {sim_id}] [Round {max_games}] [{bo_type}] {winner} > {loser}  -> {winner}({rec[winner][0]}-{rec[winner][1]}) {loser}({rec[loser][0]}-{rec[loser][1]})"
                    )
                    played_set.add((a.lower(), b.lower()))
            continue

        # 2. 采样逻辑（计算BLG对手概率）
        next_round = max_games + 1
        idx = next_round - CURRENT_ROUND - 1

        if 'BLG' in alive and 0 <= idx < (5 - CURRENT_ROUND):
            w_blg, l_blg = rec['BLG']
            # 对手池：非BLG、存活、同战绩、未对战
            pool = [
                t for t in alive
                if t != 'BLG'
                and rec[t][0] == w_blg 
                and rec[t][1] == l_blg 
                and can_play('BLG', t, played_set)
            ]
            total = len(pool)
            if total == 0:
                continue

            # 计算概率
            region_count = defaultdict(int)
            for t in pool:
                region_count[REGION[t]] += 1
            for region in ALL_REGIONS:
                blg_region_prob[idx][region] = region_count[region] / total if total > 0 else 0.0

            for team in LPL_TEAMS:
                blg_lpl_teams_prob[idx][team] = 1 / total if (team in pool) else 0.0

            for team in LCK_TEAMS:
                blg_lck_teams_prob[idx][team] = 1 / total if (team in pool) else 0.0

        # 3. 终止条件（8强决出）
        advanced = {t for t in ALL_TEAMS if rec[t][0] == 3}
        alive_2_2 = {t for t in alive if rec[t] == [2, 2]}
        if len(advanced) >= 8 or (len(advanced) + len(alive_2_2) == 8):
            return advanced, blg_region_prob, blg_lpl_teams_prob, blg_lck_teams_prob

        # 4. 下一轮对战（按当前战绩分桶）
        buckets = defaultdict(list)
        for t in alive:
            buckets[(rec[t][0], rec[t][1])].append(t)

        next_matches = []
        for (w, l), teams in buckets.items():
            is_bo3 = (w, l) in [(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]
            if next_round == 1:
                pairs = draw_pairs_avoid(teams, played_set, REGION)
            else:
                pairs = draw_pairs(teams, played_set)
            for a, b in pairs:
                next_matches.append((a, b, is_bo3))

        # 模拟下一轮比赛并输出日志
        for a, b, is_bo3 in next_matches:
            bo_type = "Bo3" if is_bo3 else "Bo1"
            # 记录赛前战绩
            a_pre = (rec[a][0], rec[a][1])
            b_pre = (rec[b][0], rec[b][1])
            # 计算胜负
            pa = pwin(a, b, bo3=is_bo3)
            if random.random() < pa:
                # a获胜
                rec[a][0] += 1
                rec[b][1] += 1
                winner, loser = a, b
            else:
                # b获胜
                rec[a][1] += 1
                rec[b][0] += 1
                winner, loser = b, a
            # 输出下一轮对战日志（属于next_round轮次）
            logger.info(
                f"[Sim {sim_id}] [Round {next_round}] [{bo_type}] {winner} > {loser}  -> {winner}({rec[winner][0]}-{rec[winner][1]}) {loser}({rec[loser][0]}-{rec[loser][1]})"
            )
            played_set.add((a.lower(), b.lower()))


# ---------- 多线程统计 ----------
counter = defaultdict(int)
blg_count = 0
region_prob = [defaultdict(float) for _ in range(5 - CURRENT_ROUND)]
lpl_teams_prob = [defaultdict(float) for _ in range(5 - CURRENT_ROUND)]
lck_teams_prob = [defaultdict(float) for _ in range(5 - CURRENT_ROUND)]
lock = threading.Lock()


def worker(sim_id, count):
    local_counter = defaultdict(int)
    local_blg = 0
    local_region = [defaultdict(float) for _ in range(5 - CURRENT_ROUND)]
    local_lpl = [defaultdict(float) for _ in range(5 - CURRENT_ROUND)]
    local_lck = [defaultdict(float) for _ in range(5 - CURRENT_ROUND)]
    
    for i in range(1, count + 1):
        adv, reg_prob, lpl_prob, lck_prob = simulate_one(sim_id)
        for t in adv:
            local_counter[t] += 1
        if 'BLG' in adv:
            local_blg += 1
        
        for r in range(5 - CURRENT_ROUND):
            for region in ALL_REGIONS:
                local_region[r][region] += reg_prob[r][region]
            for team in LPL_TEAMS:
                local_lpl[r][team] += lpl_prob[r][team]
            for team in LCK_TEAMS:
                local_lck[r][team] += lck_prob[r][team]
        
        if i % 1000 == 0:
            print(f"Sim {sim_id}: {i}/{count} done")
    
    with lock:
        for t, c in local_counter.items():
            counter[t] += c
        global blg_count
        blg_count += local_blg
        
        for r in range(5 - CURRENT_ROUND):
            for region in ALL_REGIONS:
                region_prob[r][region] += local_region[r][region]
            for team in LPL_TEAMS:
                lpl_teams_prob[r][team] += local_lpl[r][team]
            for team in LCK_TEAMS:
                lck_teams_prob[r][team] += local_lck[r][team]


# ---------- 主入口 ----------
def main():
    per_thread = N_SIM // N_THREAD
    remainder = N_SIM % N_THREAD
    counts = [per_thread + 1 if i < remainder else per_thread for i in range(N_THREAD)]
    
    print(f"Start simulation | total={N_SIM} thread={N_THREAD} current=Round{CURRENT_ROUND + 1}")
    start = time.time()
    
    with ThreadPoolExecutor(max_workers=N_THREAD) as pool:
        pool.map(worker, range(N_THREAD), counts)
    
    print(f"Done in {time.time() - start:.1f}s")
    
    print("\n=== 晋级概率（Top 8） ===")
    for t, c in sorted(counter.items(), key=lambda x: -x[1]):
        print(f"{t:4s}: {c / N_SIM * 100:.2f}%")
    print(f"\nBLG 最终晋级概率：{blg_count / N_SIM * 100:.2f}%")
    
    print(f"\n=== BLG 从第{CURRENT_ROUND + 1}轮起遇到各赛区的总概率 ===")
    for r in range(5 - CURRENT_ROUND):
        round_num = r + CURRENT_ROUND + 1
        print(f"\nRound {round_num}:")
        for region in ALL_REGIONS:
            prob = region_prob[r][region] / N_SIM * 100
            print(f"  {region}: {prob:.1f}%")
    
    print(f"\n=== BLG 从第{CURRENT_ROUND + 1}轮起遇到 LPL 各队伍的概率 ===")
    for r in range(5 - CURRENT_ROUND):
        round_num = r + CURRENT_ROUND + 1
        print(f"\nRound {round_num}:")
        for team in sorted(LPL_TEAMS):
            prob = lpl_teams_prob[r][team] / N_SIM * 100
            print(f"  {team}: {prob:.1f}%")
    
    print(f"\n=== BLG 从第{CURRENT_ROUND + 1}轮起遇到 LCK 各队伍的概率 ===")
    for r in range(5 - CURRENT_ROUND):
        round_num = r + CURRENT_ROUND + 1
        print(f"\nRound {round_num}:")
        for team in sorted(LCK_TEAMS):
            prob = lck_teams_prob[r][team] / N_SIM * 100
            print(f"  {team}: {prob:.1f}%")


if __name__ == '__main__':
    main()