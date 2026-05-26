"""
環状道路 IDM + MOBIL シミュレーション（シミュレーション本体）
- n_lanes = 1 のとき従来どおり動作（experiments.py 変更不要）
- n_lanes >= 2 のとき
    * 外側車線ほど円周が長い（lane_length_ratio）
    * 外側車線ほど希望速度が高い（v0_lane_boost）
    → 車線ごとに異なる密度・速度が生まれ、自然な車線変更が発生する
"""

import numpy as np
from dataclasses import dataclass, field


# =========================================================
# パラメータ定義
# =========================================================

@dataclass
class IDMParams:
    """IDM のパラメータ（Treiber et al. 2000 の標準値）"""
    v0: float     = 16.0   # 希望速度 (m/s) ← 内側車線の基準値
    T: float      = 1.2    # 安全車間時間 (s)
    a_max: float  = 0.8    # 最大加速度 (m/s^2)
    b_comf: float = 1.5    # 快適減速度 (m/s^2)
    delta: int    = 4      # 加速度指数
    s0: float     = 2.0    # 最小車間距離 (m)
    l_veh: float  = 5.0    # 車長 (m)
    gap_closing_gain: float = 0.5  # 車間距離が広がったときに追加で加速する強さ
    gap_closing_scale: float = 10.0 # 追加加速の距離スケール (m)


@dataclass
class MOBILParams:
    """MOBIL 車線変更パラメータ（Kesting et al. 2007）"""
    politeness: float             = 0.2    # 礼儀係数 p
    b_safe: float                 = 3.0    # 安全減速度閾値 (m/s^2)
    a_thr: float                  = 0.05   # 車線変更の加速度閾値 (m/s^2)
    cooldown_steps: int           = 12     # 車線変更後のクールダウン
    crowd_bias: float             = 0.8    # 車線全体の台数差ボーナス重み
    congestion_bias: float        = 1.2    # 局所混雑を避けるボーナス重み
    local_congestion_range: float = 60.0   # 局所混雑判定の範囲 (m)


@dataclass
class SimParams:
    """シミュレーション設定"""
    ring_length: float       = 350.0      # 最内側車線の円周 (m)
    n_cars: int              = 28
    dt: float                = 0.25
    n_steps: int             = 1000
    n_lanes: int             = 1
    perturbation: str        = 'stop_one' # 'stop_one' | 'small' | 'none'
    lane_change_start: int   = 0          # このステップから車線変更を有効にする
    # ── 多車線物理パラメータ ──
    lane_length_ratio: float  = 1.08      # 1車線外側になるごとの円周の倍率
                                          # 例: 350m → 378m → 408m
    v0_lane_boost: float      = 1.5       # 1車線外側になるごとの希望速度の増加 (m/s)
                                          # 例: 16 m/s → 17.5 m/s → 19 m/s
    lane_car_weights: tuple   = None      # 車線ごとの初期配車比率（None = 均等）
                                          # 例: (3, 1) → 内側75%・外側25%
                                          # 例: (2, 1, 1) → 3車線で内側50%・他各25%
    driver_variation: float   = 0.15      # 運転特性のばらつき（相対標準偏差）
    driver_seed: int          = 1         # 乱数シード


@dataclass
class DriverParams:
    v0: np.ndarray
    T: np.ndarray
    a_max: np.ndarray
    gap_closing_gain: np.ndarray


# =========================================================
# 車線ごとのパラメータヘルパー
# =========================================================

def _lane_ring_length(lane_idx: int, sim: SimParams) -> float:
    """車線ごとの実効リング長（外側ほど長い）"""
    return sim.ring_length * (sim.lane_length_ratio ** lane_idx)


def _lane_v0(lane_idx: int, idm: IDMParams, sim: SimParams) -> float:
    """車線ごとの希望速度（外側ほど高い）"""
    return idm.v0 + sim.v0_lane_boost * lane_idx


# =========================================================
# IDM ヘルパー
# =========================================================

def _idm_accel(v_ego, v_lead, gap, idm: IDMParams, *,
                v0: np.ndarray | float = None,
                a_max: np.ndarray | float = None,
                T: np.ndarray | float = None,
                gap_closing_gain: np.ndarray | float = None):
    """IDM 加速度。個別パラメータを指定できる。"""
    if v0 is None:
        v0 = idm.v0
    if a_max is None:
        a_max = idm.a_max
    if T is None:
        T = idm.T
    if gap_closing_gain is None:
        gap_closing_gain = idm.gap_closing_gain

    sqrt_ab = np.sqrt(a_max * idm.b_comf)
    dv      = v_ego - v_lead
    s_star  = idm.s0 + np.maximum(0.0, v_ego * T + v_ego * dv / (2 * sqrt_ab))
    accel   = a_max * (1.0 - (v_ego / v0) ** idm.delta - (s_star / gap) ** 2)

    gap_diff  = gap - s_star
    gap_boost = gap_closing_gain * np.tanh(gap_diff / idm.gap_closing_scale)
    gap_boost = np.where(gap_diff > 0, gap_boost, 0.0)
    accel     = np.minimum(accel + gap_boost, a_max)

    return accel


def _idm_free(v_ego, idm: IDMParams, *,
              v0: np.ndarray | float = None,
              a_max: np.ndarray | float = None):
    """前車なし（自由走行）の加速度"""
    if v0 is None:
        v0 = idm.v0
    if a_max is None:
        a_max = idm.a_max
    return a_max * (1.0 - (v_ego / v0) ** idm.delta)


# =========================================================
# 車線内の加速度一括計算
# =========================================================

def _lane_accels(x, v, lanes, lane_idx, idm: IDMParams, sim: SimParams,
                 driver_params: DriverParams):
    """指定車線内の全車の加速度を計算。戻り値: {car_idx: accel}"""
    cars    = np.where(lanes == lane_idx)[0]
    ring_len = _lane_ring_length(lane_idx, sim)
    result  = {}

    if len(cars) == 0:
        return result

    if len(cars) == 1:
        ci = int(cars[0])
        result[ci] = float(_idm_free(
            v[ci], idm,
            v0=driver_params.v0[ci] + sim.v0_lane_boost * lane_idx,
            a_max=driver_params.a_max[ci]
        ))
        return result

    order    = np.argsort(x[cars])
    sc       = cars[order]
    sx       = x[sc]
    sv       = v[sc]

    x_lead   = np.roll(sx, -1)
    v_lead   = np.roll(sv, -1)
    gap      = (x_lead - sx) % ring_len - idm.l_veh
    gap      = np.maximum(gap, 0.1)

    v0_vec = driver_params.v0[sc] + sim.v0_lane_boost * lane_idx
    accels = _idm_accel(
        sv, v_lead, gap, idm,
        v0=v0_vec,
        a_max=driver_params.a_max[sc],
        T=driver_params.T[sc],
        gap_closing_gain=driver_params.gap_closing_gain[sc]
    )
    for ci, a in zip(sc, accels):
        result[int(ci)] = float(a)

    return result


# =========================================================
# MOBIL ヘルパー
# =========================================================

def _find_neighbors(ci, target_lane, x, v, lanes, sim: SimParams, idm: IDMParams):
    """
    car ci が target_lane にいると仮定したときの前後車を探す。
    gap 計算には target_lane のリング長を使う。
    """
    others   = np.where(lanes == target_lane)[0]
    others   = others[others != ci]
    ring_len = _lane_ring_length(target_lane, sim)

    if len(others) == 0:
        return None, np.inf, None, np.inf

    dx_fwd     = (x[others] - x[ci]) % ring_len
    lead_local = int(np.argmin(dx_fwd))
    leader     = int(others[lead_local])
    gap_lead   = max(float(dx_fwd[lead_local]) - idm.l_veh, 0.1)

    foll_local = int(np.argmax(dx_fwd))
    follower   = int(others[foll_local])
    gap_foll   = max(ring_len - float(dx_fwd[foll_local]) - idm.l_veh, 0.1)

    return leader, gap_lead, follower, gap_foll


def _local_congestion_score(ci, lane, x, lanes, sim: SimParams, mobil: MOBILParams):
    """自車周辺の局所混雑度（距離が近いほど重みが高い）"""
    others = np.where(lanes == lane)[0]
    others = others[others != ci]
    if len(others) == 0:
        return 0.0
    ring_len = _lane_ring_length(lane, sim)
    dx   = (x[others] - x[ci]) % ring_len
    dx   = np.minimum(dx, ring_len - dx)
    mask = dx < mobil.local_congestion_range
    if not mask.any():
        return 0.0
    weights = np.maximum(0.0, mobil.local_congestion_range - dx[mask])
    return float(np.sum(weights) / mobil.local_congestion_range)


def _mobil_gain(ci, cur_lane, tgt_lane, x, v, lanes, accel_dict,
                idm: IDMParams, sim: SimParams, mobil: MOBILParams,
                driver_params: DriverParams):
    """
    MOBIL 利得式:
        gain = (ã_c - a_c) + p×[(ã_n - a_n) + (ã_o - a_o)]
               + crowd_gain + congestion_gain
    """
    a_ego_old = accel_dict[ci]
    v0_tgt    = driver_params.v0[ci] + sim.v0_lane_boost * tgt_lane
    v0_cur    = driver_params.v0[ci] + sim.v0_lane_boost * cur_lane

    # ── 目標車線での自車加速度 ──
    leader_new, gap_lead_new, follower_new, gap_foll_new = _find_neighbors(
        ci, tgt_lane, x, v, lanes, sim, idm
    )
    if leader_new is None:
        a_ego_new = _idm_free(
            v[ci], idm,
            v0=v0_tgt,
            a_max=driver_params.a_max[ci]
        )
    else:
        a_ego_new = _idm_accel(
            v[ci], v[leader_new], gap_lead_new, idm,
            v0=v0_tgt,
            a_max=driver_params.a_max[ci],
            T=driver_params.T[ci],
            gap_closing_gain=driver_params.gap_closing_gain[ci]
        )

    # ── 目標車線の新後続車 ──
    if follower_new is not None:
        foll_v0 = driver_params.v0[follower_new] + sim.v0_lane_boost * tgt_lane
        a_foll_new_after = _idm_accel(
            v[follower_new], v[ci], gap_foll_new, idm,
            v0=foll_v0,
            a_max=driver_params.a_max[follower_new],
            T=driver_params.T[follower_new],
            gap_closing_gain=driver_params.gap_closing_gain[follower_new]
        )
        if a_foll_new_after < -mobil.b_safe:
            return None
        if leader_new is None or leader_new == follower_new:
            a_foll_new_before = _idm_free(
                v[follower_new], idm,
                v0=foll_v0,
                a_max=driver_params.a_max[follower_new]
            )
        else:
            ring_len = _lane_ring_length(tgt_lane, sim)
            gap = max((x[leader_new] - x[follower_new]) % ring_len - idm.l_veh, 0.1)
            a_foll_new_before = _idm_accel(
                v[follower_new], v[leader_new], gap, idm,
                v0=foll_v0,
                a_max=driver_params.a_max[follower_new],
                T=driver_params.T[follower_new],
                gap_closing_gain=driver_params.gap_closing_gain[follower_new]
            )
    else:
        a_foll_new_after = a_foll_new_before = 0.0

    # ── 現在車線の旧後続車 ──
    leader_old, _, follower_old, _ = _find_neighbors(
        ci, cur_lane, x, v, lanes, sim, idm
    )
    if follower_old is not None and follower_old != ci:
        a_foll_old_before = accel_dict.get(follower_old, 0.0)
        foll_old_v0 = driver_params.v0[follower_old] + sim.v0_lane_boost * cur_lane
        if leader_old is None or leader_old == ci:
            a_foll_old_after = _idm_free(
                v[follower_old], idm,
                v0=foll_old_v0,
                a_max=driver_params.a_max[follower_old]
            )
        else:
            ring_len = _lane_ring_length(cur_lane, sim)
            gap = max((x[leader_old] - x[follower_old]) % ring_len - idm.l_veh, 0.1)
            a_foll_old_after = _idm_accel(
                v[follower_old], v[leader_old], gap, idm,
                v0=foll_old_v0,
                a_max=driver_params.a_max[follower_old],
                T=driver_params.T[follower_old],
                gap_closing_gain=driver_params.gap_closing_gain[follower_old]
            )
    else:
        a_foll_old_before = a_foll_old_after = 0.0

    # ── 混雑バイアス（密度ベース：台数/リング長） ──
    count_cur    = int(np.sum(lanes == cur_lane))
    count_tgt    = int(np.sum(lanes == tgt_lane))
    rl_cur       = _lane_ring_length(cur_lane, sim)
    rl_tgt       = _lane_ring_length(tgt_lane, sim)
    density_cur  = count_cur / rl_cur
    density_tgt  = count_tgt / rl_tgt
    crowd_gain   = mobil.crowd_bias * max(0, density_cur - density_tgt)

    local_cur    = _local_congestion_score(ci, cur_lane, x, lanes, sim, mobil)
    local_tgt    = _local_congestion_score(ci, tgt_lane, x, lanes, sim, mobil)
    cong_gain    = mobil.congestion_bias * max(0.0, local_cur - local_tgt)

    gain = (
        (a_ego_new - a_ego_old)
        + mobil.politeness * (
            (a_foll_new_after  - a_foll_new_before)
            + (a_foll_old_after - a_foll_old_before)
        )
        + crowd_gain + cong_gain
    )
    return gain


# =========================================================
# 初期化
# =========================================================

def initialize_state(idm: IDMParams, sim: SimParams):
    """
    初期位置・速度・車線インデックスを生成。

    lane_car_weights が指定された場合はその比率で車を振り分ける。
    例: (3, 1) → 内側 Lane0 に 75%、外側 Lane1 に 25%
    None の場合は均等分配。
    """
    x = np.zeros(sim.n_cars, dtype=float)
    v = np.full(sim.n_cars, idm.v0 * 0.6)

    # ── 車線ごとの台数を決定 ──
    if sim.lane_car_weights is not None and sim.n_lanes > 1:
        weights = np.array(list(sim.lane_car_weights)[:sim.n_lanes], dtype=float)
        weights = weights / weights.sum()
        counts  = (weights * sim.n_cars).astype(int)
        counts[0] += sim.n_cars - counts.sum()   # 端数を内側車線に積む
    else:
        base   = sim.n_cars // sim.n_lanes
        counts = np.full(sim.n_lanes, base, dtype=int)
        counts[0] += sim.n_cars - base * sim.n_lanes

    # ── 車インデックスを車線に割り当て ──
    lanes = np.empty(sim.n_cars, dtype=int)
    idx = 0
    for lane_idx, cnt in enumerate(counts):
        lanes[idx:idx + cnt] = lane_idx
        idx += cnt

    # ── 車線ごとに位置を設定（各車線の円周に合わせて均等配置）──
    for lane_idx in range(sim.n_lanes):
        cars_in_lane = np.where(lanes == lane_idx)[0]
        ring_len     = _lane_ring_length(lane_idx, sim)
        n            = len(cars_in_lane)
        if n > 0:
            for j, ci in enumerate(cars_in_lane):
                x[ci] = j * ring_len / n

    # ── 摂動：各車線の先頭車を停止 or 減速 ──
    if sim.perturbation == 'stop_one':
        for lane_idx in range(sim.n_lanes):
            cars_in_lane = np.where(lanes == lane_idx)[0]
            if len(cars_in_lane) > 0:
                v[cars_in_lane[0]] = 0.0
    elif sim.perturbation == 'small':
        for lane_idx in range(sim.n_lanes):
            cars_in_lane = np.where(lanes == lane_idx)[0]
            if len(cars_in_lane) > 0:
                v[cars_in_lane[0]] *= 0.95

    return x, v, lanes


# =========================================================
# 1 ステップ更新
# =========================================================

def step(x, v, lanes, cooldown, idm: IDMParams, sim: SimParams,
         mobil: MOBILParams = None, driver_params: DriverParams = None,
         step_idx: int = 0):

    if driver_params is None:
        driver_params = DriverParams(
            v0=np.full(sim.n_cars, idm.v0),
            T=np.full(sim.n_cars, idm.T),
            a_max=np.full(sim.n_cars, idm.a_max),
            gap_closing_gain=np.full(sim.n_cars, idm.gap_closing_gain),
        )

    # ── 1. 全車線の加速度を一括計算 ──
    accel_dict = {}
    for lane_idx in range(sim.n_lanes):
        accel_dict.update(_lane_accels(
            x, v, lanes, lane_idx, idm, sim, driver_params
        ))
    accel = np.array([accel_dict[i] for i in range(sim.n_cars)])

    # ── 2. MOBIL 車線変更判定 ──
    new_lanes = lanes.copy()
    cooldown  = np.maximum(cooldown - 1, 0)

    if sim.n_lanes > 1 and mobil is not None and step_idx >= sim.lane_change_start:
        for ci in np.random.permutation(sim.n_cars):
            if cooldown[ci] > 0:
                continue
            cur_lane  = int(new_lanes[ci])
            best_gain = mobil.a_thr
            best_lane = cur_lane

            for tgt in [cur_lane - 1, cur_lane + 1]:
                if tgt < 0 or tgt >= sim.n_lanes:
                    continue
                gain = _mobil_gain(
                    ci, cur_lane, tgt, x, v, lanes, accel_dict,
                    idm, sim, mobil, driver_params
                )
                if gain is not None and gain > best_gain:
                    best_gain = gain
                    best_lane = tgt

            if best_lane != cur_lane:
                new_lanes[ci] = best_lane
                cooldown[ci]  = mobil.cooldown_steps

    # ── 3. 速度・位置の更新（位置は車線ごとのリング長でラップ）──
    v_new = np.clip(v + accel * sim.dt, 0.0, None)
    lane_ring_lens = np.array([_lane_ring_length(i, sim) for i in range(sim.n_lanes)])
    x_new = (x + v_new * sim.dt) % lane_ring_lens[new_lanes]

    return x_new, v_new, new_lanes, cooldown


# =========================================================
# シミュレーション本体
# =========================================================

def run_simulation(idm: IDMParams = None, sim: SimParams = None,
                   mobil: MOBILParams = None, record_positions: bool = False):
    if idm is None:
        idm = IDMParams()
    if sim is None:
        sim = SimParams()

    x, v, lanes = initialize_state(idm, sim)
    cooldown    = np.zeros(sim.n_cars, dtype=int)

    rng = np.random.default_rng(sim.driver_seed)
    v0_driver = idm.v0 * np.clip(
        1.0 + sim.driver_variation * rng.standard_normal(sim.n_cars),
        0.7, 1.3
    )
    T_driver = idm.T * np.clip(
        1.0 + sim.driver_variation * rng.standard_normal(sim.n_cars),
        0.7, 1.3
    )
    a_max_driver = idm.a_max * np.clip(
        1.0 + sim.driver_variation * rng.standard_normal(sim.n_cars),
        0.6, 1.4
    )
    gap_gain_driver = idm.gap_closing_gain * np.clip(
        1.0 + sim.driver_variation * rng.standard_normal(sim.n_cars),
        0.2, 2.0
    )
    driver_params = DriverParams(
        v0=v0_driver,
        T=T_driver,
        a_max=a_max_driver,
        gap_closing_gain=gap_gain_driver,
    )

    avg_v_hist = np.zeros(sim.n_steps)
    min_v_hist = np.zeros(sim.n_steps)

    if record_positions:
        x_history    = np.zeros((sim.n_steps, sim.n_cars))
        v_history    = np.zeros((sim.n_steps, sim.n_cars))
        lane_history = np.zeros((sim.n_steps, sim.n_cars), dtype=int)

    for step_idx in range(sim.n_steps):
        x, v, lanes, cooldown = step(
            x, v, lanes, cooldown, idm, sim, mobil,
            driver_params, step_idx
        )
        avg_v_hist[step_idx] = v.mean()
        min_v_hist[step_idx] = v.min()
        if record_positions:
            x_history[step_idx]    = x
            v_history[step_idx]    = v
            lane_history[step_idx] = lanes

    result = {'avg_v': avg_v_hist, 'min_v': min_v_hist}
    if record_positions:
        result['x_history']    = x_history
        result['v_history']    = v_history
        result['lane_history'] = lane_history
    return result


def summary_stats(result, transient: int = 200):
    avg_s = result['avg_v'][transient:]
    min_s = result['min_v'][transient:]
    return {
        'mean_avg_v'    : float(avg_s.mean()),
        'mean_min_v'    : float(min_s.mean()),
        'std_avg_v'     : float(avg_s.std()),
        'wave_amplitude': float(avg_s.mean() - min_s.mean()),
    }


# =========================================================
# 動作確認
# =========================================================

if __name__ == '__main__':
    import time
    for n_lanes in [1, 2]:
        print(f'\n--- n_lanes = {n_lanes} ---')
        sim   = SimParams(n_lanes=n_lanes, n_cars=28*n_lanes, n_steps=600)
        mobil = MOBILParams() if n_lanes > 1 else None
        t0    = time.perf_counter()
        result = run_simulation(sim=sim, mobil=mobil)
        t1    = time.perf_counter()
        stats = summary_stats(result)
        print(f'  elapsed : {(t1-t0)*1000:.1f} ms')
        print(f'  stats   : {stats}')