# -*- coding: utf-8 -*-

import numpy as np
from dataclasses import dataclass


# =========================================================
# IDM パラメータ
# =========================================================

@dataclass
class IDMParams:

    v0: float = 16.0
    T: float = 1.2
    a_max: float = 0.8
    b_comf: float = 1.5
    delta: int = 4
    s0: float = 2.0
    l_veh: float = 5.0


# =========================================================
# シミュレーション設定
# =========================================================

@dataclass
class SimParams:

    ring_length: float = 350.0
    n_cars: int = 24

    dt: float = 0.25
    n_steps: int = 600

    perturbation: str = 'small'

    # -----------------------------
    # カーブ区間
    # -----------------------------

    curve_start: float = 100.0
    curve_end: float = 170.0

    # カーブでの希望速度
    curve_v0: float = 12.0

    # -----------------------------
    # ランダムブレーキ
    # -----------------------------

    random_brake_prob: float = 0.003

    random_brake_strength: float = 1.0


# =========================================================
# 初期状態
# =========================================================

def initialize_state(idm, sim):

    x = (
        np.arange(sim.n_cars)
        * sim.ring_length
        / sim.n_cars
    )

    v = np.full(
        sim.n_cars,
        idm.v0 * 0.75
    )

    if sim.perturbation == 'stop_one':

        v[0] = 0.0

    elif sim.perturbation == 'small':

        v[0] *= 0.95

    return x, v


# =========================================================
# 1ステップ更新
# =========================================================

def step(x, v, idm, sim):

    order = np.argsort(x)

    x_sorted = x[order]
    v_sorted = v[order]

    # 前車
    x_lead = np.roll(x_sorted, -1)
    v_lead = np.roll(v_sorted, -1)

    # 車間距離
    gap = (
        (x_lead - x_sorted)
        % sim.ring_length
    ) - idm.l_veh

    gap = np.maximum(gap, 0.1)

    # =====================================================
    # カーブ判定
    # =====================================================

    in_curve = (
        (x_sorted >= sim.curve_start)
        & (x_sorted <= sim.curve_end)
    )

    # カーブ内では希望速度を低下
    v0_local = np.where(
        in_curve,
        sim.curve_v0,
        idm.v0
    )

    # =====================================================
    # IDM
    # =====================================================

    sqrt_ab = np.sqrt(
        idm.a_max * idm.b_comf
    )

    dv = v_sorted - v_lead

    s_star = (
        idm.s0
        + np.maximum(
            0.0,
            v_sorted * idm.T
            + (
                v_sorted * dv
            ) / (2 * sqrt_ab)
        )
    )

    accel = idm.a_max * (
        1
        - (v_sorted / v0_local) ** idm.delta
        - (s_star / gap) ** 2
    )

    # =====================================================
    # ランダムブレーキ
    # =====================================================

    random_mask = (
        np.random.rand(sim.n_cars)
        < sim.random_brake_prob
    )

    accel[random_mask] -= (
        sim.random_brake_strength
    )

    # =====================================================
    # 更新
    # =====================================================

    v_sorted_new = (
        v_sorted
        + accel * sim.dt
    )

    v_sorted_new = np.maximum(
        v_sorted_new,
        0.0
    )

    x_sorted_new = (
        x_sorted
        + v_sorted_new * sim.dt
    ) % sim.ring_length

    # 元順へ戻す

    x_new = np.empty_like(x)
    v_new = np.empty_like(v)

    x_new[order] = x_sorted_new
    v_new[order] = v_sorted_new

    accel_original = np.empty_like(accel)
    accel_original[order] = accel

    gap_original = np.empty_like(gap)
    gap_original[order] = gap

    return (
        x_new,
        v_new,
        accel_original,
        gap_original
    )


# =========================================================
# シミュレーション本体
# =========================================================

def run_simulation(
    idm=None,
    sim=None,
    record_positions=False
):

    if idm is None:
        idm = IDMParams()

    if sim is None:
        sim = SimParams()

    x, v = initialize_state(idm, sim)

    avg_v_hist = np.zeros(sim.n_steps)
    min_v_hist = np.zeros(sim.n_steps)

    std_v_hist = np.zeros(sim.n_steps)

    jam_count_hist = np.zeros(sim.n_steps)

    mean_gap_hist = np.zeros(sim.n_steps)

    mean_accel_hist = np.zeros(sim.n_steps)

    if record_positions:

        x_history = np.zeros(
            (sim.n_steps, sim.n_cars)
        )

        v_history = np.zeros(
            (sim.n_steps, sim.n_cars)
        )

    for step_idx in range(sim.n_steps):

        x, v, accel, gap = step(
            x,
            v,
            idm,
            sim
        )

        avg_v_hist[step_idx] = v.mean()

        min_v_hist[step_idx] = v.min()

        std_v_hist[step_idx] = v.std()

        jam_count_hist[step_idx] = (
            v < 5.0
        ).sum()

        mean_gap_hist[step_idx] = gap.mean()

        mean_accel_hist[step_idx] = accel.mean()

        if record_positions:

            x_history[step_idx] = x

            v_history[step_idx] = v

    result = {

        'avg_v': avg_v_hist,

        'min_v': min_v_hist,

        'std_v': std_v_hist,

        'jam_count': jam_count_hist,

        'mean_gap': mean_gap_hist,

        'mean_accel': mean_accel_hist,
    }

    if record_positions:

        result['x_history'] = x_history

        result['v_history'] = v_history

    return result


# =========================================================
# 統計表示
# =========================================================

def summary_stats(
    result,
    transient=200
):

    avg_steady = result['avg_v'][transient:]

    return {

        'mean_avg_v':
            float(avg_steady.mean()),

        'std_avg_v':
            float(avg_steady.std()),

        'max_jam_count':
            float(result['jam_count'].max()),
    }


if __name__ == '__main__':

    print('Running simulation...')

    result = run_simulation()

    print(summary_stats(result))