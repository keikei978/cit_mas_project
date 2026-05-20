# -*- coding: utf-8 -*-
"""
環状道路 IDM シミュレーション（シミュレーション本体）
- matplotlib に依存しない
- 全車の状態を numpy 配列で一括計算（ベクトル化）
- このファイル単体では何も描画しない
- 描画/アニメーションは visualize_animation.py などから呼び出す
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class IDMParams:
    """IDM のパラメータ（Treiber et al. 2000 の標準値）"""
    v0: float = 16.0        # 希望速度 (m/s)
    T: float = 1.2          # 安全車間時間 (s)
    a_max: float = 0.8      # 最大加速度 (m/s^2)
    b_comf: float = 1.5     # 快適減速度 (m/s^2)
    delta: int = 4          # 加速度指数
    s0: float = 2.0         # 最小車間距離 (m)
    l_veh: float = 5.0      # 車長 (m)


@dataclass
class SimParams:
    """シミュレーション設定"""
    ring_length: float = 350.0
    n_cars: int = 28
    dt: float = 0.25
    n_steps: int = 600
    perturbation: str = 'stop_one'  # 'stop_one' | 'small' | 'none'


def initialize_state(idm: IDMParams, sim: SimParams):
    """初期位置・速度を作る"""
    x = np.arange(sim.n_cars, dtype=float) * sim.ring_length / sim.n_cars
    v = np.full(sim.n_cars, idm.v0 * 0.6)

    if sim.perturbation == 'stop_one':
        v[0] = 0.0
    elif sim.perturbation == 'small':
        v[0] *= 0.95  # 5% だけ落とす
    # 'none' は何もしない

    return x, v


def step(x, v, idm: IDMParams, sim: SimParams):
    """
    1ステップ進める。全車の状態を numpy 配列で一括更新する。
    x, v を新しい配列にして返す。
    """
    # 位置順にソート（前車を特定するため）
    order = np.argsort(x)
    x_sorted = x[order]
    v_sorted = v[order]

    # 前車 = ソート順で次の車（最後は先頭に戻る）
    x_lead = np.roll(x_sorted, -1)
    v_lead = np.roll(v_sorted, -1)

    # 車間距離（mod でリング一周を処理）
    gap = (x_lead - x_sorted) % sim.ring_length - idm.l_veh
    gap = np.maximum(gap, 0.1)  # 0除算回避

    # IDM の加速度を全車一括計算
    sqrt_ab = np.sqrt(idm.a_max * idm.b_comf)
    dv = v_sorted - v_lead
    s_star = idm.s0 + np.maximum(
        0.0, v_sorted * idm.T + (v_sorted * dv) / (2 * sqrt_ab)
    )
    accel = idm.a_max * (
        1 - (v_sorted / idm.v0) ** idm.delta - (s_star / gap) ** 2
    )

    # 速度・位置の更新
    v_sorted = v_sorted + accel * sim.dt
    v_sorted = np.maximum(v_sorted, 0.0)  # バックしない
    x_sorted = (x_sorted + v_sorted * sim.dt) % sim.ring_length

    # 元の順序に戻す
    x_new = np.empty_like(x)
    v_new = np.empty_like(v)
    x_new[order] = x_sorted
    v_new[order] = v_sorted

    return x_new, v_new


def run_simulation(idm: IDMParams = None, sim: SimParams = None, record_positions: bool = False):
    """
    シミュレーションを1回走らせる。

    引数:
        idm, sim: パラメータ。None ならデフォルト
        record_positions: True にすると全車の全ステップの位置・速度を保存
                          （時空間プロットに使う。メモリを食うので必要時のみ）

    戻り値:
        dict 形式
        - avg_v: 各ステップの平均速度 (n_steps,)
        - min_v: 各ステップの最低速度 (n_steps,)
        - x_history: record_positions=True のとき (n_steps, n_cars)
        - v_history: record_positions=True のとき (n_steps, n_cars)
    """
    if idm is None:
        idm = IDMParams()
    if sim is None:
        sim = SimParams()

    x, v = initialize_state(idm, sim)

    avg_v_hist = np.zeros(sim.n_steps)
    min_v_hist = np.zeros(sim.n_steps)

    if record_positions:
        x_history = np.zeros((sim.n_steps, sim.n_cars))
        v_history = np.zeros((sim.n_steps, sim.n_cars))

    for step_idx in range(sim.n_steps):
        x, v = step(x, v, idm, sim)
        avg_v_hist[step_idx] = v.mean()
        min_v_hist[step_idx] = v.min()
        if record_positions:
            x_history[step_idx] = x
            v_history[step_idx] = v

    result = {'avg_v': avg_v_hist, 'min_v': min_v_hist}
    if record_positions:
        result['x_history'] = x_history
        result['v_history'] = v_history
    return result


def summary_stats(result, transient: int = 200):
    """過渡状態を除いた定常部分の要約統計"""
    avg_steady = result['avg_v'][transient:]
    min_steady = result['min_v'][transient:]
    return {
        'mean_avg_v': float(avg_steady.mean()),
        'mean_min_v': float(min_steady.mean()),
        'std_avg_v': float(avg_steady.std()),
        'wave_amplitude': float(avg_steady.mean() - min_steady.mean()),
    }


# このファイルを直接実行したときの動作確認
if __name__ == '__main__':
    import time
    print('Running default simulation...')
    t0 = time.perf_counter()
    result = run_simulation()
    t1 = time.perf_counter()
    stats = summary_stats(result)
    print(f'  elapsed: {(t1-t0)*1000:.1f} ms')
    print(f'  stats: {stats}')