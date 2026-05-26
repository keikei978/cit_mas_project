# -*- coding: utf-8 -*-

"""
環状道路シミュレーション：実験2・実験3

必要ファイル：
    traffic_jam_mas.py

このスクリプトで行うこと：
    - 実験2：N スイープ
    - 実験3：T スイープ
    - グラフ表示
    - PNG 保存
    - npz データ保存

保存先：
    この .py ファイルと同じフォルダ
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from traffic_jam_mas import (
    IDMParams,
    SimParams,
    run_simulation,
    summary_stats
)

# =========================================================
# 保存先設定
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("Save directory:")
print(BASE_DIR)

# =========================================================
# 共通スイープ関数
# =========================================================

def run_sweep(varying_param, values, fixed_params=None):

    fixed_params = fixed_params or {}

    mean_v = []
    min_v = []

    for val in values:

        idm_kwargs = {}
        sim_kwargs = {}

        # 固定パラメータを分類
        for k, v in fixed_params.items():

            if k in [
                'v0',
                'T',
                'a_max',
                'b_comf',
                'delta',
                's0',
                'l_veh'
            ]:
                idm_kwargs[k] = v

            else:
                sim_kwargs[k] = v

        # 変化させるパラメータ
        if varying_param in [
            'v0',
            'T',
            'a_max',
            'b_comf',
            'delta',
            's0',
            'l_veh'
        ]:
            idm_kwargs[varying_param] = val

        else:
            sim_kwargs[varying_param] = val

        # オブジェクト生成
        idm = IDMParams(**idm_kwargs)
        sim = SimParams(**sim_kwargs)

        # シミュレーション実行
        result = run_simulation(idm=idm, sim=sim)

        # 統計量
        stats = summary_stats(result, transient=300)

        mean_v.append(stats['mean_avg_v'])
        min_v.append(stats['mean_min_v'])

        print(
            f'{varying_param} = {val} '
            f'| mean = {stats["mean_avg_v"]:.2f} '
            f'| min = {stats["mean_min_v"]:.2f}'
        )

    return mean_v, min_v


# =========================================================
# 実験2：N スイープ
# =========================================================

print('\n==============================')
print('Experiment 2 : N Sweep')
print('==============================')

n_values = list(range(10, 41))

mean_v_n, min_v_n = run_sweep(
    varying_param='n_cars',
    values=n_values,
    fixed_params={
        'n_steps': 800,
        'ring_length': 350.0
    }
)

# データ保存
exp2_path = os.path.join(BASE_DIR, 'exp2_data.npz')

np.savez(
    exp2_path,
    n=n_values,
    mean_v=mean_v_n,
    min_v=min_v_n
)

print(f'Experiment 2 data saved:\n{exp2_path}')


# =========================================================
# 実験3：T スイープ
# =========================================================

print('\n==============================')
print('Experiment 3 : T Sweep')
print('==============================')

t_values = np.arange(0.6, 2.05, 0.1)

mean_v_t, min_v_t = run_sweep(
    varying_param='T',
    values=t_values,
    fixed_params={
        'n_cars': 28,
        'n_steps': 800,
        'ring_length': 350.0
    }
)

# データ保存
exp3_path = os.path.join(BASE_DIR, 'exp3_data.npz')

np.savez(
    exp3_path,
    T=t_values,
    mean_v=mean_v_t,
    min_v=min_v_t
)

print(f'Experiment 3 data saved:\n{exp3_path}')


# =========================================================
# グラフ描画
# =========================================================

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(14, 5)
)

# ---------------------------------------------------------
# 実験2
# ---------------------------------------------------------

ax1.plot(
    n_values,
    mean_v_n,
    'o-',
    label='Mean velocity'
)

ax1.plot(
    n_values,
    min_v_n,
    's-',
    label='Min velocity'
)

ax1.set_xlabel('Number of cars N')
ax1.set_ylabel('Velocity (m/s)')
ax1.set_title('Experiment 2 : N Sweep')

ax1.grid(True, alpha=0.3)
ax1.legend()

# ---------------------------------------------------------
# 実験3
# ---------------------------------------------------------

ax2.plot(
    t_values,
    mean_v_t,
    'o-',
    label='Mean velocity'
)

ax2.plot(
    t_values,
    min_v_t,
    's-',
    label='Min velocity'
)

ax2.set_xlabel('Safe time headway T (s)')
ax2.set_ylabel('Velocity (m/s)')
ax2.set_title('Experiment 3 : T Sweep')

ax2.grid(True, alpha=0.3)
ax2.legend()

# =========================================================
# レイアウト調整
# =========================================================

plt.tight_layout()

# =========================================================
# PNG 保存
# =========================================================

graph_path = os.path.join(
    BASE_DIR,
    'experiments_2_and_3.png'
)

plt.savefig(
    graph_path,
    dpi=150
)

print(f'\nGraph saved:\n{graph_path}')

# =========================================================
# 表示
# =========================================================

plt.show()