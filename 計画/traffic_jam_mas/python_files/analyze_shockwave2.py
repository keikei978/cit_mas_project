# -*- coding: utf-8 -*-
"""
安全車間時間による
時空間図比較

比較：
    - T=1.2
    - T=1.5
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from traffic_jam_mas import (
    IDMParams,
    SimParams,
    run_simulation
)

# =========================================================
# 日本語フォント
# =========================================================

plt.rcParams['font.family'] = 'MS Gothic'

# =========================================================
# 保存先
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# =========================================================
# シミュレーション条件
# =========================================================

sim = SimParams(
    ring_length=350.0,
    n_cars=28,
    dt=0.25,
    n_steps=800,
    perturbation='stop_one'
)

# =========================================================
# T=1.2
# =========================================================

idm_12 = IDMParams(
    v0=16.0,
    T=1.2,
    a_max=0.8,
    b_comf=1.5,
    delta=4,
    s0=2.0,
    l_veh=5.0
)

result_12 = run_simulation(
    idm=idm_12,
    sim=sim,
    record_positions=True
)

# =========================================================
# T=1.5
# =========================================================

idm_15 = IDMParams(
    v0=16.0,
    T=1.5,
    a_max=0.8,
    b_comf=1.5,
    delta=4,
    s0=2.0,
    l_veh=5.0
)

result_15 = run_simulation(
    idm=idm_15,
    sim=sim,
    record_positions=True
)

# =========================================================
# データ取得
# =========================================================

x_hist_12 = result_12['x_history']
v_hist_12 = result_12['v_history']

x_hist_15 = result_15['x_history']
v_hist_15 = result_15['v_history']

# =========================================================
# 描画
# =========================================================

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(14, 6)
)

# =========================================================
# T=1.2
# =========================================================

for t in range(sim.n_steps):

    scatter1 = ax1.scatter(
        x_hist_12[t],
        np.full(
            sim.n_cars,
            t * sim.dt
        ),
        c=v_hist_12[t],
        cmap='RdYlGn',
        vmin=0,
        vmax=idm_12.v0,
        s=5
    )

ax1.set_xlabel(
    '環状道路上の位置 (m)'
)

ax1.set_ylabel(
    '時間 (s)'
)

ax1.set_title(
    '時空間図 (T=1.2)'
)

# =========================================================
# T=1.5
# =========================================================

for t in range(sim.n_steps):

    scatter2 = ax2.scatter(
        x_hist_15[t],
        np.full(
            sim.n_cars,
            t * sim.dt
        ),
        c=v_hist_15[t],
        cmap='RdYlGn',
        vmin=0,
        vmax=idm_15.v0,
        s=5
    )

ax2.set_xlabel(
    '環状道路上の位置 (m)'
)

ax2.set_ylabel(
    '時間 (s)'
)

ax2.set_title(
    '時空間図 (T=1.5)'
)

# =========================================================
# カラーバー
# =========================================================

cbar = fig.colorbar(
    scatter2,
    ax=[ax1, ax2],
    shrink=0.8
)

cbar.set_label(
    '速度 (m/s)'
)

# =========================================================
# レイアウト
# =========================================================

plt.tight_layout()

# =========================================================
# 保存
# =========================================================

save_path = os.path.join(
    BASE_DIR,
    'space_time_comparison.png'
)

plt.savefig(
    save_path,
    dpi=150
)

print('\nSaved:')
print(save_path)

# =========================================================
# 表示
# =========================================================

plt.show()