# -*- coding: utf-8 -*-
"""
安全車間時間Tと渋滞発生の関係

内容：
    - Tを変化
    - 平均速度
    - 最低速度
    - 時空間図

を比較する
"""

import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'MS Gothic'

from traffic_jam_mas import (
    IDMParams,
    SimParams,
    run_simulation
)

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
# T sweep
# =========================================================

T_values = np.arange(
    0.6,
    2.1,
    0.1
)

mean_velocities = []
min_velocities = []

# =========================================================
# 実験
# =========================================================

print('Running T sweep...')

for T in T_values:

    idm = IDMParams(
        v0=16.0,
        T=T,
        a_max=0.8,
        b_comf=1.5,
        delta=4,
        s0=2.0,
        l_veh=5.0
    )

    result = run_simulation(
        idm=idm,
        sim=sim,
        record_positions=False
    )

    # 最後100step平均
    mean_v = np.mean(
        result['avg_v'][-100:]
    )

    min_v = np.mean(
        result['min_v'][-100:]
    )

    mean_velocities.append(mean_v)
    min_velocities.append(min_v)

    print(
        f'T={T:.1f} '
        f'Mean={mean_v:.2f} '
        f'Min={min_v:.2f}'
    )

print('Done.')

# =========================================================
# グラフ
# =========================================================

fig, ax = plt.subplots(
    figsize=(8, 6)
)

ax.plot(
    T_values,
    mean_velocities,
    marker='o',
    label='平均速度'
)

ax.plot(
    T_values,
    min_velocities,
    marker='s',
    label='最低速度'
)

ax.set_xlabel(
    '安全車間時間 T (s)'
)

ax.set_ylabel(
    '速度 (m/s)'
)

ax.set_title(
    '安全車間時間と渋滞の関係'
)

ax.grid(True)

ax.legend()

# =========================================================
# 保存
# =========================================================

save_path = os.path.join(
    BASE_DIR,
    'T_sweep_experiment.png'
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