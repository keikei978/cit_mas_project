# -*- coding: utf-8 -*-
"""
実験4 拡張：N x T の2次元安定性マップ

台数 N と 車間時間 T を同時に動かし、各 (N, T) で成長率 gamma を測る。
gamma の符号で平面を「渋滞ゾーン / 自由流ゾーン」に塗り分け、
gamma = 0 の境界線（綱引きの均衡線）を描く。

必要ファイル：traffic_jam_mas.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from traffic_jam_mas import IDMParams, SimParams, run_simulation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 成長率測定の設定（実験4本体と同じ）
FIT_START = 100
FIT_END = 500
N_STEPS = 800


def measure_growth_rate(v_history, dt, fit_start=FIT_START, fit_end=FIT_END):
    """log-振幅の傾きとして成長率 gamma を測る。"""
    amp = np.maximum(v_history.std(axis=1), 1e-12)
    log_amp = np.log(amp)
    t = np.arange(fit_start, fit_end) * dt
    y = log_amp[fit_start:fit_end]
    slope, _ = np.polyfit(t, y, 1)
    return slope


# =========================================================
# 2次元グリッドのスイープ
# =========================================================

N_values = np.arange(12, 41, 1)        # 台数 12〜40
T_values = np.arange(0.6, 2.21, 0.1)   # 車間時間 0.6〜2.2 s

gamma_grid = np.zeros((len(T_values), len(N_values)))  # 行=T, 列=N

print(f'Sweeping {len(N_values)} x {len(T_values)} = '
      f'{len(N_values) * len(T_values)} cases...')

for i, T in enumerate(T_values):
    for j, N in enumerate(N_values):
        idm = IDMParams(T=T)
        sim = SimParams(n_cars=int(N), ring_length=350.0,
                        perturbation='small', n_steps=N_STEPS)
        result = run_simulation(idm=idm, sim=sim, record_positions=True)
        gamma_grid[i, j] = measure_growth_rate(result['v_history'], sim.dt)
    print(f'  T = {T:.1f} done')

# データ保存
np.savez(os.path.join(BASE_DIR, 'exp4_NT_map.npz'),
         N=N_values, T=T_values, gamma=gamma_grid)

# =========================================================
# 描画
# =========================================================

fig, ax = plt.subplots(figsize=(10, 7))

# gamma の値を色で表示（赤=不安定/正、青=安定/負、白=ゼロ付近）
vmax = np.abs(gamma_grid).max()
mesh = ax.pcolormesh(N_values, T_values, gamma_grid,
                     cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                     shading='nearest')
cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label(r'Growth rate $\gamma$ (1/s)')

# gamma = 0 の境界線（綱引きの均衡線）
cs = ax.contour(N_values, T_values, gamma_grid,
                levels=[0.0], colors='black', linewidths=2.5)
ax.clabel(cs, fmt={0.0: 'gamma = 0'}, fontsize=11)

ax.set_xlabel('Number of cars N')
ax.set_ylabel('Safe time headway T (s)')
ax.set_title('Exp 4 : N x T stability map\n'
             'red = jam grows (unstable),  blue = jam decays (stable)')

# ゾーン注記
ax.text(0.97, 0.95, 'JAM ZONE', transform=ax.transAxes,
        ha='right', va='top', fontsize=12, fontweight='bold',
        color='darkred')
ax.text(0.03, 0.05, 'FREE-FLOW ZONE', transform=ax.transAxes,
        ha='left', va='bottom', fontsize=12, fontweight='bold',
        color='darkblue')

plt.tight_layout()
out_path = os.path.join(BASE_DIR, 'exp4_NT_map.png')
plt.savefig(out_path, dpi=150)
print(f'\nSaved: {out_path}')
plt.show()
