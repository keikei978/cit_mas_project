# -*- coding: utf-8 -*-
"""
実験5：ドライバー混在実験
「全員が慎重」でなく「慎重な人が何割いれば渋滞が消えるか」を調べる。

慎重なドライバー（車間時間 T_cautious）と
せっかちなドライバー（T_aggressive）を1つのリングに混在させ、
慎重派の割合 f を 0 → 1 でスイープし、各 f で成長率 gamma を測る。
gamma がゼロを切る f が「渋滞を消すのに必要な慎重派の最低比率」。

設計方針：
    traffic_jam_mas.py は改変しない。
    車ごとに T を変えるには step() 内のソート処理に T 配列を
    通す必要があるため、車別 T 対応版の step をこのファイルに持つ。
    IDM の数式は traffic_jam_mas.py と完全に同一。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from traffic_jam_mas import IDMParams, SimParams

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 成長率測定の設定（実験4と同一）
FIT_START = 100
FIT_END = 500
N_STEPS = 800


# =========================================================
# 車別 T 対応版 step（IDM 数式は本体と同一）
# =========================================================

def step_heterogeneous(x, v, T_array, idm: IDMParams, sim: SimParams):
    """
    1ステップ進める。車ごとに異なる T_array を使える版。

    本体 traffic_jam_mas.step() との違いは T だけ。
    重要：argsort でソートするとき T_array も同じ順序で並べ替える。
          そうしないと「位置順で並べた車」に「元の順序のT」を
          当ててしまい、別の車のTを使うバグになる。
    """
    order = np.argsort(x)
    x_sorted = x[order]
    v_sorted = v[order]
    T_sorted = T_array[order]          # ← T も同じ順序でソート

    x_lead = np.roll(x_sorted, -1)
    v_lead = np.roll(v_sorted, -1)

    gap = (x_lead - x_sorted) % sim.ring_length - idm.l_veh
    gap = np.maximum(gap, 0.1)

    sqrt_ab = np.sqrt(idm.a_max * idm.b_comf)
    dv = v_sorted - v_lead
    s_star = idm.s0 + np.maximum(
        0.0, v_sorted * T_sorted + (v_sorted * dv) / (2 * sqrt_ab)
    )                                   # ← idm.T ではなく T_sorted
    accel = idm.a_max * (
        1 - (v_sorted / idm.v0) ** idm.delta - (s_star / gap) ** 2
    )

    v_sorted = v_sorted + accel * sim.dt
    v_sorted = np.maximum(v_sorted, 0.0)
    x_sorted = (x_sorted + v_sorted * sim.dt) % sim.ring_length

    x_new = np.empty_like(x)
    v_new = np.empty_like(v)
    x_new[order] = x_sorted
    v_new[order] = v_sorted
    return x_new, v_new


def run_mixed_simulation(T_array, idm: IDMParams, sim: SimParams):
    """車別 T でシミュレーションを1回走らせ、速度履歴を返す。"""
    x = np.arange(sim.n_cars, dtype=float) * sim.ring_length / sim.n_cars
    v = np.full(sim.n_cars, idm.v0 * 0.6)
    if sim.perturbation == 'stop_one':
        v[0] = 0.0
    elif sim.perturbation == 'small':
        v[0] *= 0.95

    v_history = np.zeros((sim.n_steps, sim.n_cars))
    for s in range(sim.n_steps):
        x, v = step_heterogeneous(x, v, T_array, idm, sim)
        v_history[s] = v
    return v_history


def measure_growth_rate(v_history, dt, fit_start=FIT_START, fit_end=FIT_END):
    """log-振幅の傾きとして成長率 gamma を測る（実験4と同一）。"""
    amp = np.maximum(v_history.std(axis=1), 1e-12)
    t = np.arange(fit_start, fit_end) * dt
    y = np.log(amp)[fit_start:fit_end]
    slope, _ = np.polyfit(t, y, 1)
    return slope


# =========================================================
# 混在の作り方
# =========================================================

def make_T_array(n_cars, frac_cautious, T_cautious, T_aggressive,
                 arrangement='random', rng=None):
    """
    慎重派 frac_cautious の割合で T 配列を作る。

    arrangement:
        'random'  … 慎重派とせっかち派をランダムに配置
        'blocked' … 慎重派を1か所に固める（配置の影響を見る用）
    """
    n_cautious = int(round(n_cars * frac_cautious))
    T_array = np.full(n_cars, T_aggressive, dtype=float)
    if n_cautious > 0:
        if arrangement == 'random':
            rng = rng or np.random.default_rng()
            idx = rng.choice(n_cars, size=n_cautious, replace=False)
        else:  # blocked
            idx = np.arange(n_cautious)
        T_array[idx] = T_cautious
    return T_array


# =========================================================
# 実験5本体：慎重派の割合をスイープ
# =========================================================

# 設定
N_CARS = 28               # 臨界付近に固定（実験3・4と同じ）
T_CAUTIOUS = 2.0          # 慎重なドライバーの車間時間
T_AGGRESSIVE = 0.8        # せっかちなドライバーの車間時間
N_TRIALS = 20             # ランダム配置の試行回数（平均を取る）

# 確認：両端が実験4の結果と整合するか
#   f=0 → 全員せっかち（T=0.8）→ 強い渋滞（gamma>0）のはず
#   f=1 → 全員慎重（T=2.0）  → 渋滞消える（gamma<0）のはず

frac_values = np.arange(0.0, 1.001, 0.05)  # 0%〜100%、5%刻み

print('Experiment 5 : Mixed-driver sweep')
print(f'  N={N_CARS}, T_cautious={T_CAUTIOUS}, T_aggressive={T_AGGRESSIVE}')
print(f'  averaging over {N_TRIALS} random arrangements per point\n')

gamma_mean = []
gamma_std = []

for f in frac_values:
    idm = IDMParams()  # T はここでは使われない（配列で渡す）
    sim = SimParams(n_cars=N_CARS, ring_length=350.0,
                    perturbation='small', n_steps=N_STEPS)
    trials = []
    for trial in range(N_TRIALS):
        rng = np.random.default_rng(trial)  # 再現性のため試行ごと固定
        T_array = make_T_array(N_CARS, f, T_CAUTIOUS, T_AGGRESSIVE,
                               arrangement='random', rng=rng)
        v_hist = run_mixed_simulation(T_array, idm, sim)
        trials.append(measure_growth_rate(v_hist, sim.dt))
    g_mean = float(np.mean(trials))
    g_std = float(np.std(trials))
    gamma_mean.append(g_mean)
    gamma_std.append(g_std)
    print(f'  cautious fraction = {f:.2f} '
          f'| gamma = {g_mean:+.5f} +/- {g_std:.5f}')

gamma_mean = np.array(gamma_mean)
gamma_std = np.array(gamma_std)


# ゼロ交差（必要な慎重派の最低比率）
def find_zero_crossing(values, gammas):
    values, gammas = np.asarray(values, float), np.asarray(gammas, float)
    out = []
    for i in range(len(gammas) - 1):
        g0, g1 = gammas[i], gammas[i + 1]
        if g0 == 0:
            out.append(values[i])
        elif g0 * g1 < 0:
            out.append(values[i] + (values[i+1]-values[i])*(-g0)/(g1-g0))
    return out


f_star = find_zero_crossing(frac_values, gamma_mean)
print(f'\nf* (minimum cautious fraction) = {f_star}')

# データ保存
np.savez(os.path.join(BASE_DIR, 'exp5_data.npz'),
         frac=frac_values, gamma_mean=gamma_mean, gamma_std=gamma_std,
         f_star=np.array(f_star),
         N_cars=N_CARS, T_cautious=T_CAUTIOUS, T_aggressive=T_AGGRESSIVE)


# =========================================================
# 描画
# =========================================================

fig, ax = plt.subplots(figsize=(9, 6))

# 安定/不安定ゾーン
ax.axhspan(0, max(gamma_mean.max()*1.2, 1e-3), color='C3', alpha=0.07)
ax.axhspan(min(gamma_mean.min()*1.2, -1e-3), 0, color='C0', alpha=0.07)

# 試行ばらつきを帯で表示
ax.fill_between(frac_values * 100,
                gamma_mean - gamma_std, gamma_mean + gamma_std,
                color='C1', alpha=0.25, label='spread over arrangements')
ax.plot(frac_values * 100, gamma_mean, 'o-', color='C1',
        label='mean growth rate', zorder=3)
ax.axhline(0, color='k', lw=1.2)

# ゼロ交差点
for fc in f_star:
    ax.axvline(fc * 100, color='k', ls='--', alpha=0.7)
    ax.annotate(f'{fc*100:.0f}%',
                xy=(fc * 100, 0), xytext=(8, 10),
                textcoords='offset points',
                fontsize=12, fontweight='bold')

ax.set_xlabel('Fraction of cautious drivers (%)')
ax.set_ylabel(r'Growth rate $\gamma$ (1/s)')
ax.set_title('Exp 5 : How many cautious drivers are needed?\n'
             f'N={N_CARS} cars,  cautious T={T_CAUTIOUS}s / '
             f'aggressive T={T_AGGRESSIVE}s')
ax.text(0.97, 0.95, 'JAM (unstable)', transform=ax.transAxes,
        ha='right', va='top', color='darkred', fontsize=11, fontweight='bold')
ax.text(0.03, 0.05, 'FREE FLOW (stable)', transform=ax.transAxes,
        ha='left', va='bottom', color='darkblue', fontsize=11, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(BASE_DIR, 'exp5_mixed_drivers.png')
plt.savefig(out_path, dpi=150)
print(f'\nSaved: {out_path}')
plt.show()
