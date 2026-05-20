# -*- coding: utf-8 -*-
"""
環状道路シミュレーション：実験4
渋滞波の成長率 gamma がゼロになる条件を L / N / T で探す

必要ファイル：
    traffic_jam_mas.py

このスクリプトで行うこと：
    - 成長率 gamma の測定（弱い摂動で指数成長の勾配をフィット）
    - 実験4-A：L（道路長）スイープ
    - 実験4-B：N（車両台数）スイープ
    - 実験4-C：T（安全車間時間）スイープ
    - 各スイープで gamma=0 のゼロ交差（臨界値 L*, N*, T*）を算出
    - gamma vs 変数 のグラフ（境界線つき）
    - gamma vs 密度 の重ね描き（L と N が密度に集約できるかの検証）
    - PNG 保存・npz データ保存

保存先：
    この .py ファイルと同じフォルダ
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from traffic_jam_mas import IDMParams, SimParams, run_simulation

# =========================================================
# 保存先設定
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("Save directory:")
print(BASE_DIR)

# =========================================================
# 成長率測定の設定（第0段階の検証で決定・固定）
# =========================================================
#
# 第0段階（step0_check）で log-振幅プロットを確認した結果：
#   - step 0〜100 付近は初期摂動が崩れる過渡。フィットに含めない。
#   - step 100〜500 で log-振幅がほぼ完全な直線（R^2 ≈ 0.99〜1.00）。
#   - この区間なら不安定側で正・安定側で負の gamma が安定して出る。
# よってフィット区間を [100, 500] に固定する。
#
FIT_START = 100
FIT_END = 500
N_STEPS = 800          # フィット区間 + 余裕
PERTURBATION = 'small' # 成長率測定では弱い摂動（1台5%減速）を必ず使う

IDM_KEYS = ['v0', 'T', 'a_max', 'b_comf', 'delta', 's0', 'l_veh']


# =========================================================
# 成長率 gamma の測定
# =========================================================

def measure_growth_rate(v_history, dt, fit_start=FIT_START, fit_end=FIT_END):
    """
    速度履歴から渋滞波の成長率 gamma を測る。

    波の振幅 = 全車速度の標準偏差。
    指数成長 A(t) ∝ exp(gamma t) は log を取ると直線になるので、
    log(振幅) を時間に対して直線フィットした傾きが gamma。

    引数:
        v_history: (n_steps, n_cars) の速度履歴
        dt: 時間刻み（秒）
        fit_start, fit_end: フィットに使うステップ範囲
    戻り値:
        gamma (1/秒), r2 (フィットの決定係数)
    """
    amp = np.maximum(v_history.std(axis=1), 1e-12)  # log(0) 回避
    log_amp = np.log(amp)

    t = np.arange(fit_start, fit_end) * dt
    y = log_amp[fit_start:fit_end]

    slope, intercept = np.polyfit(t, y, 1)

    # 決定係数 R^2（直線性のチェック用）
    pred = slope * t + intercept
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return slope, r2


# =========================================================
# 共通スイープ関数
# =========================================================

def run_growth_sweep(varying_param, values, fixed_params=None):
    """
    1変数を動かして各点の成長率 gamma を測る。

    引数:
        varying_param: 動かすパラメータ名（'ring_length' / 'n_cars' / 'T' など）
        values: 走査する値のリスト
        fixed_params: 固定パラメータの dict
    戻り値:
        gammas (np.array), r2s (np.array)
    """
    fixed_params = fixed_params or {}
    gammas = []
    r2s = []

    for val in values:
        idm_kwargs = {}
        sim_kwargs = {}

        # 固定パラメータを IDM 用 / Sim 用に振り分け
        for k, v in fixed_params.items():
            if k in IDM_KEYS:
                idm_kwargs[k] = v
            else:
                sim_kwargs[k] = v

        # 動かすパラメータを振り分け
        if varying_param in IDM_KEYS:
            idm_kwargs[varying_param] = val
        else:
            sim_kwargs[varying_param] = val

        # 成長率測定の固定設定
        sim_kwargs['perturbation'] = PERTURBATION
        sim_kwargs.setdefault('n_steps', N_STEPS)

        idm = IDMParams(**idm_kwargs)
        sim = SimParams(**sim_kwargs)

        result = run_simulation(idm=idm, sim=sim, record_positions=True)
        gamma, r2 = measure_growth_rate(result['v_history'], sim.dt)

        gammas.append(gamma)
        r2s.append(r2)

        flag = '' if r2 > 0.9 else '  <-- R2 low, check fit'
        print(
            f'{varying_param} = {val:7.3g} '
            f'| gamma = {gamma:+.5f} '
            f'| R2 = {r2:.4f}{flag}'
        )

    return np.array(gammas), np.array(r2s)


# =========================================================
# ゼロ交差（臨界値）の算出
# =========================================================

def find_zero_crossing(values, gammas):
    """
    gamma の符号が変わる点を線形補間で求める。
    複数の交差がありうるので全て返す。
    """
    values = np.asarray(values, dtype=float)
    gammas = np.asarray(gammas, dtype=float)

    crossings = []
    for i in range(len(gammas) - 1):
        g0, g1 = gammas[i], gammas[i + 1]
        if g0 == 0.0:
            crossings.append(values[i])
        elif g0 * g1 < 0.0:  # 符号が変わった
            x = values[i] + (values[i + 1] - values[i]) * (-g0) / (g1 - g0)
            crossings.append(x)
    return crossings


# =========================================================
# 1パネル分の描画ヘルパー
# =========================================================

def plot_panel(ax, values, gammas, xlabel, title, crossings, cross_unit=''):
    """gamma vs 変数 の1パネルを描く。安定/不安定ゾーンを色分け。"""
    values = np.asarray(values, dtype=float)

    # 安定/不安定ゾーンの背景色
    ax.axhspan(0, max(gammas.max() * 1.2, 1e-3),
               color='C3', alpha=0.07)
    ax.axhspan(min(gammas.min() * 1.2, -1e-3), 0,
               color='C0', alpha=0.07)

    ax.plot(values, gammas, 'o-', color='C1', zorder=3)
    ax.axhline(0, color='k', lw=1.2)  # gamma = 0 の境界線

    # ゼロ交差点
    for xc in crossings:
        ax.axvline(xc, color='k', ls='--', alpha=0.7)
        ax.annotate(f'{xc:.2f}{cross_unit}',
                    xy=(xc, 0),
                    xytext=(6, 8), textcoords='offset points',
                    fontsize=10, fontweight='bold')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'Growth rate $\gamma$ (1/s)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # ゾーン注記
    ax.text(0.02, 0.95, 'unstable (jam grows)', transform=ax.transAxes,
            color='C3', fontsize=9, va='top')
    ax.text(0.02, 0.05, 'stable (jam decays)', transform=ax.transAxes,
            color='C0', fontsize=9, va='bottom')


# =========================================================
# 実験4-A：L（道路長）スイープ
# =========================================================

print('\n==============================')
print('Experiment 4-A : L (ring length) Sweep')
print('==============================')

L_values = np.arange(250.0, 601.0, 25.0)  # 250〜600 m、25 m 刻み

gamma_L, r2_L = run_growth_sweep(
    varying_param='ring_length',
    values=L_values,
    fixed_params={'n_cars': 28},
)

L_star = find_zero_crossing(L_values, gamma_L)
print(f'L* (gamma=0 crossing) = {L_star}')

np.savez(
    os.path.join(BASE_DIR, 'exp4A_data.npz'),
    L=L_values, gamma=gamma_L, r2=r2_L,
    L_star=np.array(L_star),
)


# =========================================================
# 実験4-B：N（車両台数）スイープ
# =========================================================

print('\n==============================')
print('Experiment 4-B : N (number of cars) Sweep')
print('==============================')

N_values = np.arange(10, 46)  # 10〜45 台、1 台刻み

gamma_N, r2_N = run_growth_sweep(
    varying_param='n_cars',
    values=N_values,
    fixed_params={'ring_length': 350.0},
)

N_star = find_zero_crossing(N_values, gamma_N)
print(f'N* (gamma=0 crossing) = {N_star}')

np.savez(
    os.path.join(BASE_DIR, 'exp4B_data.npz'),
    N=N_values, gamma=gamma_N, r2=r2_N,
    N_star=np.array(N_star),
)


# =========================================================
# 実験4-C：T（安全車間時間）スイープ
# =========================================================

print('\n==============================')
print('Experiment 4-C : T (safe time headway) Sweep')
print('==============================')

T_values = np.arange(0.6, 2.21, 0.1)  # 0.6〜2.2 s、0.1 s 刻み

gamma_T, r2_T = run_growth_sweep(
    varying_param='T',
    values=T_values,
    fixed_params={'n_cars': 28, 'ring_length': 350.0},
)

T_star = find_zero_crossing(T_values, gamma_T)
print(f'T* (gamma=0 crossing) = {T_star}')

np.savez(
    os.path.join(BASE_DIR, 'exp4C_data.npz'),
    T=T_values, gamma=gamma_T, r2=r2_T,
    T_star=np.array(T_star),
)


# =========================================================
# グラフ1：gamma vs L / N / T（3パネル）
# =========================================================

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

plot_panel(ax1, L_values, gamma_L,
           'Ring length L (m)',
           'Exp 4-A : gamma vs L  (N=28 fixed)',
           L_star, cross_unit=' m')

plot_panel(ax2, N_values, gamma_N,
           'Number of cars N',
           'Exp 4-B : gamma vs N  (L=350 m fixed)',
           N_star, cross_unit='')

plot_panel(ax3, T_values, gamma_T,
           'Safe time headway T (s)',
           'Exp 4-C : gamma vs T  (N=28, L=350 m fixed)',
           T_star, cross_unit=' s')

plt.tight_layout()
graph1_path = os.path.join(BASE_DIR, 'exp4_growth_rate.png')
plt.savefig(graph1_path, dpi=150)
print(f'\nGraph saved:\n{graph1_path}')


# =========================================================
# グラフ2：gamma vs 密度（L と N を密度軸で重ね描き）
# =========================================================
#
# L スイープは N=28 固定なので密度 rho = 28 / L
# N スイープは L=350 固定なので密度 rho = N / 350
# 渋滞が密度だけで決まるなら、この2本は重なるはず。
#
rho_from_L = 28.0 / L_values
rho_from_N = N_values / 350.0

fig2, ax = plt.subplots(figsize=(8, 6))

ax.axhspan(0, max(gamma_L.max(), gamma_N.max()) * 1.2,
           color='C3', alpha=0.07)
ax.axhspan(min(gamma_L.min(), gamma_N.min()) * 1.2, 0,
           color='C0', alpha=0.07)

ax.plot(rho_from_L, gamma_L, 'o-', color='C0',
        label='from L sweep (N=28 fixed)')
ax.plot(rho_from_N, gamma_N, 's-', color='C3',
        label='from N sweep (L=350 fixed)')
ax.axhline(0, color='k', lw=1.2)

ax.set_xlabel(r'Density $\rho = N / L$  (cars/m)')
ax.set_ylabel(r'Growth rate $\gamma$ (1/s)')
ax.set_title('gamma vs density : do L-sweep and N-sweep collapse?')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
graph2_path = os.path.join(BASE_DIR, 'exp4_density_collapse.png')
plt.savefig(graph2_path, dpi=150)
print(f'Graph saved:\n{graph2_path}')


# =========================================================
# 結果サマリーの出力
# =========================================================

print('\n==============================')
print('SUMMARY : critical conditions (gamma = 0)')
print('==============================')
print(f'L* = {L_star}  (m)   [N=28 fixed]')
print(f'N* = {N_star}        [L=350 m fixed]')
print(f'T* = {T_star}  (s)   [N=28, L=350 m fixed]')

if len(L_star) > 0:
    rho_c_from_L = 28.0 / L_star[0]
    print(f'\ncritical density from L* : rho_c = 28 / {L_star[0]:.1f} '
          f'= {rho_c_from_L:.4f} /m')
if len(N_star) > 0:
    rho_c_from_N = N_star[0] / 350.0
    print(f'critical density from N* : rho_c = {N_star[0]:.2f} / 350 '
          f'= {rho_c_from_N:.4f} /m')

print('\n(Reference - prior real-vehicle experiments)')
print('  Sugiyama 2008 : 0.096 /m')
print('  Tadaki 2013   : 0.08-0.09 /m')
print('  Stern 2018    : 0.081 /m')

plt.show()
