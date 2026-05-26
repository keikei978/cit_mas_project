"""
環状道路シミュレーションのアニメーション可視化（多車線対応版）
- 各車線を同心楕円として描画
- 角度計算に車線ごとの実効リング長を使用
- 車線変更時に TRANSITION_FRAMES かけて半径をなめらかに補間
- 車の色 = 速度（RdYlGn カラーマップ）
"""
 
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
 
from traffic_jam_mas import (
    IDMParams, SimParams, MOBILParams,
    run_simulation, _lane_ring_length, _lane_v0
)
 
 
# =========================================================
# 可視化設定
# =========================================================
 
CMAP              = 'RdYlGn'
ELLIPSE_RATIO     = 0.70
TRANSITION_FRAMES = 10
 
_LANE_RADII = [0.55, 0.75, 0.95, 1.10]
 
 
def _lane_radius(lane_idx: int) -> float:
    if lane_idx < len(_LANE_RADII):
        return _LANE_RADII[lane_idx]
    return 0.55 + lane_idx * 0.20
 
 
def to_xy(s, ring_lengths_per_car, radii):
    """
    弧長 s を XY 座標に変換。
    ring_lengths_per_car: 各車のリング長（配列）
    radii              : 各車の描画半径（配列）
    """
    theta = (s / ring_lengths_per_car) * 2 * np.pi
    return radii * np.cos(theta), radii * ELLIPSE_RATIO * np.sin(theta)
 
 
# =========================================================
# 事前計算
# =========================================================
 
def _compute_display_radii(lane_hist, n_steps, n_cars):
    """車線変更を TRANSITION_FRAMES かけて線形補間した描画半径を返す"""
    raw_r     = np.vectorize(_lane_radius)(lane_hist).astype(float)
    display_r = raw_r.copy()
 
    for ci in range(n_cars):
        frame = 1
        while frame < n_steps:
            if abs(raw_r[frame, ci] - raw_r[frame - 1, ci]) > 1e-6:
                old_r = display_r[frame - 1, ci]
                new_r = raw_r[frame, ci]
                end   = min(frame + TRANSITION_FRAMES, n_steps)
                for t_off, f in enumerate(range(frame, end)):
                    if f > frame and abs(raw_r[f, ci] - new_r) > 1e-6:
                        break
                    display_r[f, ci] = old_r + (new_r - old_r) * (t_off + 1) / TRANSITION_FRAMES
                frame = end
            else:
                frame += 1
 
    return display_r
 
 
def _compute_ring_lengths_hist(lane_hist, n_steps, n_cars, sim: SimParams):
    """各フレーム・各車の実効リング長 (n_steps, n_cars)"""
    rl = np.array([_lane_ring_length(i, sim) for i in range(sim.n_lanes)])
    return rl[lane_hist]   # fancy indexing
 
 
# =========================================================
# アニメーション生成
# =========================================================
 
def make_animation(idm: IDMParams, sim: SimParams,
                   mobil: MOBILParams = None, save_gif: bool = True):
 
    print('Running simulation...', end='', flush=True)
    result = run_simulation(idm, sim, mobil=mobil, record_positions=True)
    print('Done.')
 
    x_hist    = result['x_history']
    v_hist    = result['v_history']
    lane_hist = result.get(
        'lane_history',
        np.zeros((sim.n_steps, sim.n_cars), dtype=int)
    )
    avg_v = result['avg_v']
    min_v = result['min_v']
 
    print('Pre-computing display data...', end='', flush=True)
    display_r = _compute_display_radii(lane_hist, sim.n_steps, sim.n_cars)
    rl_hist   = _compute_ring_lengths_hist(lane_hist, sim.n_steps, sim.n_cars, sim)
    
    # 各フレーム・各車線の平均速度を計算
    avg_v_per_lane = np.zeros((sim.n_steps, sim.n_lanes))
    for f in range(sim.n_steps):
        for lane_idx in range(sim.n_lanes):
            lane_mask = (lane_hist[f] == lane_idx)
            if lane_mask.any():
                avg_v_per_lane[f, lane_idx] = v_hist[f, lane_mask].mean()
            else:
                avg_v_per_lane[f, lane_idx] = 0.0
    
    print('Done.')

    lane_counts = np.array([
        np.bincount(lane_hist[f], minlength=sim.n_lanes)
        for f in range(sim.n_steps)
    ])
    lane_changes = np.zeros(sim.n_steps, dtype=int)
    if sim.n_steps > 1:
        lane_changes[1:] = np.sum(lane_hist[1:] != lane_hist[:-1], axis=1)
    
    # 累積車線変更数
    lane_changes_cumsum = np.cumsum(lane_changes)
 
    # ── 描画準備 ──
    fig = plt.figure(figsize=(16, 12))
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)
 
    max_r  = _lane_radius(sim.n_lanes - 1)
    margin = 0.25
    ax1.set_xlim(-(max_r + margin), max_r + margin)
    ax1.set_ylim(-(max_r + margin) * ELLIPSE_RATIO, (max_r + margin))
    ax1.set_aspect('auto')
    ax1.axis('off')
 
    # 同心楕円（車線ガイド）と車線情報ラベル
    theta_ring = np.linspace(0, 2 * np.pi, 360)
    for lane_idx in range(sim.n_lanes):
        r    = _lane_radius(lane_idx)
        v0_l = _lane_v0(lane_idx, idm, sim)
        rl   = _lane_ring_length(lane_idx, sim)
        ax1.plot(
            r * np.cos(theta_ring),
            r * ELLIPSE_RATIO * np.sin(theta_ring),
            color='lightgray', linewidth=1.5, zorder=0
        )
        ax1.text(
            0, r * ELLIPSE_RATIO + 0.03,
            f'Lane {lane_idx}  ({rl:.0f} m / v₀={v0_l:.1f} m/s)',
            ha='center', va='bottom', fontsize=7.5, color='gray'
        )
 
    # scatter（全車を1オブジェクトで管理）
    xs0, ys0 = to_xy(x_hist[0], rl_hist[0], display_r[0])
    sc = ax1.scatter(
        xs0, ys0, c=v_hist[0],
        cmap=CMAP, vmin=0, vmax=idm.v0 + sim.v0_lane_boost * (sim.n_lanes - 1),
        s=45, zorder=2, edgecolors='none'
    )
 
    # 車線変更した車を黒枠で強調
    sc_hi = ax1.scatter(
        [], [], facecolors='none', edgecolors='black',
        linewidths=1.5, s=130, zorder=3
    )
 
    # カラーバー（外側車線の最高速度まで対応）
    vmax = idm.v0 + sim.v0_lane_boost * (sim.n_lanes - 1)
    sm   = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1, fraction=0.03, pad=0.02)
    cbar.set_label('Speed (m/s)', fontsize=9)
 
    info_text = ax1.text(
        -(max_r + margin - 0.02),
        (max_r + margin - 0.04) * ELLIPSE_RATIO,
        '', ha='left', va='top', fontsize=8, color='#333333',
        fontfamily='monospace'
    )
 
    # 右上：速度グラフ
    ax2.set_xlim(0, sim.n_steps)
    ax2.set_ylim(0, vmax * 1.1)
    ax2.set_xlabel('time Step (s)')
    ax2.set_ylabel('Sokudo (m/s)')
    ax2.set_title('Zentai Sokudo')
    line_avg, = ax2.plot([], [], label='Heikin', color='royalblue', linewidth=1.5)
    line_min, = ax2.plot([], [], label='Saishoku',     color='tomato',    linewidth=1.5)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 左下：車線別平均速度グラフ
    ax3.set_xlim(0, sim.n_steps)
    ax3.set_ylim(0, vmax * 1.1)
    ax3.set_xlabel('time Step (s)')
    ax3.set_ylabel('Heikin Sokudo (m/s)')
    ax3.set_title('Shasen-betsu Heikin Sokudo')
    
    # 各車線用の線を作成（色分け）
    lane_colors = ['royalblue', 'orangered', 'forestgreen', 'purple', 'brown']
    lane_lines = []
    for lane_idx in range(sim.n_lanes):
        color = lane_colors[lane_idx % len(lane_colors)]
        line, = ax3.plot([], [], label=f'Shasen {lane_idx}', color=color, linewidth=1.5)
        lane_lines.append(line)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 右下：累積車線変更数と平均速度の相関グラフ
    ax4.set_xlim(0, sim.n_steps)
    ax4.set_xlabel('time Step (s)')
    ax4.set_ylabel('goukei Shasen Henkou (Kai)')
    ax4.set_title('goukei Shasen Henkou vs Heikin Sokudo')
    
    # 左軸：累積車線変更数
    ax4_twin = ax4.twinx()
    ax4_twin.set_ylabel('Heikin Sokudo (m/s)', color='steelblue')
    ax4_twin.tick_params(axis='y', labelcolor='steelblue')
    ax4_twin.set_ylim(0, vmax * 1.1)
    
    line_changes, = ax4.plot([], [], label='goukei Henkou', color='coral', linewidth=2)
    line_avg_speed, = ax4_twin.plot([], [], label='Heikin Sokudo', color='steelblue', linewidth=2, linestyle='--')
    
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, None)
    
    # 凡例
    lines = [line_changes, line_avg_speed]
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left', fontsize=9)
 
    # ── アニメーション関数 ──
    def animate(frame):
        xs, ys = to_xy(x_hist[frame], rl_hist[frame], display_r[frame])
        sc.set_offsets(np.c_[xs, ys])
        sc.set_array(v_hist[frame])
 
        if frame > 0:
            changed = lane_hist[frame] != lane_hist[frame - 1]
            sc_hi.set_offsets(np.c_[xs[changed], ys[changed]] if changed.any()
                              else np.empty((0, 2)))
        else:
            sc_hi.set_offsets(np.empty((0, 2)))
 
        counts_str = '  '.join(
            f'Shasen{i}:{lane_counts[frame, i]:>2}' for i in range(sim.n_lanes)
        )
        info_text.set_text(f'{counts_str}\nHenkou: {lane_changes[frame]:>2}/step')
        ax1.set_title(f'Step {frame:>4}  |  {sim.n_lanes} Shasen', fontsize=10)
        line_avg.set_data(range(frame + 1), avg_v[:frame + 1])
        line_min.set_data(range(frame + 1), min_v[:frame + 1])
        
        # 車線別グラフの更新
        for lane_idx in range(sim.n_lanes):
            lane_lines[lane_idx].set_data(
                range(frame + 1),
                avg_v_per_lane[:frame + 1, lane_idx]
            )
        
        # 累積車線変更数と平均速度の相関グラフを更新
        line_changes.set_data(range(frame + 1), lane_changes_cumsum[:frame + 1])
        line_avg_speed.set_data(range(frame + 1), avg_v[:frame + 1])
        
        # Y軸の最大値を動的に設定
        if frame > 0:
            max_changes = lane_changes_cumsum[frame]
            ax4.set_ylim(0, max(max_changes * 1.1, 10))
        
        return sc, sc_hi, line_avg, line_min, info_text, *lane_lines, line_changes, line_avg_speed
 
    ani = animation.FuncAnimation(
        fig, animate, frames=sim.n_steps,
        blit=False, repeat=False, interval=30
    )
 
    plt.tight_layout()
    plt.show()
 
    if save_gif:
        now      = datetime.datetime.now()
        filename = 'traffic_' + now.strftime('%Y%m%d_%H%M%S') + '.gif'
        gif_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        print('Saving GIF...', end='', flush=True)
        ani.save(gif_path, writer='pillow', fps=20)
        print(f'Done. -> {gif_path}')
 
 
# =========================================================
# エントリポイント
# =========================================================
 
if __name__ == '__main__':
 
    idm = IDMParams()
 
    n_lanes = 2
    sim = SimParams(
        n_cars             = 28 * n_lanes,
        ring_length        = 350.0,
        n_steps            = 1000,
        n_lanes            = n_lanes,
        perturbation       = 'stop_one',
        lane_change_start  = 0,
        lane_length_ratio  = 1.08,    # 外側車線は 8% 長い（350m → 378m）
        v0_lane_boost      = 1.5,     # 外側車線の希望速度 +1.5 m/s
        lane_car_weights   = (2, 1),  # 内側 75%・外側 25% で配車
                                      # → 外側は初期状態から空いている
    )
    mobil = MOBILParams(
        politeness             = 0.2,
        b_safe                 = 3.0,
        a_thr                  = 0.05,
        cooldown_steps         = 12,
        crowd_bias             = 0.8,
        congestion_bias        = 1.2,
        local_congestion_range = 60.0,
    )
    make_animation(idm, sim, mobil=mobil, save_gif=True)