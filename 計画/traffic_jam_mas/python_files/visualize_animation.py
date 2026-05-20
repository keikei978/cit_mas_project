# -*- coding: utf-8 -*-
"""
環状道路シミュレーションのアニメーション可視化
- 元コードの可視化部分だけを切り出し、シミュレーション本体は traffic_sim.py に任せる
- GIF 保存も従来通り
"""

import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from traffic_jam_mas import IDMParams, SimParams, run_simulation


# 可視化設定
CMAP = 'RdYlGn'


def to_xy(s, ring_length):
    """環状道路の弧長 s を xy 座標に変換"""
    theta = (s / ring_length) * 2 * np.pi
    return np.cos(theta), np.sin(theta)


def make_animation(idm: IDMParams, sim: SimParams, save_gif: bool = True):
    # シミュレーションを走らせて全ステップの位置・速度を記録
    print('Running simulation...', end='', flush=True)
    result = run_simulation(idm, sim, record_positions=True)
    print('Done.')

    x_hist = result['x_history']     # (n_steps, n_cars)
    v_hist = result['v_history']
    avg_v = result['avg_v']
    min_v = result['min_v']

    # 描画準備
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # 左：環状道路
    ax1.set_xlim(-1.3, 1.3)
    ax1.set_ylim(-1.3, 1.3)
    ax1.set_aspect('equal')
    theta = np.linspace(0, 2 * np.pi, 200)
    ax1.plot(np.cos(theta), np.sin(theta), color='lightgray', linewidth=1)

    # 右：速度の時系列
    ax2.set_xlim(0, sim.n_steps)
    ax2.set_ylim(0, idm.v0 * 1.1)
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Velocity over time')
    line_avg, = ax2.plot([], [], label='Average', color='blue')
    line_min, = ax2.plot([], [], label='Min', color='red')
    ax2.legend()
    ax2.grid(True)

    # 初期状態の散布図
    xs0, ys0 = to_xy(x_hist[0], sim.ring_length)
    scatter = ax1.scatter(xs0, ys0, c=v_hist[0], cmap=CMAP,
                          vmin=0, vmax=idm.v0, s=50)

    def animate(frame):
        xs, ys = to_xy(x_hist[frame], sim.ring_length)
        scatter.set_offsets(np.c_[xs, ys])
        scatter.set_array(v_hist[frame])
        ax1.set_title(f'Frame: {frame}')
        line_avg.set_data(range(frame + 1), avg_v[:frame + 1])
        line_min.set_data(range(frame + 1), min_v[:frame + 1])
        return scatter, line_avg, line_min

    ani = animation.FuncAnimation(
        fig, animate, frames=sim.n_steps,
        blit=False, repeat=False, interval=30
    )

    plt.tight_layout()
    plt.show()

    # GIF 保存
    if save_gif:
        now = datetime.datetime.now()
        filename = 'traffic_' + now.strftime('%Y%m%d_%H%M%S') + '.gif'
        gif_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        print('Saving GIF...', end='', flush=True)
        ani.save(gif_path, writer='pillow', fps=20)
        print(f'Done. -> {gif_path}')


if __name__ == '__main__':
    # 元コードと同じパラメータでアニメーション
    idm = IDMParams()
    sim = SimParams(n_cars=28, ring_length=350.0, n_steps=600)
    make_animation(idm, sim, save_gif=True)