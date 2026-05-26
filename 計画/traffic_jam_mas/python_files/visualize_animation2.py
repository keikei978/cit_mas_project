# -*- coding: utf-8 -*-

import datetime
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from traffic_jam_mas2 import (
    IDMParams,
    SimParams,
    run_simulation
)


CMAP = 'RdYlGn'


# =========================================================
# 円座標変換
# =========================================================

def to_xy(s, ring_length):

    theta = (
        s / ring_length
    ) * 2 * np.pi

    return (
        np.cos(theta),
        np.sin(theta)
    )


# =========================================================
# 画像保存
# =========================================================

def save_graph(fig, name):

    now = datetime.datetime.now()

    filename = (
        name
        + '_'
        + now.strftime('%Y%m%d_%H%M%S')
        + '.png'
    )

    fig.savefig(
        filename,
        dpi=300
    )

    print('saved ->', filename)


# =========================================================
# メイン
# =========================================================

def make_animation(
    idm,
    sim,
    save_gif=True
):

    print('Running simulation...')

    result = run_simulation(
        idm,
        sim,
        record_positions=True
    )

    print('Simulation done.')

    x_hist = result['x_history']
    v_hist = result['v_history']

    avg_v = result['avg_v']
    min_v = result['min_v']

    std_v = result['std_v']

    jam_count = result['jam_count']

    mean_gap = result['mean_gap']

    mean_accel = result['mean_accel']

    # =====================================================
    # アニメーション
    # =====================================================

    print('Creating animation...')

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(12, 6)
    )

    ax1.set_xlim(-1.3, 1.3)

    ax1.set_ylim(-1.3, 1.3)

    ax1.set_aspect('equal')

    theta = np.linspace(
        0,
        2*np.pi,
        300
    )

    ax1.plot(
        np.cos(theta),
        np.sin(theta),
        color='lightgray'
    )

    # カーブ区間

    curve_theta = np.linspace(
        sim.curve_start
        / sim.ring_length
        * 2*np.pi,

        sim.curve_end
        / sim.ring_length
        * 2*np.pi,

        100
    )

    ax1.plot(
        np.cos(curve_theta),
        np.sin(curve_theta),
        linewidth=5
    )

    xs0, ys0 = to_xy(
        x_hist[0],
        sim.ring_length
    )

    scatter = ax1.scatter(
        xs0,
        ys0,
        c=v_hist[0],
        cmap=CMAP,
        vmin=0,
        vmax=idm.v0,
        s=50
    )

    ax2.set_xlim(0, sim.n_steps)

    ax2.set_ylim(0, idm.v0 * 1.1)

    line_avg, = ax2.plot(
        [],
        [],
        label='Average'
    )

    line_min, = ax2.plot(
        [],
        [],
        label='Minimum'
    )

    ax2.legend()

    ax2.grid(True)

    def animate(frame):

        xs, ys = to_xy(
            x_hist[frame],
            sim.ring_length
        )

        scatter.set_offsets(
            np.c_[xs, ys]
        )

        scatter.set_array(
            v_hist[frame]
        )

        line_avg.set_data(
            range(frame+1),
            avg_v[:frame+1]
        )

        line_min.set_data(
            range(frame+1),
            min_v[:frame+1]
        )

        ax1.set_title(
            f'frame={frame}'
        )

        return (
            scatter,
            line_avg,
            line_min
        )

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=sim.n_steps,
        interval=30
    )

    plt.tight_layout()

    plt.show()

    # =====================================================
    # GIF保存
    # =====================================================

    if save_gif:

        print('Saving GIF...')

        now = datetime.datetime.now()

        gif_name = (
            'traffic_'
            + now.strftime('%Y%m%d_%H%M%S')
            + '.gif'
        )

        ani.save(
            gif_name,
            writer='pillow',
            fps=20
        )

        print('GIF saved.')

    # =====================================================
    # グラフ作成
    # =====================================================

    print('Creating graphs...')

    # 平均速度

    fig1 = plt.figure(figsize=(10, 5))

    plt.plot(avg_v)

    plt.plot(min_v)

    plt.title('Velocity Transition')

    plt.grid(True)

    save_graph(fig1, 'velocity_transition')

    # 速度分散

    fig2 = plt.figure(figsize=(10, 5))

    plt.plot(std_v)

    plt.title('Velocity Variance')

    plt.grid(True)

    save_graph(fig2, 'velocity_variance')

    # 渋滞台数

    fig3 = plt.figure(figsize=(10, 5))

    plt.plot(jam_count)

    plt.title('Jam Count')

    plt.grid(True)

    save_graph(fig3, 'jam_count')

    # 車間距離

    fig4 = plt.figure(figsize=(10, 5))

    plt.plot(mean_gap)

    plt.title('Mean Gap')

    plt.grid(True)

    save_graph(fig4, 'mean_gap')

    # 加速度

    fig5 = plt.figure(figsize=(10, 5))

    plt.plot(mean_accel)

    plt.title('Mean Acceleration')

    plt.grid(True)

    save_graph(fig5, 'mean_acceleration')

    # =====================================================
    # 時空間図
    # =====================================================

    print('Creating spatio-temporal diagram...')

    fig6 = plt.figure(figsize=(12, 7))

    plt.imshow(
        v_hist,
        aspect='auto',
        cmap='RdYlGn',
        origin='lower'
    )

    plt.colorbar()

    plt.title('Spatio Temporal Diagram')

    save_graph(fig6, 'spatio_temporal')

    print('All done.')


if __name__ == '__main__':

    idm = IDMParams()

    sim = SimParams()

    make_animation(
        idm,
        sim,
        save_gif=True
    )