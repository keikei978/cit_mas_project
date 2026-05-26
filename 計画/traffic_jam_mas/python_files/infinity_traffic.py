# -*- coding: utf-8 -*-

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

ROAD_SCALE = 2.5


# =========================================================
# ∞コース
# =========================================================

def to_xy(s, L, a=ROAD_SCALE):
    t = (s / L) * 2 * np.pi
    return a * np.sin(t), a * np.sin(t) * np.cos(t)


def branch_index(s, L):
    t = (s / L) * 2 * np.pi
    return 0 if np.cos(t) >= 0 else 1


# =========================================================
# 距離
# =========================================================

def gap(front, me, L):
    return (front - me + L) % L


# =========================================================
# パラメータ
# =========================================================

V0 = 8.0
DT = 0.2

SAFE_GAP = 12.0
STOP_ZONE = 0.18  # 交差点サイズ
ENTER_ZONE = STOP_ZONE * 4

STOP_TIME = 4.0


# =========================================================
# 交差点判定（中心）
# =========================================================

def in_cross(x, y):
    return abs(x) < STOP_ZONE and abs(y) < STOP_ZONE


def approaching_cross(x, y):
    return abs(x) < ENTER_ZONE and abs(y) < ENTER_ZONE


# =========================================================
# シミュレーション
# =========================================================

def run_simulation(n_cars=28, n_steps=600, L=700.0):

    s = np.linspace(0, L, n_cars, endpoint=False)
    s = (s + 0.5 * L / n_cars) % L
    while np.any([in_cross(*to_xy(si, L)) for si in s]):
        s = (s + 0.25 * L / n_cars) % L

    v = np.ones(n_cars) * V0 * 0.6

    stop_timer = np.zeros(n_cars)
    in_intersection = np.zeros(n_cars, dtype=bool)
    active_branch = None

    x_hist, v_hist, avg_v, min_v = [], [], [], []

    for _ in range(n_steps):

        pos = np.array([to_xy(si, L) for si in s])
        xs, ys = pos[:, 0], pos[:, 1]

        front = np.roll(s, -1)
        g = gap(front, s, L)

        acc = np.zeros(n_cars)

        # =====================================================
        # ① 交差点占有チェック＋ランダム譲り
        # =====================================================

        candidates = [i for i in range(n_cars) if approaching_cross(xs[i], ys[i]) and not in_intersection[i]]
        if active_branch is None:
            cross_candidates = [i for i in range(n_cars) if in_cross(xs[i], ys[i]) and not in_intersection[i]]
            if cross_candidates:
                branches = {}
                for i in cross_candidates:
                    bi = branch_index(s[i], L)
                    branches.setdefault(bi, []).append(i)
                active_branch = random.choice(list(branches.keys()))

        if active_branch is not None and len(candidates) > 0:
            for i in candidates:
                bi = branch_index(s[i], L)
                if bi == active_branch:
                    in_intersection[i] = True
                elif stop_timer[i] == 0:
                    stop_timer[i] = STOP_TIME + 1.0

        # =====================================================
        # ② 交差点ルール（停止＋占有）
        # =====================================================

        for i in range(n_cars):

            if stop_timer[i] > 0:

                # 完全停止
                acc[i] = -3.0 * v[i]
                stop_timer[i] = max(0.0, stop_timer[i] - DT)
                continue

        # =====================================================
        # ③ 通常走行（車間距離制御）
        # =====================================================

        for i in range(n_cars):

            if stop_timer[i] > 0:
                continue

            # 前車に追従（安全距離）
            desired_gap = SAFE_GAP + 1.0 * v[i]

            if g[i] < desired_gap:
                acc[i] -= 0.6 * (desired_gap - g[i])

            # 基本走行
            acc[i] += 0.05 * (V0 - v[i])

        # =====================================================
        # ④ 更新
        # =====================================================

        v += acc * DT
        v = np.clip(v, 0, V0)

        s = (s + v * DT) % L

        # 交差点を出たら占有解除（停止後の再発進を許可）
        pos = np.array([to_xy(si, L) for si in s])
        xs, ys = pos[:, 0], pos[:, 1]
        for i in range(n_cars):
            if in_intersection[i] and stop_timer[i] == 0 and not in_cross(xs[i], ys[i]):
                in_intersection[i] = False

        if active_branch is not None:
            remaining = [i for i in range(n_cars)
                         if branch_index(s[i], L) == active_branch and
                         (in_cross(xs[i], ys[i]) or approaching_cross(xs[i], ys[i]))]
            if len(remaining) == 0:
                active_branch = None

        # =====================================================
        # 記録
        # =====================================================

        pos = np.array([to_xy(si, L) for si in s])
        xs, ys = pos[:, 0], pos[:, 1]

        x_hist.append(np.c_[xs, ys])
        v_hist.append(v.copy())
        avg_v.append(np.mean(v))
        min_v.append(np.min(v))

    return np.array(x_hist), np.array(v_hist), np.array(avg_v), np.array(min_v)


# =========================================================
# アニメーション
# =========================================================

def main():

    x_hist, v_hist, avg_v, min_v = run_simulation()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    t = np.linspace(0, 2*np.pi, 500)

    ax1.plot(ROAD_SCALE*np.sin(t), ROAD_SCALE*np.sin(t)*np.cos(t), color="lightgray")
    ax1.set_aspect("equal")

    scatter = ax1.scatter(
        x_hist[0][:, 0],
        x_hist[0][:, 1],
        c=v_hist[0],
        cmap="RdYlGn",
        vmin=0,
        vmax=V0
    )

    ax2.set_xlim(0, len(avg_v))
    ax2.set_ylim(0, V0)

    line1, = ax2.plot([], [], label="avg")
    line2, = ax2.plot([], [], label="min")

    ax2.legend()
    ax2.grid()

    def update(i):
        scatter.set_offsets(x_hist[i])
        scatter.set_array(v_hist[i])

        line1.set_data(range(i), avg_v[:i])
        line2.set_data(range(i), min_v[:i])

        return scatter, line1, line2

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(x_hist),
        interval=30,
        blit=False
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()