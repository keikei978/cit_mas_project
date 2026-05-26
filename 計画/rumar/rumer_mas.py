# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime
import os

# シミュレーションパラメータの定義
WIDTH = 50                       # 空間の幅
HEIGHT = 50                      # 空間の高さ
AGENTS_NUMBER = 300              # エージェント数
INITIAL_SPREADER_NUMBER = 10     # 初期拡散者数
INTERACTION_RADIUS = 3.0         # 相互作用半径
SPREADING_PROBABILITY = 0.1     # 拡散確率（S → I：未知者が拡散者と出会ったとき）
STIFLING_PROBABILITY = 0.5      # 飽き確率（I → R：拡散者が拡散者または飽き者と出会ったとき）
SIMULATION_TIME = 200            # シミュレーション時間
AGENT_SPEED = 1.0                # エージェントの移動速度

# エージェントの状態の定義
SUSCEPTIBLE = 0  # 未知者：噂を知らない
SPREADER = 1     # 拡散者：噂を知っていて広めている
STIFLER = 2      # 飽き者：噂を知っているが広めない

# 状態の色の定義
COLOR_SUSCEPTIBLE = 'blue'   # 未知者
COLOR_SPREADER = 'red'       # 拡散者
COLOR_STIFLER = 'green'      # 飽き者

# エージェントクラスの定義
class Agent:

    # 初期化関数の定義
    def __init__(self, x, y, state):
        self.x = x          # x座標
        self.y = y          # y座標
        self.state = state  # 状態

    # ランダムな方向への移動
    def move(self):
        angle = np.random.rand() * 2 * np.pi   # 移動角度をランダムに決定
        self.x += AGENT_SPEED * np.cos(angle)  # x座標更新
        self.y += AGENT_SPEED * np.sin(angle)  # y座標更新
        self.x = max(0, min(self.x, WIDTH))    # 空間内に収まるように調整
        self.y = max(0, min(self.y, HEIGHT))   # 空間内に収まるように調整

# シミュレーション初期化関数の定義
def initialize_simulation():

    # エージェント初期化
    agents = []
    for _ in range(AGENTS_NUMBER):
        x = np.random.rand() * WIDTH      # x座標を初期化
        y = np.random.rand() * HEIGHT     # y座標を初期化
        agent = Agent(x, y, SUSCEPTIBLE)  # 未知者エージェントを作成
        agents.append(agent)              # リストに追加

    # 初期拡散者設定
    spreader_indices = np.random.choice(AGENTS_NUMBER, INITIAL_SPREADER_NUMBER, replace=False)
    for i in spreader_indices:
        agents[i].state = SPREADER  # 選ばれたエージェントの状態を拡散者に変更

    return agents

# 拡散判定関数の定義（S → I）
def check_spreading(spreader, other):
    # other が未知者なら、確率 SPREADING_PROBABILITY で拡散者に変える
    if other.state == SUSCEPTIBLE:
        distance = np.sqrt((spreader.x - other.x)**2 + (spreader.y - other.y)**2)
        if distance < INTERACTION_RADIUS and np.random.rand() < SPREADING_PROBABILITY:
            other.state = SPREADER

# 飽き判定関数の定義（I → R、相互作用ベース）
def check_stifling(spreader, other):
    # other が拡散者または飽き者なら、spreader は確率 STIFLING_PROBABILITY で飽き者になる
    # 「みんなもう知ってる」ことが分かって広めるのをやめる、という解釈
    if other.state == SPREADER or other.state == STIFLER:
        distance = np.sqrt((spreader.x - other.x)**2 + (spreader.y - other.y)**2)
        if distance < INTERACTION_RADIUS and np.random.rand() < STIFLING_PROBABILITY:
            spreader.state = STIFLER

# シミュレーション実行関数の定義
def run_simulation():

    # 各状態のエージェント数の履歴
    count_susceptible = []
    count_spreader = []
    count_stifler = []

    # シミュレーション履歴
    history = []

    # シミュレーション初期化
    agents = initialize_simulation()

    # メインループ：フレームごとの処理
    for frame in range(SIMULATION_TIME):

        # 全エージェントの移動
        for agent in agents:
            agent.move()

        # 相互作用判定
        for i, agent1 in enumerate(agents):
            if agent1.state == SPREADER:  # agent1 が拡散者の場合のみ処理
                for j, agent2 in enumerate(agents):
                    if i == j:
                        continue
                    if agent1.state != SPREADER:  # 途中で飽き者になったら抜ける
                        break
                    if agent2.state == SUSCEPTIBLE:
                        check_spreading(agent1, agent2)  # 拡散判定
                    else:
                        check_stifling(agent1, agent2)   # 飽き判定

        # 各状態のエージェント数の更新と履歴への追加
        num_susceptible = sum(1 for agent in agents if agent.state == SUSCEPTIBLE)
        num_spreader = sum(1 for agent in agents if agent.state == SPREADER)
        num_stifler = sum(1 for agent in agents if agent.state == STIFLER)
        count_susceptible.append(num_susceptible)
        count_spreader.append(num_spreader)
        count_stifler.append(num_stifler)

        # 現在の状態をシミュレーション履歴に追加
        agents_copy = [Agent(agent.x, agent.y, agent.state) for agent in agents]                 # 全エージェントの状態をコピー
        history.append((agents_copy, count_susceptible[:], count_spreader[:], count_stifler[:])) # 現在のフレームの全データを履歴に追加

    return history

# 描画設定
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# 散布図の初期化
ax1.set_xlim(0, WIDTH)
ax1.set_ylim(0, HEIGHT)
ax1.set_aspect('equal')
ax1.set_title("Frame: 0")

# 状態数推移グラフの初期化
ax2.set_xlim(0, SIMULATION_TIME)
ax2.set_ylim(0, AGENTS_NUMBER)
ax2.set_xlabel('Time')
ax2.set_ylabel('Number of Agents')
ax2.set_title('Number of Agents over Time')
line_susceptible, = ax2.plot([], [], label='Susceptible (unknown)', color=COLOR_SUSCEPTIBLE)
line_spreader,    = ax2.plot([], [], label='Spreader (active)',    color=COLOR_SPREADER)
line_stifler,     = ax2.plot([], [], label='Stifler (stopped)',    color=COLOR_STIFLER)
ax2.legend()
ax2.grid(True)

# シミュレーション実行
history = run_simulation()

# 初期状態描画
agents = history[0][0]
x = [agent.x for agent in agents]
y = [agent.y for agent in agents]
colors = [COLOR_SUSCEPTIBLE if agent.state == SUSCEPTIBLE else COLOR_SPREADER if agent.state == SPREADER else COLOR_STIFLER for agent in agents]
scatter = ax1.scatter(x, y, c=colors, s=20)

# アニメーション関数（履歴をもとに描画）
def animate(frame):
    agents, count_susceptible, count_spreader, count_stifler = history[frame]

    # 散布図の更新
    x = [agent.x for agent in agents]
    y = [agent.y for agent in agents]
    colors = [COLOR_SUSCEPTIBLE if agent.state == SUSCEPTIBLE else COLOR_SPREADER if agent.state == SPREADER else COLOR_STIFLER for agent in agents]
    scatter.set_offsets(np.c_[x, y])
    scatter.set_facecolor(colors)

    # 図の更新
    ax1.set_title(f"Frame: {frame}")
    line_susceptible.set_data(range(len(count_susceptible)), count_susceptible)
    line_spreader.set_data(range(len(count_spreader)), count_spreader)
    line_stifler.set_data(range(len(count_stifler)), count_stifler)

    return scatter, line_susceptible, line_spreader, line_stifler,

# アニメーション実行
ani = animation.FuncAnimation(fig, animate, frames=SIMULATION_TIME, blit=False, repeat=False)

# 画面表示（ウィンドウは手動で閉じる）
plt.tight_layout()
plt.show()

# アニメーション保存（不要な場合は以下をコメントアウト）
now = datetime.datetime.now()
filename = 'rumor_simulation_' + now.strftime('%Y%m%d_%H%M%S') + '.gif'
gif_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
print('Now Saving...', end='', flush=True)
ani.save(gif_path, writer="pillow", fps=10)  # Pillow ライブラリが必要
print('Done.')