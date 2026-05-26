# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime
import os

# シミュレーションパラメータの定義
WIDTH = 50                   # 空間の幅
HEIGHT = 50                  # 空間の高さ
AGENTS_NUMBER = 300          # エージェント数
INITIAL_INFECTED_NUMBER = 1  # 初期感染者数
INFECTION_RADIUS = 3.0       # 感染半径
INFECTION_PROBABILITY = 0.1  # 感染率
RECOVERY_TIME = 30           # 回復までの期間
SIMULATION_TIME = 200        # シミュレーション時間
AGENT_SPEED = 1.0            # エージェントの移動速度

# エージェントの状態の定義
SUSCEPTIBLE = 0  # 未感染
INFECTED = 1     # 感染
RECOVERED = 2    # 回復

# 状態の色の定義
COLOR_SUSCEPTIBLE = 'blue'  # 未感染
COLOR_INFECTED = 'red'      # 感染
COLOR_RECOVERED = 'green'   # 回復

# エージェントクラスの定義
class Agent:

    # 初期化関数の定義
    def __init__(self, x, y, state):
        self.x = x                # x座標
        self.y = y                # y座標
        self.state = state        # 状態
        self.infection_time = -1  # 感染時刻（-1は未感染）
        self.recovery_time = -1   # 回復時刻（-1は未回復）

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
        agent = Agent(x, y, SUSCEPTIBLE)  # 未感染エージェントを作成
        agents.append(agent)              # リストに追加

    # 初期感染者設定
    infected_indices = np.random.choice(AGENTS_NUMBER, INITIAL_INFECTED_NUMBER, replace=False)  # エージェントをランダムに選択
    for i in infected_indices:
        agents[i].state = INFECTED    # 選ばれたエージェントの状態を感染に変更
        agents[i].infection_time = 0  # 感染時刻をゼロに設定

    return agents

# 感染判定関数の定義
def check_infection(agent1, agent2, frame):
    if agent2.state == SUSCEPTIBLE:  # agent2 が未感染の場合
        distance = np.sqrt((agent1.x - agent2.x)**2 + (agent1.y - agent2.y)**2)       # エージェント間の距離を計算
        if distance < INFECTION_RADIUS and np.random.rand() < INFECTION_PROBABILITY:  # 距離が感染半径以内，かつ乱数が感染率を下回った場合
            agent2.state = INFECTED        # 状態を感染に更新
            agent2.infection_time = frame  # 感染時刻を記録

# 回復判定関数の定義
def check_recovery(agent, frame):
    if agent.state == INFECTED and frame - agent.infection_time > RECOVERY_TIME:  # 感染状態かつ回復期間を経過した場合
        agent.state = RECOVERED      # 状態を回復に更新
        agent.recovery_time = frame  # 回復時刻を記録
        agent.infection_time = -1    # 感染時刻をリセット

# シミュレーション実行関数の定義
def run_simulation():

    # 各状態のエージェント数の履歴
    count_susceptible = []
    count_infected = []
    count_recovered = []

    # シミュレーション履歴
    history = []

    # シミュレーション初期化
    agents = initialize_simulation()

    # メインループ：フレームごとの処理
    for frame in range(SIMULATION_TIME):

        # 全エージェントの移動
        for agent in agents:
            agent.move()

        # 感染判定
        for i, agent1 in enumerate(agents):
            if agent1.state == INFECTED:  # agent1 が感染の場合
                for j, agent2 in enumerate(agents):
                    if i != j:  # 同じエージェントではない場合
                        check_infection(agent1, agent2, frame)  # 感染判定

        # 回復判定
        for agent in agents:
            check_recovery(agent, frame)  # 回復判定

        # 各状態のエージェント数の更新と履歴への追加
        num_susceptible = sum(1 for agent in agents if agent.state == SUSCEPTIBLE)
        num_infected = sum(1 for agent in agents if agent.state == INFECTED)
        num_recovered = sum(1 for agent in agents if agent.state == RECOVERED)
        count_susceptible.append(num_susceptible)
        count_infected.append(num_infected)
        count_recovered.append(num_recovered)

        # 現在の状態をシミュレーション履歴に追加
        agents_copy = [Agent(agent.x, agent.y, agent.state) for agent in agents]                    # 全エージェントの状態をコピー
        history.append((agents_copy, count_susceptible[:], count_infected[:], count_recovered[:]))  # 現在のフレームの全データを履歴に追加

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
line_susceptible, = ax2.plot([], [], label='Susceptible', color=COLOR_SUSCEPTIBLE)
line_infected, = ax2.plot([], [], label='Infected', color=COLOR_INFECTED)
line_recovered, = ax2.plot([], [], label='Recovered', color=COLOR_RECOVERED)
ax2.legend()
ax2.grid(True)

# シミュレーション実行
history = run_simulation()

# 初期状態描画
agents = history[0][0]
x = [agent.x for agent in agents]
y = [agent.y for agent in agents]
colors = [COLOR_SUSCEPTIBLE if agent.state == SUSCEPTIBLE else COLOR_INFECTED if agent.state == INFECTED else COLOR_RECOVERED for agent in agents]
scatter = ax1.scatter(x, y, c=colors, s=20)

# アニメーション関数（履歴をもとに描画）
def animate(frame):
    agents, count_susceptible, count_infected, count_recovered = history[frame]

    # 散布図の更新
    x = [agent.x for agent in agents]
    y = [agent.y for agent in agents]
    colors = [COLOR_SUSCEPTIBLE if agent.state == SUSCEPTIBLE else COLOR_INFECTED if agent.state == INFECTED else COLOR_RECOVERED for agent in agents]
    scatter.set_offsets(np.c_[x, y])
    scatter.set_facecolor(colors)

    # 図の更新
    ax1.set_title(f"Frame: {frame}")
    line_susceptible.set_data(range(len(count_susceptible)), count_susceptible)
    line_infected.set_data(range(len(count_infected)), count_infected)
    line_recovered.set_data(range(len(count_recovered)), count_recovered)

    return scatter, line_susceptible, line_infected, line_recovered,

# アニメーション実行
ani = animation.FuncAnimation(fig, animate, frames=SIMULATION_TIME, blit=False, repeat=False)

# 画面表示（ウィンドウは手動で閉じる）
plt.tight_layout()
plt.show()

# アニメーション保存（不要な場合は以下をコメントアウト）
now = datetime.datetime.now()
filename = 'simulation_' + now.strftime('%Y%m%d_%H%M%S') + '.gif'
gif_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
print('Now Saving...', end='', flush=True)
ani.save(gif_path, writer="pillow", fps=10)  # Pillow ライブラリが必要
print('Done.')