# -*- coding: utf-8 -*-

"""
車両数を変化させて
自然渋滞の発生しやすさを調べる
"""

import datetime
import pandas as pd
import matplotlib.pyplot as plt

from traffic_jam_mas2 import (
    IDMParams,
    SimParams,
    run_simulation,
    summary_stats
)


# =========================================================
# 実験条件
# =========================================================

# 車両数を変化
car_list = range(16, 41, 2)

# 結果保存用
results = []


# =========================================================
# 実験開始
# =========================================================

print('==============================')
print('START EXPERIMENT')
print('==============================')

for n_cars in car_list:

    print(f'\nNow running: n_cars = {n_cars}')

    # ---------------------------------------------
    # シミュレーション条件
    # ---------------------------------------------

    sim = SimParams(

        n_cars=n_cars,

        ring_length=350,

        n_steps=600,

        curve_v0=16.0,

        random_brake_prob=0.003,

        random_brake_strength=1.0
    )

    idm = IDMParams()

    # ---------------------------------------------
    # シミュレーション実行
    # ---------------------------------------------

    result = run_simulation(
        idm,
        sim,
        record_positions=False
    )

    stats = summary_stats(result)

    # ---------------------------------------------
    # 結果保存
    # ---------------------------------------------

    results.append({

        'n_cars': n_cars,

        'mean_avg_v':
            stats['mean_avg_v'],

        'std_avg_v':
            stats['std_avg_v'],

        'max_jam_count':
            stats['max_jam_count']
    })

    print('done')

print('\n==============================')
print('ALL EXPERIMENTS DONE')
print('==============================')


# =========================================================
# DataFrame化
# =========================================================

df = pd.DataFrame(results)

print('\nResult Table')
print(df)


# =========================================================
# CSV保存
# =========================================================

now = datetime.datetime.now()

csv_name = (
    'experiment_ncars_'
    + now.strftime('%Y%m%d_%H%M%S')
    + '.csv'
)

df.to_csv(
    csv_name,
    index=False
)

print('\nCSV saved ->', csv_name)


# =========================================================
# グラフ1
# 平均速度
# =========================================================

fig1 = plt.figure(figsize=(10, 5))

plt.plot(
    df['n_cars'],
    df['mean_avg_v'],
    marker='o'
)

plt.xlabel('Number of Cars')

plt.ylabel('Mean Average Velocity')

plt.title('Average Velocity vs Number of Cars')

plt.grid(True)

graph1_name = (
    'avg_velocity_vs_ncars_'
    + now.strftime('%Y%m%d_%H%M%S')
    + '.png'
)

fig1.savefig(
    graph1_name,
    dpi=300
)

print('saved ->', graph1_name)


# =========================================================
# グラフ2
# 最大渋滞台数
# =========================================================

fig2 = plt.figure(figsize=(10, 5))

plt.plot(
    df['n_cars'],
    df['max_jam_count'],
    marker='o'
)

plt.xlabel('Number of Cars')

plt.ylabel('Maximum Jam Count')

plt.title('Jam Count vs Number of Cars')

plt.grid(True)

graph2_name = (
    'jamcount_vs_ncars_'
    + now.strftime('%Y%m%d_%H%M%S')
    + '.png'
)

fig2.savefig(
    graph2_name,
    dpi=300
)

print('saved ->', graph2_name)


# =========================================================
# グラフ3
# 速度変動
# =========================================================

fig3 = plt.figure(figsize=(10, 5))

plt.plot(
    df['n_cars'],
    df['std_avg_v'],
    marker='o'
)

plt.xlabel('Number of Cars')

plt.ylabel('Velocity Std')

plt.title('Velocity Variance vs Number of Cars')

plt.grid(True)

graph3_name = (
    'variance_vs_ncars_'
    + now.strftime('%Y%m%d_%H%M%S')
    + '.png'
)

fig3.savefig(
    graph3_name,
    dpi=300
)

print('saved ->', graph3_name)


# =========================================================
# 表示
# =========================================================

plt.show()

print('\nALL DONE')