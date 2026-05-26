# -*- coding: utf-8 -*-

"""
curve_v0 の違いによる比較
"""

import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# CSV読み込み
# =========================================================

print('Loading CSV files...')

df10 = pd.read_csv(
    'traffic_sim/curve10.csv'
)

df12 = pd.read_csv(
    'traffic_sim/curve12.csv'
)

df14 = pd.read_csv(
    'traffic_sim/curve14.csv'
)

df16 = pd.read_csv(
    'traffic_sim/curve16.csv'
)

print('Done.')


# =========================================================
# グラフ1
# 平均速度比較
# =========================================================

print('Creating graph...')

plt.figure(figsize=(10, 5))

plt.plot(
    df10['n_cars'],
    df10['mean_avg_v'],
    marker='o',
    label='curve_v0 = 10'
)

plt.plot(
    df12['n_cars'],
    df12['mean_avg_v'],
    marker='o',
    label='curve_v0 = 12'
)

plt.plot(
    df14['n_cars'],
    df14['mean_avg_v'],
    marker='o',
    label='curve_v0 = 14'
)

plt.plot(
    df16['n_cars'],
    df16['mean_avg_v'],
    marker='o',
    label='curve_v0 = 16'
)

plt.xlabel('Number of Cars')

plt.ylabel('Mean Average Velocity')

plt.title('Comparison of curve_v0')

plt.grid(True)

plt.legend()


# =========================================================
# 保存
# =========================================================

save_name = 'compare_curve_v0.png'

plt.savefig(
    save_name,
    dpi=300
)

print('saved ->', save_name)


# =========================================================
# 表示
# =========================================================

plt.show()

print('ALL DONE')