"""GUI 子包：matplotlib 后端与全局字体设置（导入即生效）。

原 gui_main.py 顶部的全局副作用迁移至此（TkAgg 后端 + SimHei 字体 + 负号显示）。
"""
import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']   # 使用黑体
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题