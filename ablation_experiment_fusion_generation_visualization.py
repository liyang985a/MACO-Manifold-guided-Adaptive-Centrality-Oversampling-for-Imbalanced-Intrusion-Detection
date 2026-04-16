import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================================================
# 1. 全局字体与样式配置（增大字体，提升可读性）
# =========================================================
plt.rcParams.update({
    'font.family': 'Arial',
    'axes.unicode_minus': False,
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
})

# =========================================================
# 2. 加载数据
# =========================================================
df = pd.read_csv("synthesis_strategy_ablation.csv")
metrics = ["F1", "Precision", "MCC", "G-means"]

# 简化配色：高对比度、色盲友好的 Tableau 颜色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
markers = ['o', 's', '^', 'D']  # 不同指标使用不同标记，增强区分度

plt.figure(figsize=(11, 6.5))

# =========================================================
# 3. 绘制折线图
# =========================================================
for i, metric in enumerate(metrics):
    plt.plot(df["Strategy"], df[metric],
             marker=markers[i], markersize=9, linewidth=2.5,
             color=colors[i], label=metric, zorder=3)

# =========================================================
# 4. 坐标轴、网格与图例（避免与标题重叠）
# =========================================================
plt.ylabel("Score", fontsize=16)
plt.xlabel("Strategy", fontsize=16)
plt.title("Effect of Synthesis Strategies on Classification Performance",
          pad=25, fontweight='bold', fontsize=18)

plt.xticks(rotation=15)

# 轻量级网格
plt.grid(True, linestyle=':', alpha=0.5, zorder=0)

# 将图例放置在内部左上角，避免与标题和上边缘重叠
plt.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98),
           frameon=True, fancybox=True, ncol=1)

# 调整边距，为标题和轴标签预留空间
plt.subplots_adjust(top=0.92, bottom=0.12, left=0.12, right=0.95)

# =========================================================
# 5. 保存高分辨率图片（满足 SCI 标准）
# =========================================================
print("⏳ 正在保存高分辨率图片...")
plt.savefig("synthesis_strategy_ablation_line_plot.pdf", format='pdf', bbox_inches='tight')
plt.savefig("synthesis_strategy_ablation_line_plot.png", dpi=600, bbox_inches='tight')
print("✅ 图片保存成功！")

plt.show()