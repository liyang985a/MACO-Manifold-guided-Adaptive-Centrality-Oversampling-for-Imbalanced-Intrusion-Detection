import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# 1. 全局字体与样式配置（进一步增大字体，提升可读性）
# =========================================================
plt.rcParams.update({
    'font.family': 'Arial',
    'axes.unicode_minus': False,
    'font.size': 16,               # 全局默认字体增大
    'axes.titlesize': 18,          # 标题字体增大
    'axes.labelsize': 16,          # 轴标签字体增大
    'xtick.labelsize': 14,         # X轴刻度字体
    'ytick.labelsize': 14,         # Y轴刻度字体
    'legend.fontsize': 14,         # 图例字体
    'figure.dpi': 100              # 屏幕显示DPI（不影响保存）
})

# =========================================================
# 2. 加载数据（请确保 CSV 文件路径正确）
# =========================================================
df = pd.read_csv("manifold_ablation_results.csv")
metrics = ["F1", "Precision", "MCC", "G-means"]

# 简化配色方案：使用色盲友好且高对比度的4种颜色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 标准 Tableau 10 前四种
markers = ['o', 's', '^', 'D']

plt.figure(figsize=(11, 6.5))  # 稍微加宽画布，为图例腾出空间

# =========================================================
# 3. 绘制折线图
# =========================================================
for i, metric in enumerate(metrics):
    plt.plot(df["Mapping"], df[metric],
             marker=markers[i], markersize=9, linewidth=2.5,
             color=colors[i], label=metric, zorder=3)

# =========================================================
# 4. 坐标轴、网格与图例（解决顶部重叠问题）
# =========================================================
plt.ylabel("Score", fontsize=16)
plt.xlabel("Manifold Mapping Methods", fontsize=16)
plt.title("Comparison of Manifold Mappings via Performance Metrics",
          pad=25, fontweight='bold', fontsize=18)

# 旋转X轴刻度，避免文字拥挤
plt.xticks(rotation=15)

# 轻量级网格
plt.grid(True, linestyle=':', alpha=0.5, zorder=0)

# 关键修改：将图例放置在图表内部右上角（避免与标题重叠）
# 若您仍希望外部放置，可调整为 bbox_to_anchor=(1.01, 1)，但需调整布局
plt.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98),
           frameon=True, fancybox=True, shadow=False, ncol=1)

# 或者使用 'best' 自动选择不重叠位置：
# plt.legend(loc='best', frameon=True, ncol=1)

# 调整子图边距，为标题和轴标签预留足够空间
plt.subplots_adjust(top=0.92, bottom=0.12, left=0.12, right=0.95)

# =========================================================
# 5. 保存高分辨率图片（满足SCI期刊要求）
# =========================================================
print("⏳ 正在保存高分辨率图片...")
plt.savefig("manifold_mapping_line_plot.pdf", format='pdf', bbox_inches='tight')
plt.savefig("manifold_mapping_line_plot.png", dpi=600, bbox_inches='tight')
print("✅ 图片保存成功！")

plt.show()