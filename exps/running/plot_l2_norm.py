import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 设置科研风格和配色
plt.style.use("seaborn-v0_8-paper")
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]  # 更多颜色选择
sns.set_palette(sns.color_palette(colors))

data_dict = {
    "Base Model": [0.1192, 0.0539, 0.0472, 0.0408],
    "Vanilla SparseMoE": [0.1173, 0.0501, 0.0445, 0.0386],
    "RQ-MoE": [0.0933, 0.0433, 0.0385, 0.0341],
}

groups = ["Hot", "Mid", "Cold", "Ultra-Cold"]

# 创建DataFrame
data_list = []
for method_name, values in data_dict.items():
    for i, group in enumerate(groups):
        data_list.append({"Group": group, "L2 Norm": values[i], "Method": method_name})

data = pd.DataFrame(data_list)

# 创建图表
fig, ax = plt.subplots(figsize=(8, 6))

# 绘制多条折线图
sns.lineplot(data=data, x="Group", y="L2 Norm", hue="Method", marker="o", markersize=8, linewidth=2.5, ax=ax)

# 设置标题和标签
ax.set_title("L2 Norm of Grouped Embeddings of MoE in Beauty", fontsize=14, fontweight="bold", pad=20)
ax.set_xlabel("Group", fontsize=12, fontweight="bold")
ax.set_ylabel("L2 Norm", fontsize=12, fontweight="bold")

# 调整坐标轴
ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 设置图例
ax.legend(title="Method", frameon=True, fancybox=True, shadow=True, framealpha=0.9)

# 调整布局
plt.tight_layout()

# 可选：保存图片
plt.savefig("l2_norm_comparison.png", dpi=300, bbox_inches="tight")
