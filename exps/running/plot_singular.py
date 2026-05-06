import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 奇异值数据（每组前10个）
data = {
    "Hot": [
        26.604706,
        11.284189,
        4.3937955,
        4.290208,
        4.1915426,
        3.8610313,
        3.8027933,
        3.5572467,
        3.4523094,
        3.3933766,
    ],
    "Mid": [
        17.605576,
        4.4112053,
        3.6281686,
        3.4122198,
        3.1664119,
        3.086927,
        3.0490398,
        3.0112593,
        2.9543078,
        2.8985455,
    ],
    "Cold": [11.77304, 2.963903, 2.3853154, 2.262263, 2.1060252, 2.0650778, 2.0319428, 2.0251844, 1.9964973, 1.9573315],
    "Ultra-Cold": [
        8.76224,
        2.1787047,
        1.7290881,
        1.6317786,
        1.5295507,
        1.4863485,
        1.4812238,
        1.4653237,
        1.4567609,
        1.4288211,
    ],
}

# 归一化：每行除以第一个元素
normalized = []
labels = ["Hot", "Mid", "Cold", "Ultra-Cold"]
for label in labels:
    arr = np.array(data[label])
    norm_arr = arr / arr[0]  # 归一化
    normalized.append(norm_arr)

# 转为 4x10 的 numpy 数组
heatmap_data = np.array(normalized)  # shape: (4, 10)

# 绘图
plt.figure(figsize=(8, 4))  # width=10, height=4（单位：英寸）
ax = sns.heatmap(
    heatmap_data,
    annot=True,  # 显示数值
    fmt=".2f",  # 保留两位小数
    cmap="Blues",  # 蓝色系
    cbar=True,
    yticklabels=labels,
    linewidths=0.5,
    square=False,
)

# 设置坐标轴标签
plt.ylabel("Group")
plt.title("Normalized Singular Values of Embeddings")

# 调整布局
plt.tight_layout()

# 显示或保存
plt.savefig("singular_heatmap.png", dpi=300)
