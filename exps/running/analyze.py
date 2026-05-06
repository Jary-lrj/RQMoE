import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm

# 1. 加载数据
data = torch.load("/root/autodl-tmp/users/liruijie/code/routers/exps/running/analysis_data/codebook_usage_stats.pt")
# keys: 'hot', 'mid', 'cold', 'ultra'

# 2. 合并 Tail (Cold + Ultra)
# 假设 indices shape 是 [N, M], 我们取第 1 层码本 (m=0) 做展示
m = 0
K = 128  # 你的码本大小

hot_indices = np.concatenate([data["hot"][:, m].numpy(), data["mid"][:, m].numpy()])
# 如果你想对比 Head vs Tail，就把 Cold 和 Ultra 合并
tail_indices = np.concatenate([data["cold"][:, m].numpy(), data["ultra"][:, m].numpy()])

print(hot_indices.sum())
print(tail_indices.sum())


# 3. 统计频率 (归一化为概率)
def get_prob(indices, k):
    counts = np.bincount(indices, minlength=k)
    return counts / (counts.sum() + 1e-9)


hot_prob = get_prob(hot_indices, K)
tail_prob = get_prob(tail_indices, K)

h_hot = -np.sum(hot_prob * np.log2(hot_prob + 1e-10))
h_tail = -np.sum(tail_prob * np.log2(tail_prob + 1e-10))
print(f"Entropy of hot items: {h_hot}")
print(f"Entropy of tail items: {h_tail}")


anchor_score = np.minimum(hot_prob, tail_prob)

# B. 差异分数 (Difference): Head - Tail
# 正值(红)代表 Head 特有，负值(蓝)代表 Tail 特有，0(白)代表平衡
diff_score = hot_prob - tail_prob

# C. Reshape 成 8x16 网格
grid_shape = (8, 16)
grid_anchor = anchor_score.reshape(grid_shape)
grid_diff = diff_score.reshape(grid_shape)

# ===========================
# 3. 绘图 (双图对比)
# ===========================
fig, axes = plt.subplots(1, 2, figsize=(8, 1.6))

# --- 图1: Shared Structural Anchors (这就是你要的证据) ---
# 使用 'magma' 或 'inferno' 色系，亮色代表高数值，非常显眼
sns.heatmap(
    grid_anchor,
    ax=axes[0],
    cmap="viridis",
    vmin=0.0,
    vmax=0.01,
    annot=False,
    cbar_kws={"label": "Shared Probability Mass"},
)
axes[0].axis("off")  # 去掉坐标轴更像指纹

# --- 图2: Distribution Shift (Difference Map) ---
# 使用 'RdBu_r' (红白蓝) 色系。
# 白色(0) = 锚点 或者 死码 (Stable)
# 红色 = Head 偏好, 蓝色 = Tail 偏好
limit = max(abs(diff_score.min()), abs(diff_score.max()))  # 确保0在中间
sns.heatmap(
    grid_diff,
    ax=axes[1],
    cmap="RdBu_r",
    center=0,
    vmin=-limit,
    vmax=limit,
    annot=False,
    cbar_kws={"label": "P(Head) - P(Tail)"},
)
axes[1].axis("off")

plt.tight_layout()
plt.savefig("codebook_fingerprint.png", bbox_inches="tight")
