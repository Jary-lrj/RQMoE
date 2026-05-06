import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# --------------------
# 1. 通用读取函数：读取并计算平均 expert_counts
# --------------------
def load_avg_counts(path):
    records = [json.loads(line) for line in open(path, "r", encoding="utf-8") if line.strip()]
    df = pd.DataFrame(records)
    counts = np.stack(df["expert_counts"].to_numpy())  # shape [N, num_experts]
    avg_counts = counts.mean(axis=0)
    return avg_counts


# --------------------
# 2. 加载三个实验文件
# --------------------
avg_counts_lb1 = load_avg_counts("gate_stats_test_lb=0.01.jsonl")  # lb=1

# --------------------
# 3. 构造绘图 DataFrame
# --------------------
num_experts = len(avg_counts_lb1)
expert_ids = [f"Expert {i}" for i in range(num_experts)]

plot_df = pd.DataFrame(
    {
        "Expert ID": expert_ids,
        "The number of received tokens": np.concatenate([avg_counts_lb1]),
        "Setting": ["lb=0.01 (AUC=0.6863)"] * num_experts,
    }
)

# --------------------
# 4. 绘图（折线图 + 不同形状）
# --------------------
plt.figure(figsize=(8, 6))

# 定义点样式：圆、三角、星
markers = {"lb=0.01 (AUC=0.6863)": "^"}
palette = {"lb=0.01 (AUC=0.6863)": "#DD8452"}

sns.lineplot(
    data=plot_df,
    x="Expert ID",
    y="The number of received tokens",
    hue="Setting",
    style="Setting",
    markers=markers,
    dashes=False,
    palette=palette,
    linewidth=2.2,
    markersize=9,
)

plt.title("Expert Token Numbers on Beauty (Sparsity=99.9993%) with Different LB Weights")
plt.xlabel("Expert ID")
plt.ylabel("The number of received tokens")
plt.legend(title="", loc="upper right")
plt.tight_layout()
plt.savefig("expert_token_distribution_lb_comparison.png", dpi=300)
plt.show()
