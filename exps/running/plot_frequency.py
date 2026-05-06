import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import Counter


def plot_frequency_log_log(freq_map, title="Frequency Distribution (Log-Log Plot)"):
    """
    绘制物品频率分布的log-log图

    参数:
    freq_map: 列表，第i个元素代表itemid为i的商品的出现个数
    title: 图表标题
    """
    # 设置科研风格
    plt.style.use("seaborn-v0_8-paper")
    sns.set_palette("husl")

    # 统计每个频率k对应的物品数量xk
    freq_counter = Counter(freq_map)

    # 转换为DataFrame以便处理
    freq_df = pd.DataFrame({"k": list(freq_counter.keys()), "xk": list(freq_counter.values())})

    # 过滤掉k=0的情况（如果有的话）
    freq_df = freq_df[freq_df["k"] > 0]

    # 按k排序
    freq_df = freq_df.sort_values("k")

    # 创建图表
    fig, ax = plt.subplots(figsize=(8, 6))

    # 绘制log-log散点图
    plt.loglog(freq_df["k"], freq_df["xk"], "o", markersize=6, alpha=0.7)

    # 添加趋势线（可选）
    # 使用numpy的polyfit在log空间拟合直线
    log_k = np.log(freq_df["k"])
    log_xk = np.log(freq_df["xk"])
    slope, intercept = np.polyfit(log_k, log_xk, 1)

    # 绘制趋势线
    trend_x = np.logspace(np.log10(freq_df["k"].min()), np.log10(freq_df["k"].max()), 100)
    trend_y = np.exp(intercept) * trend_x**slope
    plt.loglog(trend_x, trend_y, "--", color="red", label=f"Trend line (slope: {slope:.2f})")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Frequency (k)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Items with Frequency k (xk)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.9)

    plt.tight_layout()
    print(f"幂律分布的估计指数 (gamma): {-slope:.4f}")
    plt.savefig("item_frequency.pdf", dpi=300)

    return freq_df, slope


plot_frequency_log_log()
