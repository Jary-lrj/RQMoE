from scipy import stats
import numpy as np


def calculate_p_value(mean1, std1, n1, mean2, std2, n2):
    # 使用 scipy 的统计函数直接从统计数据计算 t值 和 p值
    # equal_var=False 表示我们不假设两个模型的方差相等 (Welch's t-test)，这在对比不同模型时更严谨
    t_stat, p_value = stats.ttest_ind_from_stats(mean1, std1, n1, mean2, std2, n2, equal_var=False)
    return p_value


# === 填入你的数据 (来自你的截图) ===
# 假设样本量 N=5 (即你跑了5次种子取平均)

# Group 1: w/o RQ (基线)
m1, s1, n1 = 0.6863, 0.0009, 5

# Group 2: Both (你的方法)
m2, s2, n2 = 0.6847, 0.0005, 5

# 计算
p = calculate_p_value(m1, s1, n1, m2, s2, n2)

print(f"P-value: {p}")

# 判断显著性
if p < 0.001:
    print("结果: *** 极度显著 (p < 0.001)")
elif p < 0.01:
    print("结果: ** 非常显著 (p < 0.01)")
elif p < 0.05:
    print("结果: * 显著 (p < 0.05)")
else:
    print("结果: 不显著 (p >= 0.05)")
