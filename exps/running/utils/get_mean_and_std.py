import numpy as np

AUC_list = np.array([0.6949, 0.6945, 0.6940])
LogLoss_list = np.array([0.3634, 0.3632, 0.3647])
Latency_list = np.array([567.52, 551.58, 546.35])
Expert_usage = np.array([2732184, 3678886, 3265566, 2452052])

print(f"AUC: {AUC_list.mean():.4f}±{AUC_list.std():.4f}")
print(f"LogLoss: {LogLoss_list.mean():.4f}±{LogLoss_list.std():.4f}")
print(f"Latency: {Latency_list.mean():.2f}±{Latency_list.std():.2f}")
# print(f"Expert usage: {Expert_usage/Expert_usage.sum()}")
