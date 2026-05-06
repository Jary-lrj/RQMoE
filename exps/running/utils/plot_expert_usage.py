import matplotlib.pyplot as plt
import seaborn as sns

vanilla_usage = [0.2547, 0.1596, 0.2169, 0.3687]
rq_usage = [0.3229, 0.2755, 0.2026, 0.1990]
x_labels = ['Expert 1', 'Expert 2', 'Expert 3', 'Expert 4']

sns.set_theme(style='white')
plt.figure(figsize=(10, 6))
sns.lineplot(x=x_labels, y=vanilla_usage, marker='o', label='Vanilla')
sns.lineplot(x=x_labels, y=rq_usage, marker='o', label='RQ')
plt.xlabel('Expert')
plt.ylabel('Usage')
plt.title('Expert Usage Comparison on ml-1m Dataset')
plt.legend()
plt.tight_layout()
plt.savefig('plots/expert_usage_comparison.png', dpi=300)
