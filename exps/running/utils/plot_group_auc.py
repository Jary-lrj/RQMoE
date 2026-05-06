import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

hot_auc = [0.7074, 0.7081]
mid_auc = [0.6972, 0.6978]
cold_auc = [0.6475, 0.6640]
ultracold_auc = [0.6250, 0.6303]

model_labels = ['Vanilla', 'RQ']
group_data = {
    'Hot': hot_auc,
    'Mid': mid_auc,
    'Cold': cold_auc,
    'Ultra-Cold': ultracold_auc,
}

records = []
for group, auc_values in group_data.items():
    for model, value in zip(model_labels, auc_values):
        records.append({'Group': group, 'Model': model, 'AUC': value})

df = pd.DataFrame(records)

sns.set_theme(style='white', font_scale=1.1)
plt.figure(figsize=(8, 5))
bar = sns.barplot(
    data=df,
    x='Group',
    y='AUC',
    hue='Model',
    palette='Set2',
    edgecolor='black',
)

for patch in bar.patches:
    height = patch.get_height()
    bar.annotate(
        f'{height:.4f}',
        (patch.get_x() + patch.get_width() / 2, height),
        ha='center',
        va='bottom',
        fontsize=10,
        xytext=(0, 3),
        textcoords='offset points',
    )

plt.ylabel('AUC')
plt.title('Group-wise AUC Comparison on Beauty Dataset')
plt.ylim(0.62, 0.72)
plt.legend(title='Model', frameon=False)
plt.tight_layout()
plt.savefig('plots/group_auc_comparison_beauty.png', dpi=300)
