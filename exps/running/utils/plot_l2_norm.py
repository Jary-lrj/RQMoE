import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 数据
group1_vanilla = [0.3721, 0.3234, 0.2648, 0.2477]
group1_rq = [0.4708, 0.4152, 0.3379, 0.3047]

group2_vanilla = [0.0977, 0.0586, 0.0518, 0.0458]
group2_rq = [0.1018, 0.0626, 0.0555, 0.0495]

experts = ['Expert 1', 'Expert 2', 'Expert 3', 'Expert 4']
group_data = {
    'Group 1': {'Vanilla': group1_vanilla, 'RQ': group1_rq},
    'Group 2': {'Vanilla': group2_vanilla, 'RQ': group2_rq},
}

# 展平数据便于绘图
records = []
for group, models in group_data.items():
    for model, values in models.items():
        for expert, value in zip(experts, values):
            records.append(
                {'Group': group, 'Expert': expert, 'Model': model, 'L2Norm': value}
            )

df = pd.DataFrame(records)

sns.set_theme(style='white', font_scale=1.05)
plot_configs = [
    ('Group 1', 'Item Embedding L2 Norm Comparison (ml-1m)',
     'plots/l2_norm_comparison_ml_1m.png'),
    ('Group 2', 'Item Embedding L2 Norm Comparison (Beauty)',
     'plots/l2_norm_comparison_beauty.png'),
]

for group_name, title, output_path in plot_configs:
    subset = df[df['Group'] == group_name]
    plt.figure(figsize=(8, 4.5))
    sns.lineplot(
        data=subset,
        x='Expert',
        y='L2Norm',
        hue='Model',
        marker='o',
        linewidth=2,
        palette='Set2',
    )
    plt.ylabel('L2 Norm')
    plt.title(title)
    ymin = max(subset['L2Norm'].min() * 0.9, 0)
    ymax = subset['L2Norm'].max() * 1.1
    plt.ylim(ymin, ymax)
    plt.legend(title='Model', frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
