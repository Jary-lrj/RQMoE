import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

hot_vanilla = [1.0, 0.5438, 0.4013, 0.3773,
               0.2661, 0.2261, 0.1894, 0.1847, 0.1689, 0.1438]
hot_rq = [1.0, 0.5735, 0.2601, 0.2207, 0.194,
          0.1694, 0.143, 0.1379, 0.1354, 0.1265]

mid_vanilla = [1.0, 0.5271, 0.3742, 0.3198,
               0.2236, 0.17, 0.1659, 0.1493, 0.1402, 0.1252]
mid_rq = [1.0, 0.5477, 0.1859, 0.1732, 0.1559,
          0.1388, 0.1248, 0.1186, 0.1165, 0.1106]

cold_vanilla = [1.0, 0.4917, 0.3342, 0.303,
                0.2324, 0.1777, 0.1645, 0.1405, 0.1301, 0.1091]
cold_rq = [1.0, 0.5079, 0.1684, 0.1621, 0.1281,
           0.1233, 0.1097, 0.0994, 0.098, 0.0879]

ultracold_vanilla = [1.0, 0.3998, 0.3473, 0.2691,
                     0.2027, 0.1782, 0.1689, 0.0983, 0.0752, 0.061]
ultracold_rq = [1.0, 0.4126, 0.142, 0.1238,
                0.1111, 0.0764, 0.0688, 0.0671, 0.0453, 0.0404]

vanilla_data = {
    'Hot': hot_vanilla,
    'Mid': mid_vanilla,
    'Cold': cold_vanilla,
    'Ultra-Cold': ultracold_vanilla,
}

rq_data = {
    'Hot': hot_rq,
    'Mid': mid_rq,
    'Cold': cold_rq,
    'Ultra-Cold': ultracold_rq,
}

sv_labels = [f'SV{i}' for i in range(1, 11)]
vanilla_df = pd.DataFrame(vanilla_data, index=sv_labels).T
rq_df = pd.DataFrame(rq_data, index=sv_labels).T

sns.set_theme(style='white', font_scale=1.0)
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

heatmap_cfg = dict(
    annot=True,
    fmt='.3f',
    cmap='YlGnBu',
    vmin=0.05,
    vmax=1.0,
    linewidths=0.5,
    cbar=False,
)

sns.heatmap(vanilla_df, ax=axes[0], **heatmap_cfg)
axes[0].set_title('Vanilla Singular Value Decay')
axes[0].set_xlabel('Singular Value Rank')
axes[0].set_ylabel('Group')

heatmap_cfg['cbar'] = True
sns.heatmap(rq_df, ax=axes[1], **heatmap_cfg)
axes[1].set_title('RQ Singular Value Decay')
axes[1].set_xlabel('Singular Value Rank')
axes[1].set_ylabel('')

plt.savefig('plots/line_svd_heatmaps.png', dpi=300)
plt.close()
