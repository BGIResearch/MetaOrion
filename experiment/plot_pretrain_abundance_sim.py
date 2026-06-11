import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. Data Loading and Preprocessing
# ==========================================
lengths = np.load(
    '/bgi-seq-model-2/codes/zhangkexin/meta_index/output/llama/v4/pretrain/abundance_sim/crc.owen.seq.len.npy')
similarities = np.load(
    '/bgi-seq-model-2/codes/zhangkexin/meta_index/output/llama/v4/pretrain/abundance_sim/crc.owen.sample.pad.sim.npy')

df = pd.DataFrame({'Length': lengths, 'Similarity': similarities})

# Calculate 5 quantile boundaries (0%, 20%, 40%, 60%, 80%, 100%)
quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
bin_edges = df['Length'].quantile(quantiles).values
group_labels = [f'Q{i + 1}' for i in range(5)]  # Q1-Q5

# Apply grouping
df['Length_Group'] = pd.cut(
    df['Length'],
    bins=bin_edges,
    labels=group_labels,
    include_lowest=True
)
df = df.sort_values('Length_Group')

# Count samples per group
group_counts = df['Length_Group'].value_counts().sort_index()

# ==========================================
# 2. Global Plot Settings
# ==========================================
plt.rcParams['font.family'] = 'Arial'
plt.rc('font', size=20)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({
    'font.size': 23,
    'axes.titlesize': 23,
    'axes.labelsize': 23,
    'xtick.labelsize': 20,
    'ytick.labelsize': 23,
    'legend.fontsize': 20
})

# ==========================================
# 3. Figure Setup
# ==========================================
fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(8, 9),
    gridspec_kw={'height_ratios': [3, 2]},
    facecolor='white'
)

custom_palette = sns.color_palette(["#8FB4DC", "#F5AA61", "#70CDBE", "#AC99D2", "#EB7E60", "#FFDD8E"])
n_groups = len(df['Length_Group'].unique())
point_colors = custom_palette[:n_groups]

# ==========================================
# 4. Top Panel (ax1): Violin + Strip + Box Plot
# ==========================================
# Base layer: Violin plot
sns.violinplot(
    x='Length_Group', y='Similarity', data=df,
    palette=custom_palette, inner=None, linewidth=0, alpha=0.6, zorder=1, ax=ax1
)

# Middle layer: Strip plot
sns.stripplot(
    x='Length_Group', y='Similarity', data=df,
    palette=point_colors, alpha=0.7, size=4, jitter=0.2, zorder=2, ax=ax1
)

# Top layer: Box plot (drawn per group to match median line colors to violin bodies)
for i, color in enumerate(point_colors):
    group_data = df[df['Length_Group'] == group_labels[i]]['Similarity']

    sns.boxplot(
        x=[group_labels[i]] * len(group_data),
        y=group_data, width=0.2, showcaps=False,
        boxprops=dict(facecolor='white', linewidth=2.2, color='white'),
        whiskerprops=dict(linewidth=2.2, color='white'),
        medianprops=dict(linewidth=3, color=color),
        showfliers=False, zorder=3, ax=ax1
    )

# Ax1 styling
new_labels = [f"{group_labels[i]}\nN={group_counts[group_labels[i]]}" for i in range(len(group_labels))]
ax1.set_xticks(range(len(group_labels)))
ax1.set_xticklabels(new_labels)
ax1.set_ylim(0.8, 1)
ax1.set_yticks(np.arange(0.8, 1.01, 0.05))
ax1.set_ylabel('Similarity', labelpad=10)
ax1.set_xlabel('Length Percentile Groups', labelpad=10)

# ==========================================
# 5. Bottom Panel (ax2): Histogram
# ==========================================
n, bins, patches = ax2.hist(
    df['Length'], bins=30, color='#4e79a7', alpha=0.8, edgecolor='white', linewidth=1.2
)

# Calculate percentiles
percentiles = [0, 20, 40, 60, 80, 100]
percentile_values = np.percentile(df['Length'], percentiles)

# Add percentile lines and annotations
for i, (p_val, perc) in enumerate(zip(percentile_values, percentiles)):
    ax2.axvline(x=p_val, color='red', linestyle='--', alpha=0.9, linewidth=1)

    # Stagger vertical positions for labels to avoid overlap
    y_pos = n.max() * 0.85 if i % 2 == 0 else n.max() * 0.75
    ax2.text(
        p_val, y_pos, f'{perc}%\n({int(p_val)})',
        ha='center', va='center', fontsize=8, color='red',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9)
    )

# Ax2 styling
ax2.set_ylabel('Frequency Count', labelpad=10)
ax2.set_xlabel('Sequence Length', labelpad=10)

# ==========================================
# 6. Final Layout & Output
# ==========================================
plt.tight_layout()
# plt.savefig('/home/share/huadjyin/home/zhangkexin2/code/meta_index/experiment/results/pretrain/owen_sample_pad_vilolin_sim_length.pdf', bbox_inches='tight')
plt.show()