#!/usr/bin/env python3
"""
Integrated MetaPhlAn4 v21 vs v22 Database Comparison
Filters out species with both prevalence=0 AND mean_abundance=0 before analysis.
Panels: A(Pie), B(Cumul), C(Prev), D(Abund), E(PrevRank), F(AbundRank)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats


def load_data(v21_file, v22_file):
    v21 = pd.read_csv(v21_file)
    v22 = pd.read_csv(v22_file)
    v21 = v21.rename(columns={'prevalence': 'prevalence_v21', 'mean_abundance': 'mean_abundance_v21'})
    v22 = v22.rename(columns={'prevalence': 'prevalence_v22', 'mean_abundance': 'mean_abundance_v22'})
    
    # Filter out species with both prevalence=0 AND mean_abundance=0
    v21_before = len(v21)
    v21 = v21[~((v21['prevalence_v21'] == 0) & (v21['mean_abundance_v21'] == 0))].copy()
    v21_after = len(v21)
    
    v22_before = len(v22)
    v22 = v22[~((v22['prevalence_v22'] == 0) & (v22['mean_abundance_v22'] == 0))].copy()
    v22_after = len(v22)
    
    print("  v21过滤前: {} -> 过滤后: {} (移除双零: {})".format(v21_before, v21_after, v21_before - v21_after))
    print("  v22过滤前: {} -> 过滤后: {} (移除双零: {})".format(v22_before, v22_after, v22_before - v22_after))
    
    return v21, v22


def get_overlap_data(v21, v22):
    species_v21 = set(v21['species'].tolist())
    species_v22 = set(v22['species'].tolist())
    overlap = species_v21 & species_v22
    unique_v21 = species_v21 - species_v22
    unique_v22 = species_v22 - species_v21
    stats_dict = {
        'v21_total': len(species_v21), 'v22_total': len(species_v22),
        'overlap': len(overlap), 'v21_unique': len(unique_v21), 'v22_unique': len(unique_v22),
        'jaccard': len(overlap) / len(species_v21 | species_v22),
        'v21_coverage': len(overlap) / len(species_v21),
        'v22_coverage': len(overlap) / len(species_v22),
    }
    v21_overlap = v21[v21['species'].isin(overlap)][['species', 'prevalence_v21', 'mean_abundance_v21']].copy()
    v22_overlap = v22[v22['species'].isin(overlap)][['species', 'prevalence_v22', 'mean_abundance_v22']].copy()
    overlap_combined = v21_overlap.merge(v22_overlap, on='species', how='outer')
    
    v21_unique_df = v21[v21['species'].isin(unique_v21)][['species', 'prevalence_v21', 'mean_abundance_v21']].copy()
    v22_unique_df = v22[v22['species'].isin(unique_v22)][['species', 'prevalence_v22', 'mean_abundance_v22']].copy()
    
    return overlap_combined, v21_unique_df, v22_unique_df, stats_dict


# ---- Panel A: Pie Chart ----
def plot_overview_pie(ax, stats):
    sizes = [stats['overlap'], stats['v21_unique'], stats['v22_unique']]
    labels = ['Overlap\n({})'.format(stats['overlap']), 'v21 Unique\n({})'.format(stats['v21_unique']), 'v22 Unique\n({})'.format(stats['v22_unique'])]
    colors = ['#2ca02c', '#1f77b4', '#d62728']
    explode = (0.05, 0, 0)
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
    ax.set_title('A. Species Overlap Overview', fontsize=12, fontweight='bold')


# ---- Panel B: Cumulative Abundance ----
def plot_overlap_cumulative_abundance(overlap_combined, v21, v22, ax):
    data = overlap_combined.dropna(subset=['mean_abundance_v21', 'mean_abundance_v22']).copy()
    total_abundance_v21 = v21['mean_abundance_v21'].sum()
    total_abundance_v22 = v22['mean_abundance_v22'].sum()
    
    sorted_v21 = data.sort_values('mean_abundance_v21', ascending=False).reset_index(drop=True)
    sorted_v21['cumsum_pct'] = sorted_v21['mean_abundance_v21'].cumsum() / total_abundance_v21 * 100
    sorted_v21['species_pct'] = np.arange(1, len(sorted_v21) + 1) / len(sorted_v21) * 100
    
    sorted_v22 = data.sort_values('mean_abundance_v22', ascending=False).reset_index(drop=True)
    sorted_v22['cumsum_pct'] = sorted_v22['mean_abundance_v22'].cumsum() / total_abundance_v22 * 100
    sorted_v22['species_pct'] = np.arange(1, len(sorted_v22) + 1) / len(sorted_v22) * 100
    
    ax.plot(sorted_v21['species_pct'], sorted_v21['cumsum_pct'], linewidth=2.5, color='#1f77b4', label='v21 (n={})'.format(len(sorted_v21)))
    ax.plot(sorted_v22['species_pct'], sorted_v22['cumsum_pct'], linewidth=2.5, color='#d62728', label='v22 (n={})'.format(len(sorted_v22)), alpha=0.8)
    
    max_cum = max(sorted_v21['cumsum_pct'].iloc[-1], sorted_v22['cumsum_pct'].iloc[-1])
    if max_cum >= 50:
        ax.axhline(50, color='gray', linestyle='--', alpha=0.3)
    
    if (sorted_v21['cumsum_pct'] >= 50).any():
        pct_50_v21 = sorted_v21.loc[sorted_v21['cumsum_pct'] >= 50, 'species_pct'].iloc[0]
        ax.scatter([pct_50_v21], [50], color='#1f77b4', s=80, zorder=5, marker='o')
        ax.annotate('v21: {:.1f}%'.format(pct_50_v21), xy=(pct_50_v21, 50), xytext=(pct_50_v21-15, 40), fontsize=9, color='#1f77b4', arrowprops=dict(arrowstyle='->', color='#1f77b4', alpha=0.7))
    
    if (sorted_v22['cumsum_pct'] >= 50).any():
        pct_50_v22 = sorted_v22.loc[sorted_v22['cumsum_pct'] >= 50, 'species_pct'].iloc[0]
        ax.scatter([pct_50_v22], [50], color='#d62728', s=80, zorder=5, marker='s')
        ax.annotate('v22: {:.1f}%'.format(pct_50_v22), xy=(pct_50_v22, 50), xytext=(pct_50_v22+10, 60), fontsize=9, color='#d62728', arrowprops=dict(arrowstyle='->', color='#d62728', alpha=0.7))
    
    ax.set_xlabel('Cumulative Species (% of overlapping)', fontsize=11)
    ax.set_ylabel('Cumulative Abundance (% of total)', fontsize=11)
    ax.set_title('B. Cumulative Abundance: Overlap Species', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, max_cum * 1.15)
    ax.legend(loc='lower right')


# ---- Panel C: Prevalence Comparison ----
def plot_prevalence_comparison_scatter(overlap_combined, ax):
    data = overlap_combined.dropna(subset=['prevalence_v21', 'prevalence_v22'])
    ax.scatter(data['prevalence_v21'], data['prevalence_v22'], alpha=0.5, s=30, c='steelblue', edgecolors='white', linewidth=0.5)
    max_val = max(data['prevalence_v21'].max(), data['prevalence_v22'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='y=x')
    corr, pval = stats.pearsonr(data['prevalence_v21'], data['prevalence_v22'])
    ax.set_xlabel('Prevalence in v21', fontsize=11)
    ax.set_ylabel('Prevalence in v22', fontsize=11)
    ax.set_title('C. Prevalence Comparison (r={:.3f})'.format(corr), fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.95, 'n={}\nr={:.3f}\np={:.2e}'.format(len(data), corr, pval), transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


# ---- Panel D: Abundance Comparison ----
def plot_abundance_comparison_scatter(overlap_combined, ax):
    data = overlap_combined.dropna(subset=['mean_abundance_v21', 'mean_abundance_v22'])
    data = data[(data['mean_abundance_v21'] > 0) & (data['mean_abundance_v22'] > 0)]
    log_v21 = np.log10(data['mean_abundance_v21'])
    log_v22 = np.log10(data['mean_abundance_v22'])
    ax.scatter(log_v21, log_v22, alpha=0.5, s=30, c='forestgreen', edgecolors='white', linewidth=0.5)
    min_val = min(log_v21.min(), log_v22.min())
    max_val = max(log_v21.max(), log_v22.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
    corr, pval = stats.pearsonr(log_v21, log_v22)
    ax.set_xlabel('Log$_{10}$(Mean Abundance) in v21', fontsize=11)
    ax.set_ylabel('Log$_{10}$(Mean Abundance) in v22', fontsize=11)
    ax.set_title('D. Abundance Comparison (r={:.3f})'.format(corr), fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.95, 'n={}\nr={:.3f}\np={:.2e}'.format(len(data), corr, pval), transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


# ---- Panel E: Prevalence Rank ----
def plot_prevalence_rank_comparison(overlap_combined, ax):
    data = overlap_combined.dropna(subset=['prevalence_v21', 'prevalence_v22']).copy()
    data['rank_v21'] = data['prevalence_v21'].rank(ascending=False)
    data['rank_v22'] = data['prevalence_v22'].rank(ascending=False)
    ax.scatter(data['rank_v21'], data['rank_v22'], alpha=0.5, s=30, c='darkorange', edgecolors='white', linewidth=0.5)
    max_rank = max(data['rank_v21'].max(), data['rank_v22'].max())
    ax.plot([0, max_rank], [0, max_rank], 'r--', linewidth=2, label='y=x')
    corr, pval = stats.spearmanr(data['rank_v21'], data['rank_v22'])
    ax.set_xlabel('Rank in v21 (1=most prevalent)', fontsize=11)
    ax.set_ylabel('Rank in v22 (1=most prevalent)', fontsize=11)
    ax.set_title('E. Prevalence Rank (ρ={:.3f})'.format(corr), fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    ax.invert_yaxis()


# ---- Panel F: Abundance Rank ----
def plot_abundance_rank_comparison(overlap_combined, ax):
    data = overlap_combined.dropna(subset=['mean_abundance_v21', 'mean_abundance_v22']).copy()
    data = data[(data['mean_abundance_v21'] > 0) & (data['mean_abundance_v22'] > 0)]
    data['rank_v21'] = data['mean_abundance_v21'].rank(ascending=False)
    data['rank_v22'] = data['mean_abundance_v22'].rank(ascending=False)
    ax.scatter(data['rank_v21'], data['rank_v22'], alpha=0.5, s=30, c='purple', edgecolors='white', linewidth=0.5)
    max_rank = max(data['rank_v21'].max(), data['rank_v22'].max())
    ax.plot([0, max_rank], [0, max_rank], 'r--', linewidth=2, label='y=x')
    corr, pval = stats.spearmanr(data['rank_v21'], data['rank_v22'])
    ax.set_xlabel('Rank in v21 (1=most abundant)', fontsize=11)
    ax.set_ylabel('Rank in v22 (1=most abundant)', fontsize=11)
    ax.set_title('F. Abundance Rank (ρ={:.3f})'.format(corr), fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    ax.invert_yaxis()


def run_integrated_comparison(v21_file, v22_file, output_pdf='integrated_v21_v22_comparison.pdf'):
    print("="*70)
    print("MetaPhlAn4 v21 vs v22 整合比较分析")
    print("="*70)
    print("\n[1/4] 加载并过滤数据...")
    v21, v22 = load_data(v21_file, v22_file)
    print("  v21有效物种数: {}".format(len(v21)))
    print("  v22有效物种数: {}".format(len(v22)))
    
    print("[2/4] 提取重合物种与独有物种数据...")
    overlap_combined, v21_unique_df, v22_unique_df, stats = get_overlap_data(v21, v22)
    print("  重合物种数: {}".format(stats['overlap']))
    print("  v21独有: {} ({:.1f}% covered)".format(stats['v21_unique'], stats['v21_coverage']*100))
    print("  v22独有: {} ({:.1f}% covered)".format(stats['v22_unique'], stats['v22_coverage']*100))
    print("  Jaccard指数: {:.1f}%".format(stats['jaccard']*100))
    
    print("[3/4] 生成可视化并保存为PDF...")
    with PdfPages(output_pdf) as pdf:
        fig = plt.figure(figsize=(18, 11))
        plot_overview_pie(plt.subplot(2, 3, 1), stats)
        plot_overlap_cumulative_abundance(overlap_combined, v21, v22, plt.subplot(2, 3, 2))
        plot_prevalence_comparison_scatter(overlap_combined, plt.subplot(2, 3, 3))
        plot_abundance_comparison_scatter(overlap_combined, plt.subplot(2, 3, 4))
        plot_prevalence_rank_comparison(overlap_combined, plt.subplot(2, 3, 5))
        plot_abundance_rank_comparison(overlap_combined, plt.subplot(2, 3, 6))
        plt.tight_layout()
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        plt.close(fig)
    print("✓ 整合图表已保存: {}".format(output_pdf))
    
    print("[4/4] 保存详细数据...")
    overlap_combined.to_csv('integrated_overlap_species_data.csv', index=False)
    v21_unique_df.to_csv('integrated_v21_unique_species_data.csv', index=False)
    v22_unique_df.to_csv('integrated_v22_unique_species_data.csv', index=False)
    print("✓ overlap物种数据: integrated_overlap_species_data.csv")
    print("✓ v21独有物种数据: integrated_v21_unique_species_data.csv")
    print("✓ v22独有物种数据: integrated_v22_unique_species_data.csv")
    
    print("\n" + "="*70)
    print("分析完成!")
    print("="*70)
    return overlap_combined, v21_unique_df, v22_unique_df, stats


if __name__ == '__main__':
    V21_FILE = 'database_comparison_v21_all_prevalence_abundance.csv'
    V22_FILE = 'database_comparison_v22_all_prevalence_abundance.csv'
    run_integrated_comparison(V21_FILE, V22_FILE)
