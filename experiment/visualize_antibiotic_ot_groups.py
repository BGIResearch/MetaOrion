#!/usr/bin/env python
"""Visualize OT distance groups for the antibiotic longitudinal cohort."""

from __future__ import annotations

import os
import argparse
from itertools import combinations
from pathlib import Path
from typing import List, Tuple
from matplotlib.lines import Line2D
from adjustText import adjust_text

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

plt.rc("font", size=22)
plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42

DATA_DIR = (
    "/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/"
    "Antibiotic.intervention.longi.complete/OT/C.last.cutoff70/"
)

DEFAULT_OUTPUT_DIR = (
    "/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/figures/finetune/11.25/networks/4.1/test.change.5.9/antibiotic.longi/ot.group/"
)

GROUP_ORDER = [
    "low_BA__low_CA",
    "low_BA__high_CA",
    "high_BA__low_CA",
    "high_BA__high_CA",
]

GROUP_COLORS = {
    "low_BA__low_CA": "#4C78A8",
    "low_BA__high_CA": "#F58518",
    "high_BA__low_CA": "#54A24B",
    "high_BA__high_CA": "#E45756",
}

GROUP_LABELS = {
    "low_BA__low_CA": "low BA\nlow CA",
    "low_BA__high_CA": "low BA\nhigh CA",
    "high_BA__low_CA": "high BA\nlow CA",
    "high_BA__high_CA": "high BA\nhigh CA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot OT distance group differences.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--result-file", default=DATA_DIR)
    return parser.parse_args()


def load_results(args: argparse.Namespace) -> tuple[pd.DataFrame, Path]:
    output_dir = Path(args.output_dir)
    result_file = Path(os.path.join(args.result_file, "individual_ot_groups.csv"))
    if not result_file.exists():
        raise FileNotFoundError(f"result file does not exist: {result_file}")

    df = pd.read_csv(result_file)
    required_cols = {"individual_id", "ot_ba", "ot_ca", "group"}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise ValueError(f"missing columns in result file: {sorted(missing_cols)}")

    df = df.dropna(subset=["ot_ba", "ot_ca", "group"]).copy()
    df["group"] = pd.Categorical(df["group"], categories=GROUP_ORDER, ordered=True)
    df = df.sort_values(["group", "individual_id"])
    return df, output_dir


def save_scatter(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))

    for group in GROUP_ORDER:
        sub = df[df["group"] == group]
        if sub.empty:
            continue
        ax.scatter(
            sub["ot_ba"],
            sub["ot_ca"],
            s=110,
            alpha=0.82,
            color=GROUP_COLORS[group],
            edgecolor="white",
            linewidth=0.5,
            label=f"{GROUP_LABELS[group].replace(chr(10), ' ')} (n={len(sub)})",
        )

    ba_cutoff = float(df["ba_cutoff"].iloc[0]) if "ba_cutoff" in df.columns else float(df["ot_ba"].median())
    ca_cutoff = float(df["ca_cutoff"].iloc[0]) if "ca_cutoff" in df.columns else float(df["ot_ca"].median())
    ax.axvline(ba_cutoff, color="#333333", linestyle="--", linewidth=1)
    ax.axhline(ca_cutoff, color="#333333", linestyle="--", linewidth=1)
    ax.set_xlabel(r"$OT_{A \to B}$")
    ax.set_ylabel(r"$OT_{A \to C}$")
    ax.legend(frameon=False, fontsize=11, loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "ot_ba_vs_ot_ca_scatter.pdf", dpi=800)
    plt.show()
    plt.close(fig)


def jitter_positions(n: int, center: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return center + rng.normal(0, 0.045, size=n)


def pvalue_to_stars(pvalue: float) -> str:
    """Return significance stars for one pairwise p value."""

    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return ""


def pairwise_significance(df: pd.DataFrame, distance_col: str) -> List[Tuple[int, int, str]]:
    """Run pairwise Mann-Whitney U tests between plotted groups."""

    try:
        from scipy.stats import mannwhitneyu
    except Exception:
        return []

    rows: List[Tuple[int, int, str]] = []
    for idx_1, idx_2 in combinations(range(len(GROUP_ORDER)), 2):
        group_1 = GROUP_ORDER[idx_1]
        group_2 = GROUP_ORDER[idx_2]
        x = df.loc[df["group"] == group_1, distance_col].dropna().to_numpy()
        y = df.loc[df["group"] == group_2, distance_col].dropna().to_numpy()
        if len(x) == 0 or len(y) == 0:
            continue
        _, pvalue = mannwhitneyu(x, y, alternative="two-sided")
        stars = pvalue_to_stars(float(pvalue))
        if stars:
            rows.append((idx_1, idx_2, stars))
    return rows


def add_significance_bars(ax: plt.Axes, df: pd.DataFrame, distance_col: str) -> None:
    """Add pairwise significance brackets above a boxplot axis."""

    pairs = pairwise_significance(df, distance_col)
    if not pairs:
        return

    y_values = df[distance_col].dropna().to_numpy()
    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))
    y_range = max(y_max - y_min, abs(y_max) * 0.1, 1e-6)
    step = y_range * 0.10
    bar_height = y_range * 0.025

    for level, (idx_1, idx_2, stars) in enumerate(pairs):
        y = y_max + step * (level + 1)
        ax.plot(
            [idx_1, idx_1, idx_2, idx_2],
            [y, y + bar_height, y + bar_height, y],
            color="#222222",
            linewidth=1.0,
        )
        ax.text(
            (idx_1 + idx_2) / 2,
            y + bar_height,
            stars,
            ha="center",
            va="bottom",
            color="#222222",
        )

    ax.set_ylim(top=y_max + step * (len(pairs) + 1.8))


def save_boxplots(df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharex=True)

    for ax, distance_col, title in zip(
        axes,
        ["ot_ba", "ot_ca"],
        ["OT distance: B -> A", "OT distance: C -> A"],
    ):
        values: List[np.ndarray] = [
            df.loc[df["group"] == group, distance_col].to_numpy() for group in GROUP_ORDER
        ]
        ax.boxplot(
            values,
            positions=np.arange(len(GROUP_ORDER)),
            widths=0.56,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#222222", "linewidth": 1.2},
            boxprops={"facecolor": "#F2F2F2", "edgecolor": "#555555", "linewidth": 0.9},
            whiskerprops={"color": "#555555", "linewidth": 0.9},
            capprops={"color": "#555555", "linewidth": 0.9},
        )

        for idx, group in enumerate(GROUP_ORDER):
            sub = df[df["group"] == group]
            if sub.empty:
                continue
            ax.scatter(
                jitter_positions(len(sub), idx, seed=idx + (0 if distance_col == "ot_ba" else 100)),
                sub[distance_col],
                s=18,
                alpha=0.78,
                color=GROUP_COLORS[group],
                edgecolor="white",
                linewidth=0.35,
            )

        # ax.set_title(title)
        ax.set_xticks(np.arange(len(GROUP_ORDER)))
        ax.set_xticklabels([GROUP_LABELS[group] for group in GROUP_ORDER])
        ax.set_ylabel("OT distance")
        ax.grid(axis="y", alpha=0.2)
        add_significance_bars(ax, df, distance_col)

    fig.tight_layout()
    fig.savefig(output_dir / "ot_distance_boxplots_by_group.pdf")
    plt.close(fig)


def save_group_summary(df: pd.DataFrame, output_dir: Path) -> None:
    summary = (
        df.groupby("group", observed=False)
        .agg(
            n_individual=("individual_id", "count"),
            ot_ba_mean=("ot_ba", "mean"),
            ot_ba_median=("ot_ba", "median"),
            ot_ba_std=("ot_ba", "std"),
            ot_ca_mean=("ot_ca", "mean"),
            ot_ca_median=("ot_ca", "median"),
            ot_ca_std=("ot_ca", "std"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "ot_distance_group_summary.csv", index=False)


def save_stat_tests(df: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    try:
        from scipy.stats import kruskal, mannwhitneyu
    except Exception:
        pd.DataFrame(
            [{"note": "scipy is not available; statistical tests were skipped."}]
        ).to_csv(output_dir / "ot_distance_group_tests.csv", index=False)
        return

    for distance_col in ["ot_ba", "ot_ca"]:
        arrays = [
            df.loc[df["group"] == group, distance_col].dropna().to_numpy()
            for group in GROUP_ORDER
        ]
        arrays_nonempty = [array for array in arrays if len(array) > 0]
        if len(arrays_nonempty) >= 2:
            stat, pvalue = kruskal(*arrays_nonempty)
            rows.append(
                {
                    "distance": distance_col,
                    "test": "Kruskal-Wallis",
                    "group_1": "all_groups",
                    "group_2": "",
                    "statistic": stat,
                    "pvalue": pvalue,
                }
            )

        for group_1, group_2 in combinations(GROUP_ORDER, 2):
            x = df.loc[df["group"] == group_1, distance_col].dropna().to_numpy()
            y = df.loc[df["group"] == group_2, distance_col].dropna().to_numpy()
            if len(x) == 0 or len(y) == 0:
                continue
            stat, pvalue = mannwhitneyu(x, y, alternative="two-sided")
            rows.append(
                {
                    "distance": distance_col,
                    "test": "Mann-Whitney U",
                    "group_1": group_1,
                    "group_2": group_2,
                    "statistic": stat,
                    "pvalue": pvalue,
                }
            )

    pd.DataFrame(rows).to_csv(output_dir / "ot_distance_group_tests.csv", index=False)


def save_group_mdi(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Load MDI and OT data, merge them by individual, filter for target groups,
    and plot a longitudinal line plot with confidence intervals.
    """

    target_groups = ['high_BA__low_CA', 'high_BA__high_CA']
    palette = {
        'high_BA__low_CA': '#3BA997',
        'high_BA__high_CA': '#B291B5'
    }

    df_ot = df
    df_mdi = pd.read_csv('/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/Antibiotic.intervention.longi.complete/shared.biomarker.mdi.csv')

    df = pd.merge(df_mdi, df_ot[['individual_id', 'group']],
                  left_on='subject', right_on='individual_id', how='inner')
    df_filtered = df[df['group'].isin(target_groups)].copy()
    df_filtered['group'] = df_filtered['group'].astype(str)

    canon_order = sorted(df_filtered['canon'].dropna().unique())
    df_filtered['canon'] = pd.Categorical(df_filtered['canon'], categories=canon_order, ordered=True)

    plt.figure(figsize=(12.3, 7))
    sns.lineplot(
        data=df_filtered, x='canon', y='MNDI', hue='group',
        palette=palette, marker='o', errorbar=('ci', 95),
        linewidth=2.5, markersize=10
    )

    plt.tick_params(axis='x', rotation=45)
    plt.xlabel('Canon Stages (Time)')
    plt.ylabel('MDI')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.grid(True, linestyle='--', linewidth=1.5, alpha=0.5, zorder=0)

    plt.savefig(output_dir / 'ot_group_mdi_lineplot_6.1.pdf', dpi=800)
    plt.show()
    plt.close()



def save_volcano_plot(output_dir: Path) -> None:
    data_csv_path = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/Antibiotic.intervention.longi.complete/OT/C.last.cutoff70/ot_driver_taxa/source_driver_taxa_persistent_shift.csv'
    biomarker_txt_path = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/features/11.25/sort.multidisease.mean/pandisease.biomarker.txt'
    top30_pos_csv_path = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/pandisease/top30.positive.o3.csv'
    top30_neg_csv_path = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/pandisease/top30.negative.o3.csv'

    # 1. 统一全局绘图风格
    plt.rcParams['font.family'] = 'Arial'
    plt.rc('font', size=20)
    plt.rcParams['pdf.fonttype'] = 42

    # 2. 加载待分析的表
    df = pd.read_csv(data_csv_path)

    # 3. 加载 Biomarker 表格，打上 PR-increasing / decreasing 标签
    biomarker = pd.read_csv(biomarker_txt_path, sep='\t', header=None, names=['taxa', 'score'])

    def get_pr_type(score):
        if score > 0:
            return 'PR-increasing'
        elif score < 0:
            return 'PR-decreasing'
        return 'Unknown'

    biomarker['pr_type'] = biomarker['score'].apply(get_pr_type)
    biomarker_dict = dict(zip(biomarker['taxa'], biomarker['pr_type']))

    # 将 PR 类型映射到主表
    df['pr_type'] = df['taxa'].map(biomarker_dict)

    # 过滤掉不存在于 biomarker 表里的菌
    df = df.dropna(subset=['pr_type'])
    df = df[df['pr_type'] != 'Unknown'].copy()

    # 计算用于纵轴的 -log10(p-value)
    # 加上 1e-300 防止 p=0 时取对数报错 (或者用你之前的 1e-5)
    df['neg_log_p'] = -np.log10(df['fdr_q'] + 1e-5)

    # 标记显著性 (这里默认采用 p < 0.05 为显著)
    df['is_sig'] = df['fdr_q'] < 0.1

    # 提取 top30 名字，并统一格式为 's-xxxx'
    top30_pos_taxa = set(pd.read_csv(top30_pos_csv_path, index_col=0).index)
    top30_neg_taxa = set(pd.read_csv(top30_neg_csv_path, index_col=0).index)

    top30_pos_taxa = ['s-' + i.lower().replace(" ", "-") for i in top30_pos_taxa]
    top30_neg_taxa = ['s-' + i.lower().replace(" ", "-") for i in top30_neg_taxa]

    # 将需要打标签的集合合并
    label_set = set(top30_pos_taxa + top30_neg_taxa)

    # 4. 初始化画布
    fig, ax = plt.subplots(figsize=(6, 6))
    texts = []  # 用于收集需要调整排版的文本对象

    # 5. 逐点绘制
    for idx, row in df.iterrows():
        # 根据类型和显著性分配颜色
        if row['is_sig']:
            color = '#E44A33' if row['pr_type'] == 'PR-increasing' else '#4DBAD6'
        else:
            color = 'gray'

        # 统一使用圆形，并控制透明度和层级
        marker = 'o'
        alpha = 0.8 if row['is_sig'] else 0.4
        zorder = 2 if row['is_sig'] else 1

        # 绘制散点
        ax.scatter(row['mean_diff_a_minus_b'], row['neg_log_p'],
                   c=color, marker=marker, alpha=alpha, edgecolors='w', s=110, zorder=zorder)

        # 【核心新增】提取显著并在Top30里的菌株做名字标注
        if row['is_sig'] and row['taxa'] in label_set:
            # 简化显示名字：去掉前缀 's-' 使得版面更清爽
            display_name = row['taxa'].replace('s-', '')
            t = ax.text(row['mean_diff_a_minus_b'], row['neg_log_p'],
                        display_name, fontsize=12, fontweight='normal', color='black',
                        zorder=3)
            texts.append(t)

    # 6. 添加辅助线
    ax.axhline(-np.log10(0.1), color='k', linestyle='--', linewidth=1.5, alpha=0.5, zorder=0)
    ax.axvline(0, color='k', linestyle='--', linewidth=1.5, alpha=0.5, zorder=0)
    # 7. 坐标轴与标题
    ax.set_xlabel(r"$\Delta OT_{A \to C}$ (Persistent Shift - Recovery Fast)")
    ax.set_ylabel('-log$_{10}$ (Q-value)')

    # 8. 自定义图例
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E44A33', markersize=12,
               label='Significant (PR-increasing)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4DBAD6', markersize=12,
               label='Significant (PR-decreasing)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=12, label='Non-significant')
    ]
    # 图例位置放在右上角或左上角，取消边框
    ax.legend(handles=legend_elements, loc='upper left', frameon=False, fontsize=14)

    # 9. 智能防重叠文本排版
    if texts:
        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle='-', color='gray', lw=1.0, alpha=0.7),  # 带灰色的连接线
                    force_text=(0.5, 0.5),  # 文字间的排斥力
                    force_points=(0.2, 0.2),  # 文字与点间的排斥力
                    expand_points=(1.2, 1.2))  # 安全距离

    plt.tight_layout()
    plt.savefig(
        output_dir / 'diver_taxa_volcano_labeled.pdf',
        dpi=800)
    plt.show()

def main() -> int:
    args = parse_args()
    df, output_dir = load_results(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_scatter(df, output_dir)
    # save_boxplots(df, output_dir)
    # save_group_summary(df, output_dir)
    # save_stat_tests(df, output_dir)
    save_group_mdi(df, output_dir)
    save_volcano_plot(output_dir)

    print(f"Loaded individuals: {len(df)}")
    print(f"Figures and tables written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())






