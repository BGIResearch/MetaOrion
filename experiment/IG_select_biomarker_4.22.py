import os
import json
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import rcParams
from matplotlib import font_manager as fm
from tqdm import tqdm
from collections import defaultdict


def extract_pandisease_biomarkers(weight_base_path, num_splits=5, occurrence_threshold=0.05):
    """
    Extract and filter pan-disease biomarker weights across multiple splits.
    Taxa appearing in fewer samples than the threshold ratio within a split are discarded.

    Args:
        weight_base_path (str): Base directory containing the split folders.
        num_splits (int): Total number of data splits.
        occurrence_threshold (float): Minimum frequency ratio for a taxa to be retained.

    Returns:
        dict: A dictionary containing 'merged_taxa' (lists of weights) and 'mean_weights'.
    """
    tax_weight_split = defaultdict(lambda: defaultdict(list))
    split_filenum = {f'split{x}': 0 for x in range(1, num_splits + 1)}

    print(">>> Extracting PanDisease Biomarkers...")
    for i in range(1, num_splits + 1):
        split = f'split{i}'
        target_dir = os.path.join(weight_base_path, split, 'best_ckpt/result/tax_weight/')

        if not os.path.exists(target_dir):
            continue

        for file in os.listdir(target_dir):
            if not file.endswith('.pkl'):
                continue
            split_filenum[split] += 1
            file_path = os.path.join(target_dir, file)
            with open(file_path, 'rb') as f:
                sample = pickle.load(f)

            for tax, weight in zip(sample['taxa'], sample['weight']):
                tax_weight_split[split][tax].append(weight)

    sorted_tax_split = {}
    for split, tax_weight in tax_weight_split.items():
        total_files = split_filenum[split]
        # Filter taxa based on occurrence threshold
        valid_taxa_weights = {
            tax: weights for tax, weights in tax_weight.items()
            if (len(weights) / total_files) >= occurrence_threshold
        }

        # Calculate mean weight across ALL files in the split (matching original logic)
        tax_mean = {tax: np.sum(weights) / total_files for tax, weights in valid_taxa_weights.items()}
        # Sort by mean weight
        sorted_tax_split[split] = dict(sorted(tax_mean.items(), key=lambda item: item[1], reverse=True))

    merged_data = defaultdict(list)
    for _, taxa_weights in sorted_tax_split.items():
        for taxa, weight in taxa_weights.items():
            merged_data[taxa].append(weight)

    final_result = {
        'merged_taxa': merged_data,
        'mean_weights': {taxa: np.mean(weights) for taxa, weights in merged_data.items()}
    }

    return final_result


def extract_disease_specific_biomarkers(metadata_base_path, weight_base_path, num_splits=5, occurrence_threshold=0.05):
    """
    Categorize samples by disease type using JSON metadata, then extract and average
    biomarker weights specifically for each disease category.

    Args:
        metadata_base_path (str): Path containing the datapath JSON pointers.
        weight_base_path (str): Path containing the actual weight pickle files.
        num_splits (int): Total number of data splits.
        occurrence_threshold (float): Minimum frequency ratio to retain a taxa.

    Returns:
        dict: A nested dictionary structure grouped by disease, containing 'merged_taxa' and 'mean_weights'.
    """
    print("\n>>> Extracting Disease-Specific Biomarkers...")
    sample_class_split = {}
    sample_class_overall = defaultdict(list)

    # 1. Gather sample IDs for each disease class across all splits
    for i in range(1, num_splits + 1):
        split_meta_dir = f'split{i}.change'
        sample_class = defaultdict(list)
        datapath_file = os.path.join(metadata_base_path, split_meta_dir, 'datapath.pandisease.test')

        if not os.path.exists(datapath_file):
            continue

        with open(datapath_file, 'r') as f:
            for line in f:
                json_path = line.strip()
                with open(json_path, 'r') as jf:
                    sample_info = json.load(jf)
                    sample_id = os.path.basename(json_path).replace('.json', '')
                    disease = sample_info['disease_united']
                    sample_class[disease].append(sample_id)

                    if sample_id not in sample_class_overall[disease]:
                        sample_class_overall[disease].append(sample_id)

        sample_class_split[split_meta_dir] = sample_class

    class_sorted_tax = defaultdict(dict)

    # 2. Extract weights grouped by disease and split
    for disease_key in tqdm(list(sample_class_overall.keys()), desc="Processing Diseases"):
        for i in range(1, num_splits + 1):
            split_meta = f'split{i}.change'
            split_weight = f'split{i}'
            tax_weight = defaultdict(list)

            if disease_key not in sample_class_split.get(split_meta, {}):
                continue

            sample_ids = sample_class_split[split_meta][disease_key]
            file_num = len(sample_ids)
            if file_num == 0:
                continue

            target_dir = os.path.join(weight_base_path, split_weight, 'best_ckpt/result/tax_weight/')

            for sample_id in sample_ids:
                pkl_path = os.path.join(target_dir, f"{sample_id}.pkl")
                if os.path.exists(pkl_path):
                    with open(pkl_path, 'rb') as f:
                        sample_data = pickle.load(f)
                    for tax, weight in zip(sample_data['taxa'], sample_data['weight']):
                        tax_weight[tax].append(weight)

            # Filter and calculate mean
            valid_tax_mean = {}
            for tax, weights in tax_weight.items():
                if (len(weights) / file_num) >= occurrence_threshold:
                    valid_tax_mean[tax] = np.sum(weights) / file_num

            class_sorted_tax[disease_key][split_meta] = dict(
                sorted(valid_tax_mean.items(), key=lambda item: item[1], reverse=True)
            )

    # 3. Merge splits to create final disease-specific dictionaries
    merged_data = defaultdict(lambda: defaultdict(list))
    for disease_key, split_data in class_sorted_tax.items():
        for split_name, taxa_weights in split_data.items():
            for taxa, weight in taxa_weights.items():
                merged_data[disease_key][taxa].append(weight)

    final_result = {}
    for disease_key, taxa_data in merged_data.items():
        final_result[disease_key] = {
            'merged_taxa': taxa_data,
            'mean_weights': {taxa: np.mean(weights) for taxa, weights in taxa_data.items()}
        }

    return final_result


def plot_top30_taxa_boxplots_custom(merged_result, target_diseases=None, save_path=None):
    """
    Plot boxplots and stripplots for the top 30 taxa features across 4 specific diseases.
    Layout: 2x2 grid (Top-Left, Top-Right, Bottom-Left, Bottom-Right).
    Ordering: Absolute Top 30 features; Negative features (Blue) sorted by absolute descending,
              followed by Positive features (Red) sorted by actual value ascending.

    Args:
        merged_result (dict): Disease-specific biomarker dictionary generated previously.
        target_diseases (list): List of 4 disease keys to plot.
        save_path (str, optional): Path to save the generated PDF figure.
    """
    if target_diseases is None:
        target_diseases = ['adenoma', 'CRC', 'IBD', 'IBS']

    print("\n>>> Generating Custom Boxplots...")

    rcParams.update({
        'font.family': 'Arial',
        'font.size': 20,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10
    })

    color_pos = '#E58579'
    color_neg = '#8AB1D2'

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 24), squeeze=False)
    axes = axes.flatten()

    for idx, disease in enumerate(target_diseases):
        ax = axes[idx]
        dict_key = disease

        # Handle potential capitalization differences in dictionary keys
        if disease not in merged_result:
            if disease.capitalize() in merged_result:
                dict_key = disease.capitalize()
            elif disease.upper() in merged_result:
                dict_key = disease.upper()
            else:
                ax.axis('off')
                continue

        data = merged_result[dict_key]
        mean_weights = data['mean_weights']

        # 1. Filter Top 30 absolute weights
        top30_items = sorted(mean_weights.items(), key=lambda x: abs(x[1]), reverse=True)[:30]
        top30_taxa_names = [taxa for taxa, _ in top30_items]

        # 2. Segment into negative and positive cohorts
        neg_taxa = [t for t in top30_taxa_names if mean_weights[t] < 0]
        pos_taxa = [t for t in top30_taxa_names if mean_weights[t] >= 0]

        # 3. Sort internally
        neg_taxa = sorted(neg_taxa, key=lambda x: abs(mean_weights[x]), reverse=True)
        pos_taxa = sorted(pos_taxa, key=lambda x: mean_weights[x], reverse=False)

        # 4. Concatenate for final plotting order
        final_taxa_order = neg_taxa + pos_taxa
        taxa_palette = {taxa: color_pos if mean_weights[taxa] > 0 else color_neg for taxa in final_taxa_order}

        # Format data for seaborn
        plot_data = []
        for taxa in final_taxa_order:
            weights = data['merged_taxa'].get(taxa, [])
            for weight in weights:
                plot_data.append({
                    'Taxa': taxa,
                    'Weight': weight,
                    'Mean_Weight': mean_weights[taxa]
                })
        df = pd.DataFrame(plot_data)

        sns.boxplot(data=df, y='Taxa', x='Weight', ax=ax,
                    width=0.6, showfliers=False,
                    hue='Taxa', palette=taxa_palette, dodge=False,
                    order=final_taxa_order)

        sns.stripplot(data=df, y='Taxa', x='Weight', ax=ax,
                      color='black', size=4, alpha=0.7, jitter=0.2,
                      order=final_taxa_order)

        if ax.legend_:
            ax.legend_.remove()

        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, zorder=0)

        # Italicize and clean Y-axis labels
        prop = fm.FontProperties(style='italic', size=14)
        labels = [i[2:].replace('-', ' ').capitalize() for i in final_taxa_order]
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontproperties=prop)

        ax.set_title(disease, fontsize=20, fontweight='bold')
        ax.set_xlabel("Weight", fontsize=16)
        ax.set_ylabel("")
        ax.grid(False)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=800, bbox_inches='tight')
        print(f">>> Figure saved to: {save_path}")
    plt.show()


# ==========================================
# Main Execution Block
# ==========================================
if __name__ == '__main__':
    # Define paths
    WEIGHT_BASE_PATH = '/bgi-seq-model-2/codes/zhangkexin/meta_index/output/llama/v4/PanDisease/Split5.specific.aug.Full.sortabu.ema.12.4/'
    METADATA_BASE_PATH = '/bgi-seq-model-2/datasets/zhangkexin/meta_index/preprocess/metaphlan4/fine-tune/nov.specific.random.5'
    OUT_TXT_PATH = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/features/11.25/sort.train.multidisease.mean/pandisease.biomarker.txt'
    OUT_PDF_PATH = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/figures/finetune/11.25/features/4.disease.biomarker.top30.pdf'

    # 1. Extract PanDisease biomarkers
    pandisease_results = extract_pandisease_biomarkers(WEIGHT_BASE_PATH)
    pandisease_taxa = sorted(pandisease_results['mean_weights'].items(), key=lambda x: x[1], reverse=True)

    # Save to TXT
    os.makedirs(os.path.dirname(OUT_TXT_PATH), exist_ok=True)
    with open(OUT_TXT_PATH, 'w') as f:
        for taxa, weight in pandisease_taxa:
            f.write(f"{taxa}\t{weight}\n")
    print(f">>> PanDisease biomarkers saved to: {OUT_TXT_PATH}")

    # 2. Extract Disease-Specific biomarkers using JSON classifications
    disease_specific_results = extract_disease_specific_biomarkers(METADATA_BASE_PATH, WEIGHT_BASE_PATH)

    # 3. Plot specific diseases (Adenoma, CRC, IBD, IBS)
    plot_top30_taxa_boxplots_custom(
        merged_result=disease_specific_results,
        target_diseases=['adenoma', 'CRC', 'IBD', 'IBS'],
        save_path=OUT_PDF_PATH
    )