import os
import random
import colorsys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pycountry_convert as pc

# ==========================================
# 1. Global Utilities & Configurations
# ==========================================
TARGET_DISEASES = [
    'IBD', 'healthy', 'CRC', 'T2D', 'OB', 'ACVD', 'AS', 'IGT',
    'metabolic_syndrome', 'melanoma', 'adenoma', 'IBS', 'COVID-19', 'BL', 'CKD'
]


def country_to_continent(country_name):
    """Map a country name to its corresponding continent."""
    try:
        country_code = pc.country_name_to_country_alpha2(country_name)
        continent_code = pc.country_alpha2_to_continent_code(country_code)
        continent_map = {
            'AF': 'Africa', 'AS': 'Asia', 'EU': 'Europe',
            'NA': 'North America', 'SA': 'South America', 'OC': 'Oceania'
        }
        return continent_map.get(continent_code, 'Unknown')
    except:
        return 'Unknown'


def format_disease_name(name):
    """Standardize disease names for display."""
    if name == 'metabolic_syndrome':
        return 'MS'
    if str(name).isupper():
        return name
    return str(name).capitalize()


def format_species_name(name):
    """Format species string from MetaPhlAn to a readable format (e.g., S. aureus)."""
    if name == 'Others':
        return name
    # Extract portion after 's__', replace underscores, and capitalize first letter
    clean_name = name.split('s__')[-1].replace('_', ' ')
    return clean_name[0].upper() + clean_name[1:]


def generate_random_colors(n=100, seed=111):
    """Generate a sequence of distinct, low-saturation pastel colors."""
    random.seed(seed)
    colors = []
    for _ in range(n):
        h = random.random()
        s = random.uniform(0.2, 0.6)  # Low saturation
        v = random.uniform(0.8, 1.0)  # High brightness
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((r, g, b))

    colors_hex = ['#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]
    return sns.color_palette(colors_hex)


def get_clean_profile(path):
    """Load the profile, transpose it, and filter strictly for species-level taxa."""
    X = pd.read_csv(path, index_col=0, sep='\t').T

    # Filter columns that contain '|s__' but not '|t__' (strain level)
    species_cols = [c for c in X.columns if '|s__' in c and not '|t__' in c]
    X = X[species_cols]

    # Simplify column names
    X.columns = [c.split('|s__')[-1] for c in X.columns]
    return X


# ==========================================
# 2. Plotting Functions
# ==========================================
def plot_continent_distribution(metadata_df, save_path):
    """
    Plot a stacked horizontal bar chart showing the continental distribution of
    samples for each disease cohort that has >120 samples.
    """
    print(">>> Plotting continent distribution...")
    df = metadata_df.copy()

    # Process metadata
    df['continent'] = df['country'].apply(country_to_continent)
    df['disease_united'] = df['disease_united'].apply(lambda x: x if x in TARGET_DISEASES else 'others')

    # Calculate group counts and format index
    count_df = df.groupby(['disease_united', 'continent']).size().unstack(fill_value=0)
    count_df.index = [format_disease_name(i) for i in count_df.index]

    # Filter out diseases with <= 120 total samples
    totals = count_df.sum(axis=1)
    mask = totals > 120
    count_df = count_df.loc[mask]
    totals = totals.loc[mask]

    # Order by total sample size descending
    order = totals.sort_values(ascending=False).index
    count_df = count_df.loc[order]
    totals = totals.loc[order]

    # Convert to percentages
    count_df = count_df.div(count_df.sum(axis=1), axis=0) * 100

    # Ensure consistent continent order
    continent_order = ['Asia', 'Europe', 'North America', 'South America', 'Oceania', 'Africa']
    count_df = count_df[[c for c in continent_order if c in count_df.columns]]

    # Plot settings
    plt.style.use('default')
    plt.rcParams['font.family'] = 'Arial'
    plt.rc('font', size=18)
    plt.rcParams['pdf.fonttype'] = 42

    fig, ax = plt.subplots(figsize=(3, 7))
    colors = ['#8AB1D2', '#F5AA61', '#9DD0C7', '#E58579', '#D9BDDB', '#F1DFA4']

    count_df.plot(kind='barh', stacked=True, width=0.7, ax=ax, color=colors[:len(count_df.columns)])
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])

    # Annotate total sample counts
    for i, total in enumerate(totals):
        ax.text(102, i, f'n={total}', va='center', fontsize=18)

    # Aesthetics
    ax.set_xlabel('Percent')
    ax.set_ylabel('Disease')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_yaxis()

    ax.legend(
        bbox_to_anchor=(1.8, -0.1), ncol=3, frameon=False,
        handlelength=1.0, handleheight=0.8, handletextpad=0.3,
        borderpad=0.3, labelspacing=0.3
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_species_abundance_stackbar(profile_df, metadata_df, disease_order, save_path):
    """
    Plot a stacked horizontal bar chart showing the Top 20 species average relative
    abundances across specified disease groups.
    """
    print(">>> Plotting species abundance...")
    # Align Data
    common_samples = metadata_df.index.intersection(profile_df.index)
    merged = profile_df.loc[common_samples].copy()
    merged['disease_group'] = metadata_df.loc[common_samples, 'disease_united']

    # Calculate group averages
    avg_abundance = merged.groupby('disease_group').mean()

    # Identify Top 20 species globally
    top_20_desc = avg_abundance.mean().sort_values(ascending=False).head(20).index.tolist()
    # Reverse to ascending order so Top 1 is plotted last (closest to 0)
    top_20_asc = top_20_desc[::-1]

    # Prepare plotting dataframe
    plot_data = avg_abundance[top_20_asc].copy()
    plot_data['Others'] = 100 - plot_data.sum(axis=1)

    # Reorder rows and format column names
    plot_data = plot_data.loc[disease_order]
    plot_data.columns = [format_species_name(c) for c in plot_data.columns]

    # Plot settings
    plt.style.use('default')
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 18, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(8, 7))

    # Palette configuration (20 random + specific core colors)
    color_palette = list(generate_random_colors(14, seed=111)) + [
        '#8AB1D2', '#F5AA61', '#9DD0C7', '#E58579', '#D9BDDB', '#F1DFA4', '#D9D9D9'
    ]

    plot_data.plot(
        kind='barh', stacked=True, ax=ax, color=color_palette,
        width=0.7, edgecolor='white', linewidth=0.3
    )

    # Aesthetics
    ax.set_xlabel('Average Relative Abundance (%)', fontsize=18, labelpad=10)
    ax.set_ylabel('')
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    sns.despine(ax=ax, top=True, right=True, trim=False)

    # Custom Legend: Reverse order to show Top 1 at the top, and italicize species names
    handles, labels = ax.get_legend_handles_labels()
    new_handles = handles[:-1][::-1] + [handles[-1]]
    new_labels = labels[:-1][::-1] + [labels[-1]]

    legend = ax.legend(
        new_handles, new_labels,
        bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False,
        fontsize=16, title_fontsize=18, handlelength=1.0, handleheight=0.8,
        handletextpad=0.3, borderpad=0.3, labelspacing=0.3
    )

    for text in legend.get_texts():
        if text.get_text() != 'Others':
            text.set_style('italic')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=800)
    plt.show()


# ==========================================
# 3. Main Execution Block
# ==========================================
if __name__ == '__main__':
    # Define Paths
    METADATA_PATH = '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/curated_LiS_CRC_20204_v0908.train_test.phe'
    PROFILE_PATH = '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/curated_LiS_CRC_20204_v1121.train_test.profile'
    SAVE_DIR = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/figures/finetune/11.25/'

    # 1. Load and Clean Metadata
    metadata_df = pd.read_csv(METADATA_PATH, sep='\t', index_col=0)
    # Remove specific unneeded dataset combination
    metadata_df = metadata_df[~((metadata_df['From'] == 'LiS') & (metadata_df['project'] == 'QinJ_2012'))]

    # Standardize disease nomenclature in metadata
    metadata_df['disease_united'] = metadata_df['disease_united'].apply(
        lambda x: x if x in TARGET_DISEASES else 'others')
    metadata_df['disease_united'] = metadata_df['disease_united'].apply(format_disease_name)

    # 2. Execute Plot 1: Continental Distribution
    continent_save_path = os.path.join(SAVE_DIR, 'finetuine.data.statics.percent.v2.pdf')
    plot_continent_distribution(metadata_df, continent_save_path)

    # 3. Load Profile Data
    profile_data = get_clean_profile(PROFILE_PATH)

    # 4. Execute Plot 2: Species Abundance
    # Determine the sorting order based on cohort sample size
    disease_order = metadata_df['disease_united'].value_counts().index
    abundance_save_path = os.path.join(SAVE_DIR, 'finetuine.data.species.statics.v3.pdf')

    plot_species_abundance_stackbar(profile_data, metadata_df, disease_order, abundance_save_path)

    print("\n>>> All visualization tasks completed successfully.")