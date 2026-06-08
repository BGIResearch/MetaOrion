import os
import numpy as np
import pandas as pd
from scipy import stats
def mannwhitneyu_effect_size(group1, group2):
    """
    Calculate the effect size (r) for the Mann-Whitney U test.
    Args:
        group1 (array-like): Abundance values for the first group.
        group2 (array-like): Abundance values for the second group.
    Returns:
        float: The calculated effect size r, or np.nan if either group is empty.
    """
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return np.nan
    u = stats.mannwhitneyu(group1, group2).statistic
    r = 1 - (2 * u) / (n1 * n2)
    return r
def case_ctrl_abundance_wilcox(X, case_sample, ctrl_sample, raw_disease, biomarker_list, save_dir):
    """
    Evaluate the abundance and statistical significance of biomarkers between case and control groups.
    Records the number of non-zero samples (prevalence), while using ALL samples (including zeros)
    for calculating mean, median, and statistical tests.
    Args:
        X (pd.DataFrame): The microbiome abundance profile (samples as rows, taxa as columns).
        case_sample (list): List of sample IDs belonging to the case/disease group.
        ctrl_sample (list): List of sample IDs belonging to the control/healthy group.
        raw_disease (str): The name of the target disease being processed.
        biomarker_list (list): List of biomarker taxa to evaluate.
        save_dir (str): Directory path to save the output CSV results.
    """
    results = {
        'species': [],
        'case.num': [],  # Number of non-zero samples (Prevalence)
        'case.median': [],  # Median of the full set (including zeros)
        'case.mean': [],  # Mean of the full set (including zeros)
        'ctrl.num': [],  # Number of non-zero samples (Prevalence)
        'ctrl.median': [],  # Median of the full set (including zeros)
        'ctrl.mean': [],  # Mean of the full set (including zeros)
        'p.value': [],  # Mann-Whitney U test p-value on the full set
        'effect.size': []  # Effect size (r) on the full set
    }
    # Ensure intersection of valid samples to prevent KeyError when querying the dataframe
    valid_cases = list(set(case_sample).intersection(X.index))
    valid_ctrls = list(set(ctrl_sample).intersection(X.index))
    for biomarker in biomarker_list:
        if biomarker not in X.columns:
            print(f'Warning: Biomarker {biomarker} does not exist in the profile.')
            continue
        try:
            # Extract abundance arrays (including zeros)
            case_abu = np.array([])
            ctrl_abu = np.array([])
            if valid_cases:
                case_data = X.loc[valid_cases, biomarker]
                if isinstance(case_data, pd.DataFrame):
                    case_abu = case_data.sum(axis=1).values
                else:
                    case_abu = case_data.values
            if valid_ctrls:
                ctrl_data = X.loc[valid_ctrls, biomarker]
                if isinstance(ctrl_data, pd.DataFrame):
                    ctrl_abu = ctrl_data.sum(axis=1).values
                else:
                    ctrl_abu = ctrl_data.values
            # Total length (for conditional checks and statistical functions)
            case_num_total = len(case_abu)
            ctrl_num_total = len(ctrl_abu)
            # Highlight: Calculate the number of non-zero samples (Prevalence) independently
            case_nonzero_num = int(np.sum(case_abu > 0)) if case_num_total > 0 else 0
            ctrl_nonzero_num = int(np.sum(ctrl_abu > 0)) if ctrl_num_total > 0 else 0
            # Safely calculate independent metrics based on TOTAL samples (including 0s)
            case_median = np.median(case_abu) if case_num_total > 0 else np.nan
            case_mean = np.mean(case_abu) if case_num_total > 0 else np.nan
            ctrl_median = np.median(ctrl_abu) if ctrl_num_total > 0 else np.nan
            ctrl_mean = np.mean(ctrl_abu) if ctrl_num_total > 0 else np.nan
            p_value = np.nan
            effect_size_r = np.nan
            # Only run statistical tests if both groups have total data
            if case_num_total > 0 and ctrl_num_total > 0:
                _, p_value = stats.mannwhitneyu(case_abu, ctrl_abu, alternative='two-sided')
                effect_size_r = mannwhitneyu_effect_size(case_abu, ctrl_abu)
            # Format species name for cleaner output
            species_name = biomarker[2:].replace('-', ' ').capitalize()
            # Append results
            results['species'].append(species_name)
            results['case.num'].append(case_nonzero_num)  # Save prevalence (non-zero count)
            results['ctrl.num'].append(ctrl_nonzero_num)  # Save prevalence (non-zero count)
            results['case.median'].append(case_median)  # Save full-set median
            results['ctrl.median'].append(ctrl_median)  # Save full-set median
            results['case.mean'].append(case_mean)  # Save full-set mean
            results['ctrl.mean'].append(ctrl_mean)  # Save full-set mean
            results['p.value'].append(p_value)  # Save full-set p-value
            results['effect.size'].append(effect_size_r)  # Save full-set effect size
        except Exception as e:
            print(f'Unknown error processing {biomarker} for {raw_disease}: {e}')
            continue
    # Export results to CSV
    biomarker_results_df = pd.DataFrame(results)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{raw_disease}.healthy.wilcox.csv')
    biomarker_results_df.to_csv(save_path, index=False)
if __name__ == '__main__':
    # ==========================================
    # Configuration and Data Loading
    # ==========================================
    PROFILE_DIR = '/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/'
    FEATURE_DIR = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/features/11.25/sort.multidisease.mean/'
    OUTPUT_DIR = '/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/features/11.25/sort.biomarker.wilcox.test.tmp.6.4/'
    DISEASE_MAPPING = {
        'Healthy': 'healthy', 'IBD': 'IBD', 'CRC': 'CRC', 'T2D': 'T2D',
        'MS': 'metabolic_syndrome', 'Others': 'others', 'AS': 'AS', 'OB': 'OB',
        'IBS': 'IBS', 'IGT': 'IGT', 'BL': 'BL', 'ACVD': 'ACVD', 'CKD': 'CKD',
        'COVID-19': 'COVID-19', 'Adenoma': 'adenoma', 'Melanoma': 'melanoma'
    }
    # Load and preprocess feature profile
    print(">>> Loading microbiome profile...")
    X = pd.read_csv(os.path.join(PROFILE_DIR, 'curated_LiS_CRC_20204_v1121.train_test.profile'), sep='\t',
                    index_col=0).transpose()
    # Filter for species-level taxa (s__) and standardize column names
    species_indices = [i for i in range(len(X.columns)) if X.columns[i].split('|')[-1].startswith('s__')]
    X = X.iloc[:, species_indices]
    X.columns = [col.split('|')[-1].replace('__', '-').replace(' ', '-').replace('_', '-').lower() for col in X.columns]
    # Load phenotype metadata
    print(">>> Loading phenotype metadata...")
    label_df = pd.read_csv(os.path.join(PROFILE_DIR, 'curated_LiS_CRC_20204_v0908.train_test.phe'), sep='\t',
                           index_col=0)
    # ==========================================
    # Main Execution Loop
    # ==========================================
    for filename in os.listdir(FEATURE_DIR):
        if not filename.endswith('.txt'):
            continue
        raw_disease = filename.split('.biomarker')[0]
        biomarker_list = []
        with open(os.path.join(FEATURE_DIR, filename), 'r') as f:
            for line in f:
                biomarker_list.append(line.strip().split('\t')[0])
        # Define case and control samples based on the target disease mapping
        if raw_disease in ['pandisease', 'Healthy']:
            ctrl_sample = label_df.loc[label_df['disease'] == 'healthy'].index
            case_sample = label_df.loc[label_df['disease'] != 'healthy'].index
        else:
            disease = DISEASE_MAPPING.get(raw_disease, raw_disease)
            # Find specific projects associated with the targeted disease
            phe_single_project = set(label_df[label_df['disease_united'] == disease]['project'])
            phe_single_df = label_df[label_df['project'].isin(phe_single_project)]
            ctrl_sample = phe_single_df.loc[phe_single_df['disease'] == 'healthy'].index
            case_sample = phe_single_df.loc[phe_single_df['disease'] == disease].index
        print(f"Processing: {raw_disease} | Cases: {len(case_sample)} | Controls: {len(ctrl_sample)}")
        # Execute abundance Wilcoxon test and export results
        case_ctrl_abundance_wilcox(
            X=X,
            case_sample=case_sample,
            ctrl_sample=ctrl_sample,
            raw_disease=raw_disease,
            biomarker_list=biomarker_list,
            save_dir=OUTPUT_DIR
        )