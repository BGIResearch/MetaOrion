import os
import json
import pandas as pd
import argparse


def standardize_disease_name(input_disease):
    """
    Converts the input disease string into a standard key recognized by the model.
    Returns 'others' if the input is not found in the predefined synonym map.
    """
    if pd.isna(input_disease):
        return 'others'

    # 1. Strip whitespaces and convert to lowercase for robustness
    clean_input = str(input_disease).strip().lower()

    # 2. Synonym mapping dictionary (all keys must be lowercase)
    # You can continuously expand this mapping based on new datasets
    synonym_map = {
        # Healthy group
        'healthy': 'healthy', 'control': 'healthy', 'normal': 'healthy', 'hc': 'healthy', 'health': 'healthy',

        # Bowel diseases
        'ibd': 'IBD', 'inflammatory bowel disease': 'IBD', 'cd': 'IBD', 'uc': 'IBD', 'crohn': 'IBD',
        'ibs': 'IBS', 'irritable bowel syndrome': 'IBS',
        'crc': 'CRC', 'colorectal cancer': 'CRC', 'colorectal carcinoma': 'CRC',
        'adenoma': 'adenoma', 'ad': 'adenoma',

        # Metabolic diseases
        't2d': 'T2D', 't2dm': 'T2D', 'type 2 diabetes': 'T2D', 'diabetes': 'T2D',
        'metabolic_syndrome': 'metabolic_syndrome', 'mets': 'metabolic_syndrome',
        'ob': 'OB', 'obesity': 'OB', 'obese': 'OB',
        'igt': 'IGT', 'impaired glucose tolerance': 'IGT',

        # Others
        'as': 'AS', 'ankylosing spondylitis': 'AS',
        'bl': 'BL',
        'acvd': 'ACVD', 'cvd': 'ACVD', 'atherosclerotic cardiovascular disease': 'ACVD',
        'ckd': 'CKD', 'chronic kidney disease': 'CKD',
        'covid-19': 'COVID-19', 'covid19': 'COVID-19', 'covid': 'COVID-19', 'sars-cov-2': 'COVID-19',
        'melanoma': 'melanoma', 'mm': 'melanoma'
    }

    # 3. Return the standardized name if matched, else fallback to 'others'
    return synonym_map.get(clean_input, 'others')


def main():
    # 1. Setup command-line argument parsing
    parser = argparse.ArgumentParser(description="Convert taxonomy profile and metadata to individual JSON files.")

    parser.add_argument('--cohort', type=str, required=True,
                        help="Cohort name or test dataset name, e.g., FanY_2023")
    parser.add_argument('--profile', type=str, required=True,
                        help="Datapath of taxonomy profile (.tsv)")
    # Make metadata optional
    parser.add_argument('--metadata', type=str, required=False, default=None,
                        help="Datapath of metadata (.tsv). Optional.")
    parser.add_argument('--out_dir', type=str, required=True,
                        help="Directory to output JSON files")

    # Optional parameters for filtering thresholds
    parser.add_argument('--min_abundance', type=float, default=1e-4,
                        help="Minimum abundance threshold for filtering")
    parser.add_argument('--min_len', type=int, default=7,
                        help="Minimum taxa length threshold")

    args = parser.parse_args()

    # 2. Load data
    print(f"Loading data for cohort: {args.cohort}...")
    data = pd.read_csv(args.profile, sep='\t', index_col=0)

    # Load metadata only if provided
    phe_df = None
    if args.metadata and os.path.exists(args.metadata):
        phe_df = pd.read_csv(args.metadata, sep='\t', index_col=0)
        print("Metadata loaded successfully.")

    # Construct and create the final save directory
    save_dir = os.path.join(args.out_dir, args.cohort)
    os.makedirs(save_dir, exist_ok=True)

    # 3. Core processing logic
    processed_count = 0
    filtered_count = 0

    for sample_id in data.columns:
        # Extract as a Series instead of a DataFrame for better performance
        sample_series = data[sample_id]

        # Apply minimum abundance filter (no need to check != 0 if min_abundance > 0)
        sample_series = sample_series[sample_series >= args.min_abundance]

        # Filter taxa to strictly species level
        seq_parts = sample_series.index.map(lambda x: x.split('|')[-1])
        mask = seq_parts.str.match(r'^[^t]*s__.*$')

        filtered_series = sample_series[mask]
        seq = seq_parts[mask].tolist()

        if len(seq) < args.min_len:
            print(f"Filtered out (length < {args.min_len}): {sample_id}")
            filtered_count += 1
            continue

        abundance = filtered_series.tolist()

        # Initialize JSON dictionary
        sample_data = {
            'sampleid': sample_id,
            'taxa': seq,
            'abundance': abundance
        }

        # Add label field ONLY if metadata was successfully loaded
        if phe_df is not None:
            # Safely check if sample_id exists in metadata and Group column exists
            if sample_id in phe_df.index and "Group" in phe_df.columns:
                study_condition = phe_df.loc[sample_id, "Group"]
                sample_data['label'] = standardize_disease_name(study_condition)
            else:
                # Fallback if sample is missing from metadata but metadata was provided
                sample_data['label'] = 'others'

                # Save to JSON
        json_path = os.path.join(save_dir, f"{sample_id}.json")
        with open(json_path, 'w') as f:
            json.dump(sample_data, f, indent=4)

        processed_count += 1

    print(f"Done! Processed: {processed_count} samples, Filtered: {filtered_count} samples.")

    # 4. Save the absolute paths to a datapath file
    datapath_file = os.path.join(args.out_dir, f'datapath.{args.cohort}')
    with open(datapath_file, 'w') as f:
        for file_name in os.listdir(save_dir):
            if file_name.endswith('.json'):
                full_path = os.path.abspath(os.path.join(save_dir, file_name))
                f.write(full_path + '\n')

    print(f"Saved datapath to: {datapath_file}")


if __name__ == "__main__":
    main()

# python data_preprocess.py     --cohort "13_Ning_2023"     --profile "../demo_data/13_Ning_2023.profile"     --metadata "../demo_data/13_Ning_2023.info"     --out_dir "../demo_data/datapaths/"
