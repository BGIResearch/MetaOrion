import os
import json
import pandas as pd
import argparse


def standardize_disease_name(input_disease):
    """
    将输入的疾病字符串转换为模型能识别的标准键值。
    如果不在列表或同义词中，则返回 'others'。
    """
    if pd.isna(input_disease):
        return 'others'

    # 1. 统一去除两端空格并转为小写，提高鲁棒性
    clean_input = str(input_disease).strip().lower()

    # 2. 构建同义词/简写映射字典 (所有 key 必须是小写)
    # 你可以根据实际遇到的数据集在这里不断扩充
    synonym_map = {
        # 健康组
        'healthy': 'healthy', 'control': 'healthy', 'normal': 'healthy', 'hc': 'healthy', 'health': 'healthy',

        # 肠道疾病
        'ibd': 'IBD', 'inflammatory bowel disease': 'IBD', 'cd': 'IBD', 'uc': 'IBD', 'crohn': 'IBD',
        'ibs': 'IBS', 'irritable bowel syndrome': 'IBS',
        'crc': 'CRC', 'colorectal cancer': 'CRC', 'colorectal carcinoma': 'CRC',
        'adenoma': 'adenoma', 'ad': 'adenoma',

        # 代谢类
        't2d': 'T2D', 't2dm': 'T2D', 'type 2 diabetes': 'T2D', 'diabetes': 'T2D',
        'metabolic_syndrome': 'metabolic_syndrome', 'mets': 'metabolic_syndrome',
        'ob': 'OB', 'obesity': 'OB', 'obese': 'OB',
        'igt': 'IGT', 'impaired glucose tolerance': 'IGT',

        # 其他
        'as': 'AS', 'ankylosing spondylitis': 'AS',
        'bl': 'BL',
        'acvd': 'ACVD', 'cvd': 'ACVD', 'atherosclerotic cardiovascular disease': 'ACVD',
        'ckd': 'CKD', 'chronic kidney disease': 'CKD',
        'covid-19': 'COVID-19', 'covid19': 'COVID-19', 'covid': 'COVID-19', 'sars-cov-2': 'COVID-19',
        'melanoma': 'melanoma', 'mm': 'melanoma'
    }

    # 3. 如果在字典里，直接返回标准化的名称
    if clean_input in synonym_map:
        return synonym_map[clean_input]

    # 4. 如果连别名匹配都失败了，统一归为 'others'
    return 'others'

def main():
    # 1. 设置命令行参数解析
    parser = argparse.ArgumentParser(description="covert taxonomy profile and metadata to individual JSON")

    parser.add_argument('--cohort', type=str, required=True,
                        help="cohort name or test dataset name, e.g. FanY_2023")
    parser.add_argument('--profile', type=str, required=True,
                        help="datapath of taxonomy profile (.tsv)")
    # parser.add_argument('--metadata', type=str, required=True,
    #                     help="datapath of metadata (.tsv)")
    parser.add_argument('--out_dir', type=str, required=True,
                        help="dir of output JSON files")

    # 也可以把过滤阈值作为可选参数暴露出来，方便未来调整
    parser.add_argument('--min_abundance', type=float, default=1e-4)
    parser.add_argument('--min_len', type=int, default=7)

    args = parser.parse_args()

    # 2. 读取数据
    print(f"Loading data for cohort: {args.cohort}...")
    data = pd.read_csv(args.profile, sep='\t', index_col=0)
    # phe_df = pd.read_csv(args.metadata, sep='\t', index_col=0)

    # 构建最终的保存路径
    save_dir = os.path.join(args.out_dir, args.cohort)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 3. 核心处理逻辑
    processed_count = 0
    filtered_count = 0

    for i in range(data.shape[1]):
        sample_id = data.columns[i]
        sample_df = data.loc[:, sample_id]

        # 使用命令行传入的阈值
        sample_df = pd.DataFrame(sample_df.loc[(sample_df != 0) & (sample_df >= args.min_abundance)])

        sample_df['seq_part'] = sample_df.index.map(lambda x: x.split('|')[-1])
        mask = sample_df['seq_part'].str.match(r'^[^t]*s__.*$')
        filtered_df = sample_df[mask]
        seq = filtered_df['seq_part'].tolist()

        if len(seq) < args.min_len:
            print(f"Filter length (length < {args.min_len}): {sample_id}")
            filtered_count += 1
            continue
        else:
            abundance = filtered_df[sample_id].tolist()

            sample_data = {}
            sample_data['sampleid'] = sample_id
            sample_data['taxa'] = seq
            sample_data['abundance'] = abundance

            # 获取疾病标签
            # study_condition = phe_df.loc[sample_id, "Group"]
            # sample_data['disease_united'] = standardize_disease_name(study_condition.lower())

            # 保存 JSON
            with open(os.path.join(save_dir, f"{sample_id}.json"), 'w') as f:
                json.dump(sample_data, f, indent=4)

            processed_count += 1

    print(f"Done! Processed: {processed_count} samples, Filtered: {filtered_count} samples.")

    # 4. 保存为datapath文件
    files = os.listdir(save_dir)

    with open(os.path.join(args.out_dir, f'datapath.{args.cohort}'), 'w') as f:
        for data in files:
            if data.endswith('.json'):
                full_path = os.path.abspath(os.path.join(save_dir, data))
            f.write(full_path + '\n')

    print(f"Saved to: {os.path.join(args.out_dir, f'datapath.{args.cohort}')}")


if __name__ == "__main__":
    main()
