import argparse
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metaorion.inference.phenotype_inference import MetaOrionPhenotypeInfer


def register_params():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--data_dir', type=str, help='data path')
    parser.add_argument('--cohort', type=str, help='cohort name')
    parser.add_argument('--model_name_or_path', type=str, help='pretrained config name or path')
    parser.add_argument(
        '--mixed_precision', type=str, default='fp16',
        help="mixed precision type. option: ['fp16', 'fp8', 'bf16', 'no']")
    parser.add_argument(
        '--accumulation_step', type=str, default=1, help='gradient accumulation steps')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--output_home', type=str, help='output home')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='dropout rate')

    return parser.parse_args()


def worker():
    args = register_params()
    args_dict = vars(args)

    output_home = vars(args)['output_home']
    model_name_or_path = vars(args)['model_name_or_path']
    cohort = vars(args)['cohort']

    result_df = {'precision': [], 'recall': [], 'F1': [], 'acc': [], 'AUC': [], 'AUPR': [], 'MCC': []}
    multi_result_df = {'precision': [], 'recall': [], 'F1': [], 'acc': [], 'MCC': []}
    metric_splits = []
    for s in ['split' + str(i) for i in range(1, 6)]:
        print(s)

        args_dict['val_data_path'] = os.path.join(args_dict['data_dir'], f'{s}.change/datapath.{cohort}.test')
        args_dict['model_name_or_path'] = os.path.join(model_name_or_path, s, 'best_ckpt/')
        args_dict['output_home'] = os.path.join(output_home, s, 'best_ckpt/result/')

        runner = MetaOrionPhenotypeInfer(**args_dict)
        pan_metrics, multi_metrics = runner.inference()

        if pan_metrics is None or multi_metrics is None:
            continue

        metric_splits.append(s)
        for i in range(len(pan_metrics)):
            result_df[list(result_df.keys())[i]].append(pan_metrics[i])
        for i in range(len(multi_metrics)):
            multi_result_df[list(multi_result_df.keys())[i]].append(multi_metrics[i])

    if not metric_splits:
        print('No labels found in any split. Only probability files were saved.')
        return

    result_df = pd.DataFrame(result_df, index=metric_splits)
    mean_row = result_df.mean(numeric_only=True)
    result_df.loc['mean'] = mean_row
    print(result_df)
    multi_result_df = pd.DataFrame(multi_result_df, index=metric_splits)
    mean_row = multi_result_df.mean(numeric_only=True)
    multi_result_df.loc['mean'] = mean_row
    print(multi_result_df)

    # result_df.round(4).to_csv(
    #     os.path.join(output_home, f'{cohort}.pandisease.test.result.csv'))
    # multi_result_df.round(4).to_csv(os.path.join(output_home, f'{cohort}.multidisease.test.result.csv'))



if __name__ == '__main__':
    worker()
