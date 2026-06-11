# @Date    : 2026/6/9 10:25
# @Email   : zhangkexin2@genomics.cn
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metaorion.inference.IG_feature_attribute import MetaOrionPhenotypeAttribute


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
        '--accumulation_step', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--output_home', type=str, help='output home')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--attribution_steps', type=int, default=100, help='integrated gradients steps')

    return parser.parse_args()


def worker():
    args = register_params()
    args_dict = vars(args)

    output_home = vars(args)['output_home']
    model_name_or_path = vars(args)['model_name_or_path']
    cohort = vars(args)['cohort']

    for s in ['split' + str(i) for i in range(1, 6)]:
        print(s)

        args_dict['val_data_path'] = os.path.join(args_dict['data_dir'], f'{s}.change/datapath.{cohort}.test')
        args_dict['model_name_or_path'] = os.path.join(model_name_or_path, s, 'best_ckpt/')
        args_dict['output_home'] = os.path.join(output_home, s, 'best_ckpt/result/')

        runner = MetaOrionPhenotypeAttribute(**args_dict)
        runner.inference()


if __name__ == '__main__':
    worker()
