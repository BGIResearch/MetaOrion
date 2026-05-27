# @Date    : 2026/5/26 16:49 
# @Email   : zhangkexin2@genomics.cn
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metagenome_model.inference.pretrainning_evaluation import MetaGenomeSEQInference


def register_params():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--val_data_path', type=str, help='data path')
    parser.add_argument('--model_name_or_path', type=str, help='pretrained config name or path')
    parser.add_argument(
        '--mixed_precision', type=str, default='fp16',
        help="mixed precision type. option: ['fp16', 'fp8', 'bf16', 'no']")
    parser.add_argument(
        '--accumulation_step', type=str, default=1, help='gradient accumulation steps')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--output_home', type=str, help='output home')

    parser.add_argument('--tracker_name', default='wandb', type=str, help='tracker name')
    parser.add_argument('--username', type=str, help='wandb user name')
    parser.add_argument('--projectname', type=str, help='wandb project name')
    parser.add_argument('--group', type=str, help='wandb group')
    parser.add_argument('--job_type', type=str, default='inference', help='wandb job type')

    return parser.parse_args()


def worker():
    args = register_params()

    runner = MetaGenomeSEQInference(**vars(args))
    runner.inference()


if __name__ == '__main__':
    worker()
