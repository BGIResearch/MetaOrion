import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metagenome_model.train.finetuning_trainer import MetaGenomeForPhenotypeTrainer


def register_params():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--data_dir', type=str, help='train data path')
    parser.add_argument('--model_name_or_path', type=str, help='pretrained config name or path')
    parser.add_argument(
        '--mixed_precision', type=str, default='fp16',
        help="mixed precision type. option: ['fp16', 'fp8', 'bf16', 'no']")
    parser.add_argument(
        '--accumulation_step', type=str, default=1, help='gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--max_epochs', type=int, help='max epoch')
    parser.add_argument('--output_home', type=str, help='output home')
    parser.add_argument('--decay_gamma', type=float, help='multiplicative factor of learning rate decay')
    parser.add_argument('--decay_step', type=int, help='period of learning rate decay')

    return parser.parse_args()


if __name__ == '__main__':
    args = register_params()
    args_dict = vars(args)

    output_home = vars(args)['output_home']

    disease = 'pandisease'
    for s in ['split' + str(i)+'.change' for i in range(1, 6)]:
        args_dict['split'] = s
        args_dict['train_data_path'] = os.path.join(vars(args)['data_dir'], s, f'datapath.{disease}.train.all')
        args_dict['val_data_path'] = os.path.join(vars(args)['data_dir'], s, f'datapath.{disease}.val')

        args_dict['output_home'] = os.path.join(output_home, s)

        runner = MetaGenomeForPhenotypeTrainer(**args_dict)
        runner.train()
        print(f'{s} model is finished training!')
