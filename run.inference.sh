
source /root/miniconda3/etc/profile.d/conda.sh

conda activate MetaOrionEnv



CUDA_VISIBLE_DEVICES=0

accelerate launch \
--config_file \
/bgi-seq-model-2/codes/zhangkexin/MetaOrion/dsub/default_config.yaml \
--machine_rank \
0 \
--main_process_port \
35612 \
--num_machines \
1 \
--num_processes \
1 \
/bgi-seq-model-2/codes/zhangkexin/MetaOrion/scripts/evaluation_phenotype_model.py \
--data_dir \
/bgi-seq-model-2/datasets/zhangkexin/meta_index/preprocess/metaphlan4/fine-tune/nov.specific.random.5 \
--model_name_or_path \
/bgi-seq-model-2/codes/zhangkexin/meta_index/output/llama/v4/PanDisease/Split5.specific.aug.Full.sortabu.ema.12.4 \
--cohort \
pandisease \
--batch_size \
48 \
--mixed_precision \
fp16 \
--accumulation_step \
2 \
--output_home \
/bgi-seq-model-2/codes/zhangkexin/meta_index/output/llama/v4/PanDisease/Split5.specific.aug.Full.sortabu.ema.12.4/ \
--dropout_rate \
0.2 \
--seed \
42 \
