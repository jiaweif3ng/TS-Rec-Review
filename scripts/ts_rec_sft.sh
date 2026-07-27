#!/bin/bash
export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE
export PYTHONPATH="$(pwd):${PYTHONPATH}"

DATASET="Industrial_and_Scientific"
# DATASET="Toys_and_Games"

MODEL="Qwen2.5-1.5B"

# TS-Rec components (set to True/False)
USE_SA_INIT=False
USE_TS_ALIGN=False

for category in $DATASET; do

    train_file=$(ls -f ./data/Amazon/train/${category}*11.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    test_file=$(ls -f ./data/Amazon/test/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)

    echo ${train_file} ${eval_file} ${info_file} ${test_file}

    torchrun --nproc_per_node 8 \
            train/ts_rec_sft.py \
            --base_model ${MODEL} \
            --batch_size 1024 \
            --micro_batch_size 16 \
            --train_file ${train_file} \
            --eval_file ${eval_file} \
            --output_dir ${MODEL}_ts_rec_sft_${category} \
            --wandb_project MiniOneRec_SFT \
            --wandb_run_name ${MODEL}_ts_rec_sft_${category} \
            --category ${category} \
            --train_from_scratch False \
            --seed 42 \
            --sid_index_path ./data/Amazon/index/${category}.index.json \
            --item_meta_path ./data/Amazon/index/${category}.item.json \
            --freeze_LLM False \
            --description_path ./data/Amazon/semantics/${category}.description_keywords.json \
            --use_sa_init ${USE_SA_INIT} \
            --use_ts_align ${USE_TS_ALIGN} \
            --learning_rate 3e-4
done