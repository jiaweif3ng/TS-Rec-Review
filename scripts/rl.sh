#!/bin/bash

export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE
export PYTHONPATH="$(pwd):${PYTHONPATH}"

DATASET="${DATASET:-Toys_and_Games}"

MODEL="Qwen2.5-1.5B"
MODEL_PATH="${MODEL_PATH:-outputs/Qwen2.5-1.5B_ts_rec_sft_Toys_and_Games}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${MODEL}_ts_rec_rl_${DATASET}}"

for category in $DATASET; do
    train_file=$(ls -f ./data/Amazon/train/${category}*.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)

    HF_ENDPOINT=https://hf-mirror.com accelerate launch \
                                    --config_file ./config/zero2_opt.yaml \
                                    --num_processes 8 --main_process_port 29501 \
                                    train/rl.py \
                        --model_path "${MODEL_PATH}" \
                        --train_batch_size 64 \
                        --eval_batch_size 128 \
                        --num_train_epochs 2 \
                        --gradient_accumulation_steps 2 \
                        --train_file ${train_file} \
                        --eval_file ${eval_file} \
                        --info_file ${info_file} \
                        --category ${category} \
                        --sample_train False \
                        --eval_step 0.0999 \
                        --reward_type ranking \
                        --num_generations 16 \
                        --mask_all_zero False \
                        --dynamic_sampling False \
                        --sync_ref_model True \
                        --beam_search True \
                        --test_during_training False \
                        --temperature 1.0 \
                        --learning_rate 1e-5 \
                        --add_gt False \
                        --beta 1e-3 \
                        --dapo False \
                        --output_dir "${OUTPUT_DIR}" \
                        --wandb_run_name MiniOneRec_SFT+GRPO \
                        --sid_index_path ./data/Amazon/index/${category}.index.json \
                        --item_meta_path ./data/Amazon/index/${category}.item.json
done
