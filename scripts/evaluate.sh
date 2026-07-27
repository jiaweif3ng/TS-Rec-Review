#!/bin/bash
set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

DATASET="${DATASET:-Industrial_and_Scientific}"
MODEL_PATH="${MODEL_PATH:-outputs/Qwen2.5-1.5B_ts_rec_sft_${DATASET}}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"

test_file=$(ls ./data/Amazon/test/${DATASET}*11.csv | head -1)
info_file=$(ls ./data/Amazon/info/${DATASET}*.txt | head -1)
model_name=$(basename "$MODEL_PATH")
temp_dir="./temp/${DATASET}-${model_name}"
output_dir="./results/${model_name}"

mkdir -p "$temp_dir" "$output_dir"

python utils/split.py \
    --input_path "$test_file" \
    --output_path "$temp_dir" \
    --cuda_list "$GPU_LIST"

IFS=',' read -ra gpu_ids <<< "$GPU_LIST"
for gpu_id in "${gpu_ids[@]}"; do
    CUDA_VISIBLE_DEVICES="$gpu_id" python -u eval/evaluate.py \
        --base_model "$MODEL_PATH" \
        --info_file "$info_file" \
        --category "$DATASET" \
        --test_data_path "$temp_dir/${gpu_id}.csv" \
        --result_json_data "$temp_dir/${gpu_id}.json" \
        --batch_size 8 \
        --num_beams 20 \
        --max_new_tokens 256 \
        --temperature 1.0 \
        --guidance_scale 1.0 \
        --length_penalty 0.0 &
done
wait

python utils/merge.py \
    --input_path "$temp_dir" \
    --output_path "$output_dir/final_result_${DATASET}.json" \
    --cuda_list "$GPU_LIST"

python utils/calc.py \
    --path "$output_dir/final_result_${DATASET}.json" \
    --item_path "$info_file"
