#!/bin/bash
export HF_HOME=[REDACTED]
export HF_DATASETS_CACHE=[REDACTED]
export TRANSFORMERS_CACHE=[REDACTED]
huggingface-cli login --token [REDACTED]

ablation_type="hm_memory_units_with_top_k"
top_k=50
top_r=0
top_s=0
top_t=25
num_turns=4
answer_with_memory_units=1
use_iterative_retrieval=0
maximum_tokens=32768
debug=0
reader_model="qwen2.5_7b_instruct"
num_noise_docs=0

gpu_id="7"
normalize_method='softmax'
norm_params=3
manual_ratio="router"
remove="chunks"
#manual_ratio="best1" or "manual" "router" 
#{'triples': -4.34375, 'atomic facts': -1.21875, 'summary': -4.28125, 'chunks': -3.40625}
datasets=( "hotpotqa" "musique" "2wikimultihopqa" )
[REDACTED]



root_weights_base=[REDACTED]

for i in {31..31}; do
[REDACTED]
    log_dir="./removeone_logs/bt_lora_800each_topk50_remove_chunks/bt_800each_router${i}_logs"
    output_dir="./removeone_results/bt_lora_800each_router${i}_topk50_remove_chunks"

    # Wait until the weights file exists
    # echo "Waiting for weights file: $weights_path"
    # while [ ! -f "$weights_path" ]; do
    #     sleep 10
    # done
    echo "Found weights: $weights_path"

    mkdir -p "$log_dir"
    mkdir -p "$output_dir"

    for dataset in "${datasets[@]}"; do
[REDACTED]

        CUDA_VISIBLE_DEVICES=$gpu_id python run_readers_hm_router_final0_removeone.py \
            --dataset "$dataset" \
            --remove "$remove" \
            --normalize_method "$normalize_method" \
            --weights_path "$weights_path" \
            --ablation_type "$ablation_type" \
            --memory_repr_weights "1,0,0,0" \
            --manual_ratio "$manual_ratio" \
            --out_dir "$output_dir" \
            --answer_with_memory_units $answer_with_memory_units \
            --use_iterative_retrieval $use_iterative_retrieval \
            --top_k $top_k \
            --top_r $top_r \
            --top_s $top_s \
            --top_t $top_t \
            --num_turns $num_turns \
            --debug $debug \
            --maximum_tokens $maximum_tokens \
            --reader_model "$reader_model" \
[REDACTED]

[REDACTED]
    done
done

echo "All runs completed."