#!/bin/bash
export HF_HOME=[REDACTED]
export HF_DATASETS_CACHE=[REDACTED]
export TRANSFORMERS_CACHE=[REDACTED]
huggingface-cli login --token [REDACTED]


# set manual_ratio == true
# Pick the best memory type
# Pick the best memory type
# Pick the best memory type
# Pick the best memory type
# Pick the best memory type
# Pick the best memory type



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
gpu_id="4"

[REDACTED]
[REDACTED]



root_weights_base=[REDACTED]

for i in {4..4}; do
[REDACTED]
    log_dir="./logs/ablation1_router4_pickbest_topk50_run0/"
    output_dir="./results/ablation1_router4_pickbest_topk50_run0"

    # Wait until the weights file exists
    echo "Waiting for weights file: $weights_path"
    while [ ! -f "$weights_path" ]; do
        sleep 10
    done
    echo "Found weights: $weights_path"

    mkdir -p "$log_dir"
    mkdir -p "$output_dir"

    for dataset in "${datasets[@]}"; do
[REDACTED]

        CUDA_VISIBLE_DEVICES=$gpu_id python run_readers_hm_router_final0.py \
            --dataset "$dataset" \
            --weights_path "$weights_path" \
            --ablation_type "$ablation_type" \
            --manual_ratio "best1" \
            --memory_repr_weights "0.25,0.25,0.25,0.25" \
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
            --num_noise_docs $num_noise_docs > "${log_dir}/hm_logs_${dataset}_${reader_model}.log" 2>&1

[REDACTED]
    done
done

echo "All runs completed."