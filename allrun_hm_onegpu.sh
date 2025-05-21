#!/bin/bash
export HF_HOME=[REDACTED]
export HF_DATASETS_CACHE=[REDACTED]
export TRANSFORMERS_CACHE=[REDACTED]
huggingface-cli login --token [REDACTED]

ablation_type="hm_memory_units_with_top_k"
top_k=100
top_r=0
top_s=0
top_t=50
num_turns=4
answer_with_memory_units=1
use_iterative_retrieval=1
maximum_tokens=4096
debug=0
reader_model="qwen2.5_7b_instruct"
num_noise_docs=0
# Define memory representations to iterate through
#memory_representations=("triples" "summary" "chunks" "atomic_facts" "mix")

[REDACTED]
#datasets=("hotpotqa")
# Specify which GPU to use
gpu_id="0,1"  # Change this to your desired GPU ID

# Iterate through all memory representations
# for memory_repr in "${memory_representations[@]}"; do
#     echo "Running with memory representation: $memory_repr"
    
# Run each dataset sequentially on the same GPU
for dataset in "${datasets[@]}"; do
    echo "Starting $dataset on GPU $gpu_id with hm memory representation "
    
    CUDA_VISIBLE_DEVICES=$gpu_id python run_readers_hm_router_v0.py \
        --dataset "$dataset" \
        --ablation_type "$ablation_type" \
        --memory_repr_weights "0.25,0.25,0.25,0.25" \
        --out_dir "./hm_router0_results" \
        --answer_with_memory_units $answer_with_memory_units \
        --use_iterative_retrieval $use_iterative_retrieval \
        --top_k $top_k \
        --top_r $top_r \
        --top_s $top_s \
        --top_t $top_t \
        --num_turns $num_turns \
        --debug $debug \
        --maximum_tokens $maximum_tokens         --reader_model "$reader_model" \
        --num_noise_docs $num_noise_docs > "./logs/hm_router_logs/hm_logs_${dataset}_${memory_repr}_${reader_model}.log" 2>&1
    
    echo "Completed $dataset with memory representation"
done
    

echo "All runs completed."