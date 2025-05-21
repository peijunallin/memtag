#!/bin/bash
export HF_HOME=[REDACTED]
export HF_DATASETS_CACHE=[REDACTED]
export TRANSFORMERS_CACHE=[REDACTED]
huggingface-cli login --token [REDACTED]
[REDACTED]
ablation_type="memory_units_with_top_k"
top_k=25
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
# Define memory representations to iterate through

memory_representations=("triples" "summary" "chunks" "atomic_facts")
datasets=("hotpotqa" "musique" "2wikimultihopqa" )

#memory_representations=("mix2")
[REDACTED]
#datasets=("hotpotqa")

# Specify which GPU to use
gpu_id="6"  # Change this to your desired GPU ID
log_dir="./logs/baseline_logs_topk25"
output_dir="./baseline_results_topk25"

mkdir -p "$output_dir"
mkdir -p "$log_dir"
# Iterate through all memory representations
for memory_repr in "${memory_representations[@]}"; do
    echo "Running with memory representation: $memory_repr"
    
    # Run each dataset sequentially on the same GPU
    for dataset in "${datasets[@]}"; do
        echo "Starting $dataset on GPU $gpu_id with memory representation $memory_repr"
        
[REDACTED]
            --dataset "$dataset" \
            --out_dir "$output_dir" \
            --memory_repr "$memory_repr" \
            --ablation_type "$ablation_type" \
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
            --num_noise_docs $num_noise_docs > "${log_dir}/logs_${dataset}_${memory_repr}_${reader_model}.log" 2>&1
        
        echo "Completed $dataset with memory representation $memory_repr"
    done
    
    echo "Completed all datasets for memory representation: $memory_repr"
done

echo "All runs completed."