#!/bin/bash

export HF_HOME=[REDACTED]
export HF_DATASETS_CACHE=[REDACTED]
export TRANSFORMERS_CACHE=[REDACTED]
huggingface-cli login --token [REDACTED]
[REDACTED]
ablation_type="memory_units_with_top_k"
top_k=100
top_r=0
top_s=0
top_t=50
num_turns=4
answer_with_memory_units=1
use_iterative_retrieval=1
maximum_tokens=4096
debug=0
reader_model="llama3.1_8b_instruct"
num_noise_docs=0

# Define memory representations to iterate through
#memory_representations=("chunks" "atomic_facts" "summary" "mix")
memory_representations=("mix2" "raw_documents")
[REDACTED]
gpu_assignments=("2" "3" "4" "5" "6" "7")

# Iterate through all memory representations
for memory_repr in "${memory_representations[@]}"; do
    echo "Running with memory representation: $memory_repr"
    
    # Launch all datasets simultaneously on their assigned GPUs
    for i in "${!datasets[@]}"; do
        dataset=${datasets[$i]}
        gpu=${gpu_assignments[$i]}
        
        echo "Starting $dataset on GPU $gpu with memory representation $memory_repr"
        
[REDACTED]
            --dataset "$dataset" \
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
            --num_noise_docs $num_noise_docs > "logs_${dataset}_${memory_repr}.log" 2>&1 &
        
        # Store the PID of the background process
        pids[$i]=$!
    done
    
    # Wait for all jobs to complete before starting the next memory representation
    for pid in ${pids[*]}; do
        wait $pid
    done
    
    echo "Completed all datasets for memory representation: $memory_repr"
done

echo "All runs completed."