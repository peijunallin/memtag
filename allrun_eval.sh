#!/bin/bash
export HF_HOME=[REDACTED]
export HF_DATASETS_CACHE=[REDACTED]
export TRANSFORMERS_CACHE=[REDACTED]
huggingface-cli login --token [REDACTED]



#bt_300each_router1_results_topk50
#root_weights_base=[REDACTED]
#root_results_base=[REDACTED]
#root_results_base=[REDACTED]

#root_results_base=[REDACTED]
root_results_base=[REDACTED]
for i in {1..10}; do
    results_path="${root_results_base}${i}_results_topk50"
    output_dir="${results_path}/1keach_router${i}.csv"

    if [ -d "$results_path" ]; then
        echo "Processing $results_path"
        python eval.py --results_dir "$results_path" --output_file "$output_dir" & 
    else
        echo "Directory $results_path does not exist. Skipping."
    fi
done

echo "All runs completed."
