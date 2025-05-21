#!/bin/bash

export HF_HOME=[REDACTED]
export HF_DATASETS_CACHE=[REDACTED]
export TRANSFORMERS_CACHE=[REDACTED]

huggingface-cli login --token [REDACTED]

# Directory containing all result folders
#root_results_base=[REDACTED]
root_results_base=[REDACTED]
# Loop through each folder inside the root directory
for results_path in "$root_results_base"/*; do
    if [ -d "$results_path" ]; then
        folder_name=$(basename "$results_path")
        output_file="${results_path}/${folder_name}.csv"
        echo "Processing $results_path"
        python eval.py --results_dir "$results_path" --output_file "$output_file" &
    else
        echo "Skipping non-directory $results_path"
    fi
done

wait
echo "All runs completed."
