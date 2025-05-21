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



# Run musique on GPU 2
[REDACTED]
    --dataset 'musique' \
    --memory_repr 'triples' \
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
    --num_noise_docs $num_noise_docs &

# Run 2wikimultihopqa on GPU 3
[REDACTED]
    --dataset '2wikimultihopqa' \
    --memory_repr 'triples' \
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
    --num_noise_docs $num_noise_docs &

[REDACTED]
[REDACTED]
[REDACTED]
    --memory_repr 'triples' \
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
    --num_noise_docs $num_noise_docs &

[REDACTED]
[REDACTED]
[REDACTED]
    --memory_repr 'triples' \
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
    --num_noise_docs $num_noise_docs &

# Run quality on GPU 6
[REDACTED]
    --dataset 'quality' \
    --memory_repr 'triples' \
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
    --num_noise_docs $num_noise_docs &

