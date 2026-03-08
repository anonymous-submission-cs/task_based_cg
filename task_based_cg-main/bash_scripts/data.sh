#!/bin/bash
# Generate data for within-k and cross-k evaluation with identity functions as task max length=7
for prompt_length in "fixed"; do
    echo "Generating data for prompt_length: $prompt_length"
    for k in {1..6}; do
        echo "Generating data for combination_$k"
        python -m scripts.generate_data --prompt_length $prompt_length --split_strategy combination_$k --n_functions 6 --task_max_length 7 --n_alphabets 26 --seq_len 6 --functions_type uniform
        python -m scripts.generate_data --prompt_length $prompt_length --split_strategy combination_$k --n_functions 6 --task_max_length 7 --n_alphabets 26 --seq_len 6 --functions_type diverse
        
    done
done

# Generate data for within-k evaluation without identity functions (as task max length=k)
for prompt_length in "fixed"; do
    echo "Generating data for prompt_length: $prompt_length"
    for k in {2..6}; do
        echo "Generating data for combination_$k"
        python -m scripts.generate_data --prompt_length $prompt_length --split_strategy combination_$k --n_functions 6 --task_max_length $k --n_alphabets 26 --seq_len 6 --functions_type uniform
        python -m scripts.generate_data --prompt_length $prompt_length --split_strategy combination_$k --n_functions 6 --task_max_length $k --n_alphabets 26 --seq_len 6 --functions_type diverse
    done
done

# Generate data for random sampling of compositions for K=6 for various train-test split ratios
train_perm_size=(1 10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700)
for size in "${train_perm_size[@]}"; do
    python -m scripts.generate_data --prompt_length fixed --split_strategy permutationrandom_6_$size --n_functions 6 --task_max_length 6 --n_alphabets 26 --seq_len 6 --functions_type uniform 
    python -m scripts.generate_data --prompt_length fixed --split_strategy permutationrandom_6_$size --n_functions 6 --task_max_length 6 --n_alphabets 26 --seq_len 6 --functions_type diverse 
done

# generate data for systematic sampling of compositions for K=6 for various train-test split ratios
train_perm_size=(1 10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700)
for size in "${train_perm_size[@]}"; do
    python -m scripts.generate_data --prompt_length fixed --split_strategy permutation_6_$size --n_functions 6 --task_max_length 6 --n_alphabets 26 --seq_len 6 --functions_type uniform 
    python -m scripts.generate_data --prompt_length fixed --split_strategy permutation_6_$size --n_functions 6 --task_max_length 6 --n_alphabets 26 --seq_len 6 --functions_type diverse 
done

# generate data to validate composition equivalence necessity using uniform-identity based equivalences
# We focus on K=6, keep train perm size fixed as 576, test perm size fixed as 144 and systematically increase number of shared compositions in train and test
shared_equivalence_size=(0 1 25 49 73 97 121 144)
for size in "${shared_equivalence_size[@]}"; do
    python -m scripts.generate_data --prompt_length fixed --split_strategy equivalence_6_576_$size --n_functions 6 --task_max_length 7 --n_alphabets 26 --seq_len 6 --functions_type uniform 
done

# generate data to validate number of training equivalences needed to learn corresponding test equivalences
# We focus on K=6, keep train perm size fixed as 576, and shared equivalences as 100% and systematically increase number of training equivalences
training_equivalence_size=(0)
for size in "${training_equivalence_size[@]}"; do
    python -m scripts.generate_data --prompt_length fixed --split_strategy uniequivalence_6_576_$size --n_functions 6 --task_max_length 7 --n_alphabets 26 --seq_len 6 --functions_type uniform 
done

