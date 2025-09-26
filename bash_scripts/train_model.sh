#!/bin/bash
PROMPT_LENGTHS=("fixed")
PROMPT_MODES=("direct" "step_by_step")
POS_EMBEDDING_TYPES=("abs" "rel_global")
FUNCTION_TYPES=("uniform" "diverse")


EPOCHS=100
N_ALPHABETS=26
SEQ_LEN=6
N_FUNCTIONS=6
NHEADS_NLAYERS="nh6_nl3"
SEEDS=(0 10 20 30 40)

TRAIN_SPLIT_STRATEGIES=("combination_2" "combination_3" "combination_4" "combination_5" "combination_6")

# task max length is k_max and gets k from the split strategy without identity modules. Fix task_max_length to 7 for identity-based train/test split
echo "=== TRAINING ==="
for split in "${TRAIN_SPLIT_STRATEGIES[@]}"; do
    for length in "${PROMPT_LENGTHS[@]}"; do
        for mode in "${PROMPT_MODES[@]}"; do
            for function_type in "${FUNCTION_TYPES[@]}"; do
                for pos_embedding_type in "${POS_EMBEDDING_TYPES[@]}"; do
                    for seed in "${SEEDS[@]}"; do
                        echo "Training: $mode - $length - $split"
                        # get task max length from split
                        TASK_MAX_LENGTH=$(echo "$split" | cut -d'_' -f2)
                        echo "Task max length: $TASK_MAX_LENGTH"
                        python -m scripts.train_model \
                            --prompt_mode "$mode" \
                            --prompt_length "$length" \
                            --train_split "$split" \
                            --epochs "$EPOCHS" \
                            --n_alphabets "$N_ALPHABETS" \
                            --seq_len "$SEQ_LEN" \
                            --n_functions "$N_FUNCTIONS" \
                            --pos_embedding_type "$pos_embedding_type" \
                            --n_heads_nlayers "$NHEADS_NLAYERS" \
                            --function_type "$function_type" \
                            --task_max_length "$TASK_MAX_LENGTH" \ 
                            --seed "$seed"
                    done
                done
            done
        done
    done
done



