#!/bin/bash
N_ALPHABETS=26
SEQ_LEN=6
N_FUNCTIONS=6
NHEADS_NLAYERS="nh6_nl3"

# set task max length to 7 for identity-based train/test split
# set task max length to k in split strategy for without identity based train/test split
python -m scripts.evaluate_model \
    --prompt_mode "direct" \
    --prompt_length "fixed" \
    --model_split "combination_3" \
    --eval_split "combination_6" \
    --nheads_nlayers "$NHEADS_NLAYERS" \
    --n_alphabets "$N_ALPHABETS" \
    --seq_len "$SEQ_LEN" \
    --n_functions "$N_FUNCTIONS" \
    --pos_embedding_type "rel_global" \
    --function_type "uniform" \
    --max_task_length 7 \
    --seed 0 \
    

# Command for representation analysis and equivalence class analysis to generate TSNE plots
# keep strings representation analysis to True 
# replace_strings_repr: True if you want to replace strings in representation analysis to a fixed string for ease of visualization of equivalence classes
# replace_strings_repr : False if you want to keep the original train/test strings in representation analysis
# keep equivalence_class_analysis to True if you want to compute equivalence class based on data and model
# equivalence_class_type: both, data, model
python -m scripts.evaluate_model \
    --prompt_mode "direct" \
    --prompt_length "fixed" \
    --model_split "combination_3" \
    --eval_split "combination_6" \
    --nheads_nlayers "$NHEADS_NLAYERS" \
    --n_alphabets "$N_ALPHABETS" \
    --seq_len "$SEQ_LEN" \
    --n_functions "$N_FUNCTIONS" \
    --pos_embedding_type "rel_global" \
    --function_type "uniform" \
    --max_task_length 7 \
    --seed 0 \
    --repr_analysis True \
    --replace_strings_repr "True" \
    --equivalence_class_analysis True \
    --equivalence_class_type "both"

