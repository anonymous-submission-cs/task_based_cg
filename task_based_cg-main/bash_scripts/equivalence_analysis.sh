#!/bin/bash
python -m scripts.equivalence_representation_analysis \
    --model_split combination_3 \
    --eval_split combination_3 \
    --equivalence_class_analysis_type model \
    --max_task_length 3 \
    --plot_type simple \
    --function_type diverse \
    --pos_embedding_type rel_global 