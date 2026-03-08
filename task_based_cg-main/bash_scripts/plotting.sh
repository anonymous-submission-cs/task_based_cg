#!/bin/bash

# Script to generate evaluation plots for different configurations
echo "=== GENERATING EVALUATION PLOTS ==="



# within-K plots including identity modules. 
# Set task max length to variable and get k from split strategy for without identity based train/test split 
# self_eval flag indicates evaluation on same-k train and test splits
# Figure 2 in the paper
strategy_prefix="combination"
function_types=(uniform diverse)
for function_type in "${function_types[@]}"; do
    RESULTS_DIR="./results/plot_test/${function_type}/${strategy_prefix}_identity"
    mkdir -p "$RESULTS_DIR"
    python -m scripts.analysis_plots \
            --prompt_length fixed \
            --train_splits ${strategy_prefix}_2 ${strategy_prefix}_3 ${strategy_prefix}_4 ${strategy_prefix}_5 ${strategy_prefix}_6 \
            --self_eval \
            --results_dir "$RESULTS_DIR" \
            --modes  step_by_step direct \
            --pos_types  abs rel_global \
            --function_type $function_type \
            --task_max_length_flag fixed \
            --task_max_length 7 \
            --plot_std \
            --seeds 0 10 20 30 40 
done

strategy_prefix="combination"
function_types=(uniform diverse)
for function_type in "${function_types[@]}"; do
    RESULTS_DIR="./results/plot_test/${function_type}/${strategy_prefix}_without_identity"
    mkdir -p "$RESULTS_DIR"
    python -m scripts.analysis_plots \
            --prompt_length fixed \
            --train_splits ${strategy_prefix}_2 ${strategy_prefix}_3 ${strategy_prefix}_4 ${strategy_prefix}_5 ${strategy_prefix}_6 \
            --self_eval \
            --results_dir "$RESULTS_DIR" \
            --modes  step_by_step direct \
            --pos_types  abs rel_global \
            --function_type $function_type \
            --task_max_length_flag variable \
            --plot_std \
            --seeds 0 10 20 30 40 
done


# cross-K evaluation plots for fixed prompt length
# Figure 3 in the paper
strategy_prefix="combination"
function_types=(uniform diverse)
for function_type in "${function_types[@]}"; do
    combo=(3 6)
    for i in ${combo[@]}; do
            python -m scripts.analysis_plots \
            --prompt_length fixed \
            --train_splits ${strategy_prefix}_${i} \
            --eval_splits ${strategy_prefix}_1 ${strategy_prefix}_2 ${strategy_prefix}_3 ${strategy_prefix}_4 ${strategy_prefix}_5 ${strategy_prefix}_6 \
            --results_dir "$RESULTS_DIR" \
            --modes  direct step_by_step \
            --pos_types  abs rel_global \
            --plot_std \
            --function_type $function_type \
            --task_max_length_flag fixed \
            --task_max_length 7 \
            --seeds 0 10 20 30 40 
    done
done

# Figure 4(a) in the paper
strategy_prefix="permutation" # systematic sampling of compositions
function_type="diverse"
# set strategy prefix to permutationrandom for random sampling of compositions
# use --plot_direct_sbs to plot final vs. full output accuracy only for step-by-step models.
RESULTS_DIR="./results/plot_test/${function_type}/${strategy_prefix}"
mkdir -p "$RESULTS_DIR"

python -m scripts.analysis_plots \
        --prompt_length fixed \
        --train_splits ${strategy_prefix}_6_1 ${strategy_prefix}_6_10 ${strategy_prefix}_6_20 ${strategy_prefix}_6_30 ${strategy_prefix}_6_40 ${strategy_prefix}_6_50 ${strategy_prefix}_6_60 ${strategy_prefix}_6_70 ${strategy_prefix}_6_80 ${strategy_prefix}_6_90 ${strategy_prefix}_6_100 ${strategy_prefix}_6_200 ${strategy_prefix}_6_300 ${strategy_prefix}_6_400 ${strategy_prefix}_6_500 ${strategy_prefix}_6_600 ${strategy_prefix}_6_700 \
        --self_eval \
        --results_dir "$RESULTS_DIR" \
        --modes  step_by_step  \
        --pos_types   abs rel_global \
        --function_type $function_type \
        --task_max_length_flag fixed \
        --task_max_length 6 \
        --seeds 0 10 20 30 40 \
        --plot_direct_sbs

# Figure 4(b) in the paper
strategy_prefix="combination"
function_type="diverse"
RESULTS_DIR="./results/plot_test/${function_type}/${strategy_prefix}"
mkdir -p "$RESULTS_DIR"
python -m scripts.analysis_plots \
    --prompt_length fixed \
    --train_splits ${strategy_prefix}_6 \
    --eval_split ${strategy_prefix}_1 ${strategy_prefix}_2 ${strategy_prefix}_3 ${strategy_prefix}_4 ${strategy_prefix}_5 ${strategy_prefix}_6 \
    --results_dir "$RESULTS_DIR" \
    --modes step_by_step \
    --pos_types abs rel_global \
    --plot_direct_sbs \
    --function_type $function_type \
    --task_max_length_flag fixed \
    --task_max_length 7 \
    --seeds 0 10 20 30 40 


# Figure 5(a) in the paper 
strategy_prefix="equivalence"
function_type="uniform"
RESULTS_DIR="./results/plot_test/${function_type}/${strategy_prefix}"
mkdir -p "$RESULTS_DIR"
python -m scripts.analysis_plots \
        --prompt_length fixed \
        --train_splits "equivalence_6_576_0" "equivalence_6_576_1" "equivalence_6_576_25" "equivalence_6_576_49" "equivalence_6_576_73" "equivalence_6_576_97" "equivalence_6_576_121" "equivalence_6_576_144" \
        --self_eval \
        --results_dir "$RESULTS_DIR" \
        --modes   direct   \
        --pos_types   abs rel_global \
        --function_type $function_type \
        --task_max_length_flag fixed \
        --task_max_length 7 \
        --self_eval \
        --seeds 0 10 20 30 40 \
        --plot_std 

# Figure 5(b) in the paper
strategy_prefix="uniequivalence_6_576"
function_type="uniform"
RESULTS_DIR="./results/plot_test/${function_type}/${strategy_prefix}"
mkdir -p "$RESULTS_DIR"
python -m scripts.analysis_plots \
        --prompt_length fixed \
        --train_splits ${strategy_prefix}_0 ${strategy_prefix}_1 ${strategy_prefix}_2 ${strategy_prefix}_3 ${strategy_prefix}_4 ${strategy_prefix}_5 \
        --self_eval \
        --results_dir "$RESULTS_DIR" \
        --modes  direct \
        --pos_types abs rel_global \
        --plot_std \
        --function_type $function_type \
        --task_max_length_flag fixed \
        --task_max_length 7 \
        --seeds 0 10 20 30 40 \

# Figure 6 in the paper.     
for strategy_prefix in "permutation" "permutationrandom"; do
    for function_type in "uniform" "diverse"; do
        RESULTS_DIR="./results/plot_test/${function_type}/${strategy_prefix}"
        mkdir -p "$RESULTS_DIR"
        python -m scripts.analysis_plots \
                --prompt_length fixed \
                --train_splits ${strategy_prefix}_6_1 ${strategy_prefix}_6_10 ${strategy_prefix}_6_100 ${strategy_prefix}_6_200 ${strategy_prefix}_6_300 ${strategy_prefix}_6_400 ${strategy_prefix}_6_500 ${strategy_prefix}_6_600 ${strategy_prefix}_6_700 \
                --self_eval \
                --results_dir "$RESULTS_DIR" \
                --modes  direct step_by_step  \
                --pos_types   abs rel_global \
                --function_type $function_type \
                --task_max_length_flag fixed \
                --task_max_length 6 \
                --seeds 0 10 20 30 40 \
                --plot_std 
    done
done