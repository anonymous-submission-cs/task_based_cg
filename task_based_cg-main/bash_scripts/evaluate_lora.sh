#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVAL_CONFIG="${PROJECT_ROOT}/config/eval/livecodebench.yaml"
DATA_ROOT="${PROJECT_ROOT}/data/livecodebench"
CKPT_ROOT="${PROJECT_ROOT}/models/ckpts/livecodebench_lora"

run_equiv_split() {
  local split_name="$1"
  local data_dir="${DATA_ROOT}/${split_name}/holdout_54/seed_0"
  local lora_path="${CKPT_ROOT}/${split_name}/holdout_54/seed_0/model_llama3/lora_r16_a32/final"

  python -m scripts.evaluate_livecodebench_equiv_leakage \
    --config "${EVAL_CONFIG}" \
    --model_name llama3 \
    --split test \
    --data_dir "${data_dir}" \
    --lora_path "${lora_path}"
}

run_heldout_split() {
  local shared_fraction="$1"
  local data_dir="${DATA_ROOT}/heldout_4_200/seed_0"
  local lora_path="${CKPT_ROOT}/equiv_leakage_shared_${shared_fraction}/holdout_54/seed_0/model_llama3/lora_r16_a32/final"

  python -m scripts.evaluate_livecodebench_equiv_leakage \
    --config "${EVAL_CONFIG}" \
    --model_name llama3 \
    --split test \
    --data_dir "${data_dir}" \
    --results_subdir "shared_${shared_fraction}" \
    --lora_path "${lora_path}"
}

run_equiv_split equiv_leakage_shared_0.0
run_equiv_split equiv_leakage_shared_1.0
run_equiv_split equiv_leakage_shared_0.66_heldoutfrac_0.0
run_equiv_split equiv_leakage_shared_0.66_heldoutfrac_0.5
run_equiv_split equiv_leakage_shared_0.66_heldoutfrac_1.0

run_heldout_split 0.0
run_heldout_split 1.0
