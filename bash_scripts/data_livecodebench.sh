#!/bin/bash
set -euo pipefail

# Minimal data-generation entrypoint for LiveCodeBench reproducibility graphs.
# Recreates the datasets used by the finetuned_livecodebench analysis notebooks.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${LIVE_CODEBENCH_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONFIG_PATH="${LIVE_CODEBENCH_CONFIG:-${PROJECT_ROOT}/config/gen/livecodebench.yaml}"
FILTERED_IDS_JSON="${LIVE_CODEBENCH_FILTERED_IDS_JSON:-${PROJECT_ROOT}/config/gen/livecodebench-jsons/livecodebench_filtered_ids.json}"
HELDOUT_FUNCTION_NAMES_JSON="${LIVE_CODEBENCH_HELDOUT_FUNCTIONS_JSON:-${PROJECT_ROOT}/config/gen/livecodebench-jsons/heldout_functions.json}"
FILTERED_FUNCTIONS_JSON="${LIVE_CODEBENCH_FILTERED_FUNCTIONS_JSON:-${PROJECT_ROOT}/config/gen/livecodebench-jsons/filtered_functions.json}"
HELDOUT_4_200_DIR="${LIVE_CODEBENCH_HELDOUT_4_200_DIR:-${PROJECT_ROOT}/data/livecodebench/heldout_4_200}"

SEED=0
HOLDOUT_SIZE=54
NUM_SHARDS=4

run_equiv_leakage_split() {
  local shared_fraction="$1"
  local heldout_fraction="$2"
  local data_subdir="$3"

  local args=(--config "${CONFIG_PATH}" --mode equiv_leakage --num_shards "${NUM_SHARDS}" --seed "${SEED}" --holdout_size "${HOLDOUT_SIZE}" --shared_fraction "${shared_fraction}" --filtered_ids_json "${FILTERED_IDS_JSON}" --heldout_function_names_json "${HELDOUT_FUNCTION_NAMES_JSON}")
  if [[ -n "${heldout_fraction}" ]]; then
    args+=(--heldout_function_fraction "${heldout_fraction}")
  fi

  for shard_id in 0 1 2 3; do
    python -m scripts.generate_livecodebench "${args[@]}" --shard_id "${shard_id}"
  done

  python -m scripts.merge_livecodebench_shards \
    --holdout_size "${HOLDOUT_SIZE}" \
    --seed "${SEED}" \
    --num_shards "${NUM_SHARDS}" \
    --data_subdir "${data_subdir}"
}

run_heldout_4_200() {
  local functions_json="$1"
  python -m scripts.generate_livecodebench_filtered_functions_test \
    --function_names_json "${functions_json}" \
    --per_function 200 \
    --seed "${SEED}" \
    --output_dir "${HELDOUT_4_200_DIR}/seed_${SEED}"
}

run_equiv_leakage_split 0.0 "" equiv_leakage_shared_0.0
run_equiv_leakage_split 1.0 "" equiv_leakage_shared_1.0

run_equiv_leakage_split 0.66 0.0 equiv_leakage_shared_0.66_heldoutfrac_0.0
run_equiv_leakage_split 0.66 0.5 equiv_leakage_shared_0.66_heldoutfrac_0.5
run_equiv_leakage_split 0.66 1.0 equiv_leakage_shared_0.66_heldoutfrac_1.0

run_heldout_4_200 "${FILTERED_FUNCTIONS_JSON}"


