#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LORA_CONFIG="${PROJECT_ROOT}/config/finetune/livecodebench_lora.yaml"

ORIGINAL_CONFIG="$(mktemp)"
cp "${LORA_CONFIG}" "${ORIGINAL_CONFIG}"
trap 'mv "${ORIGINAL_CONFIG}" "${LORA_CONFIG}"' EXIT

run_lora_split() {
  local data_subdir="$1"
  cp "${ORIGINAL_CONFIG}" "${LORA_CONFIG}"
  sed -i -E "s|^([[:space:]]*data_subdir:) .*|\\1 \"${data_subdir}\"|" "${LORA_CONFIG}"
  python -m scripts.finetune_livecodebench_lora
}

run_lora_split "equiv_leakage_shared_0.0"
run_lora_split "equiv_leakage_shared_1.0"
run_lora_split "equiv_leakage_shared_0.66_heldoutfrac_0.0"
run_lora_split "equiv_leakage_shared_0.66_heldoutfrac_0.5"
run_lora_split "equiv_leakage_shared_0.66_heldoutfrac_1.0"
