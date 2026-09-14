#!/usr/bin/env bash
set -euo pipefail

repo_root=${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_dynamic_multiframe_selection}
output_dir=${OUTPUT_DIR:-/mnt/vepfs02/output/kevin_wang/memorywam/analysis/putback_four_phase_multilayer_wam_features_v1}
four_phase_latents=${FOUR_PHASE_LATENTS:?FOUR_PHASE_LATENTS is required}
text_cache=${TEXT_CACHE:?TEXT_CACHE is required}
dataset_root=${DATASET_ROOT:?DATASET_ROOT is required}
action_init=${ACTION_INIT:?ACTION_INIT is required}
model_base=${MODEL_BASE:?MODEL_BASE is required}
init_reference=${INIT_REFERENCE:?INIT_REFERENCE is required}
phase0_bank=${PHASE0_BANK:?PHASE0_BANK is required}
runtime_bin=${RUNTIME_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin}
preflight_only=${PREFLIGHT_ONLY:-0}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
[[ -d "$repo_root" ]] || die "repository missing"
[[ -f "$repo_root/scripts/extract_putback_four_phase_wam_features.py" ]] || die "feature extractor missing"
[[ -s "$four_phase_latents/manifest.json" ]] || die "four-phase latent manifest missing"
[[ -d "$text_cache" ]] || die "text cache missing"
[[ -d "$dataset_root" ]] || die "dataset root missing"
[[ -s "$action_init" ]] || die "ActionDiT initialization missing"
[[ -d "$model_base" ]] || die "model base missing"
[[ -s "$init_reference" ]] || die "initialization reference missing"
[[ -s "$phase0_bank/bank_manifest.json" ]] || die "phase-0 reference bank missing"
[[ -x "$runtime_bin/python" && -x "$runtime_bin/torchrun" ]] || die "runtime missing"
[[ ! -e "$output_dir/bank_manifest.json" ]] || die "refusing complete feature bank"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
[[ "${#gpu_ids[@]}" -eq 8 ]] || die "exactly eight visible GPUs are required"
[[ "$(printf '%s\n' "${gpu_ids[@]}" | sort -u | wc -l | tr -d ' ')" -eq 8 ]] || die "exactly eight unique visible GPUs are required"

printf 'preflight_status=ok\n'
printf 'feature_layers=5,11,17,23,29\n'
printf 'feature_regions=global,left_wrist,right_wrist,head\n'
printf 'model_source=initialization\n'
printf 'policy_checkpoint=null\n'
if [[ "$preflight_only" == 1 ]]; then
  exit 0
fi

export PATH="$runtime_bin:$PATH"
export PYTHONPATH="$repo_root/src:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export DIFFSYNTH_MODEL_BASE_PATH="$model_base"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export FASTWAM_ACTION_DIT_INIT="$action_init"
cd "$repo_root"
"$runtime_bin/torchrun" --standalone --nproc_per_node=8 \
  scripts/extract_putback_four_phase_wam_features.py \
  --repo "$repo_root" \
  --output "$output_dir" \
  --latent-cache "$four_phase_latents" \
  --text-cache "$text_cache" \
  --dataset-root "$dataset_root" \
  --action-init "$action_init" \
  --model-base "$model_base" \
  --initialization-reference "$init_reference" \
  --phase0-bank "$phase0_bank" \
  --episodes 0-49 \
  --feature-window 8 \
  --seed 42 \
  2>&1 | tee -a "${output_dir}.log"
