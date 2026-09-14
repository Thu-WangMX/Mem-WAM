#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_wam_embedding_surprise}"
output_dir="${OUTPUT_DIR:-/mnt/vepfs02/output/kevin_wang/memorywam/analysis/putback_init_wam_embedding_bank_v1}"
latent_cache="${LATENT_CACHE:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_continuous_episode_stride16_v4}"
text_cache="${TEXT_CACHE:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_text_cache_rgb_v3}"
dataset_root="${DATASET_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/data/rmbench_lerobot_v21_rgb_v3/put_back_block}"
action_init="${ACTION_INIT:-/mnt/vepfs02/output/kevin_wang/memorywam/model_assets/fastwam_fullkv_official_init/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
model_base="${MODEL_BASE:-/mnt/vepfs02/datasets/kevin_wang/code/projects/FastWAM/checkpoints}"
init_reference="${INIT_REFERENCE:-/mnt/vepfs02/output/kevin_wang/memorywam/analysis/putback_wam_embedding_surprise_init_v1/initialization_manifest.json}"
runtime_bin="${RUNTIME_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin}"
preflight_only="${PREFLIGHT_ONLY:-0}"

die() { echo "ERROR: $*" >&2; exit 2; }
[[ "$preflight_only" == 0 || "$preflight_only" == 1 ]] || die "PREFLIGHT_ONLY must be 0 or 1"
[[ -d "$repo_root" ]] || die "repository missing: $repo_root"
[[ -x "$runtime_bin/python" && -x "$runtime_bin/torchrun" ]] || die "runtime missing python/torchrun: $runtime_bin"
[[ -f "$repo_root/scripts/extract_putback_init_wam_embedding_bank.py" ]] || die "bank extractor missing"
[[ -s "$latent_cache/manifest.json" ]] || die "latent manifest missing"
[[ -d "$text_cache" ]] || die "text cache missing"
[[ -d "$dataset_root" ]] || die "dataset root missing"
[[ -s "$action_init" ]] || die "ActionDiT initialization missing"
[[ -d "$model_base/Wan-AI/Wan2.2-TI2V-5B" ]] || die "Wan initialization root missing"
[[ -s "$init_reference" ]] || die "initialization reference missing"
[[ ! -e "$output_dir/bank_manifest.json" ]] || die "refusing to overwrite complete feature bank"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a gpu_ids <<<"$CUDA_VISIBLE_DEVICES"
[[ "${#gpu_ids[@]}" -eq 8 ]] || die "exactly eight visible GPUs are required"
unique_count="$(printf '%s\n' "${gpu_ids[@]}" | sort -u | wc -l | tr -d ' ')"
[[ "$unique_count" -eq 8 ]] || die "exactly eight unique visible GPUs are required"

echo "repo_root=$repo_root"
echo "output_dir=$output_dir"
echo "episodes=0-49"
echo "gpus=$CUDA_VISIBLE_DEVICES"
echo "model_source=initialization"
echo "policy_checkpoint=null"
if [[ "$preflight_only" == 1 ]]; then
  echo "preflight_status=ok"
  exit 0
fi

command -v nvidia-smi >/dev/null || die "nvidia-smi unavailable"
physical_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
[[ "$physical_count" -ge 8 ]] || die "host exposes fewer than eight GPUs"
export PATH="$runtime_bin:$PATH"
export PYTHONPATH="$repo_root/src:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export DIFFSYNTH_MODEL_BASE_PATH="$model_base"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export FASTWAM_ACTION_DIT_INIT="$action_init"
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8

cd "$repo_root"
"$runtime_bin/torchrun" --standalone --nproc_per_node=8 \
  scripts/extract_putback_init_wam_embedding_bank.py \
  --repo "$repo_root" \
  --output "$output_dir" \
  --latent-cache "$latent_cache" \
  --text-cache "$text_cache" \
  --dataset-root "$dataset_root" \
  --action-init "$action_init" \
  --model-base "$model_base" \
  --initialization-reference "$init_reference" \
  --episodes 0-49 \
  --feature-window 8 \
  --feature-layer -1 \
  --seed 42 \
  2>&1 | tee -a "${output_dir}.log"
echo "feature_bank_status=complete"
