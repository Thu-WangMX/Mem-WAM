#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_multires_latent_kernel_k8}"
runtime_bin="${RUNTIME_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin}"
output_dir="${OUTPUT_DIR:-/mnt/vepfs01/output/kevin.wang/memorywam/train/multires_latent_kernel_k8_putback_init_joint20k_lr2e4_seed42_v1}"
manifest_root="${LATENT_KERNEL_REGIME_MANIFEST:-/mnt/vepfs02/output/kevin_wang/memorywam/analysis/trainfree_multires_latent_kernel_v1/putback}"
task_name=rmbench_putback_latent_kernel_regime_k8_init_20k
zero2_config=scripts/accelerate_configs/accelerate_zero2_ds.yaml
preflight_only="${PREFLIGHT_ONLY:-0}"
require_idle_gpus="${REQUIRE_IDLE_GPUS:-0}"

die() { echo "ERROR: $*" >&2; exit 2; }

[[ "${preflight_only}" == 0 || "${preflight_only}" == 1 ]] || die "PREFLIGHT_ONLY must be 0 or 1"
[[ "${require_idle_gpus}" == 0 || "${require_idle_gpus}" == 1 ]] || die "REQUIRE_IDLE_GPUS must be 0 or 1"
[[ -d "${repo_root}" ]] || die "Missing isolated repository: ${repo_root}"
[[ -x "${runtime_bin}/python" ]] || die "Missing Python runtime: ${runtime_bin}/python"
[[ -s "${repo_root}/${zero2_config}" ]] || die "Missing ZeRO2 config"
[[ "$(readlink -m "${output_dir}")" == /mnt/vepfs01/output/* ]] || die "OUTPUT_DIR must be on vepfs01"

export PATH="${runtime_bin}:${PATH}"
export PYTHONPATH="${repo_root}/src:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/vepfs02/datasets/kevin_wang/code/projects/FastWAM/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export RMBENCH_PUTBACK_LEROBOT="${RMBENCH_PUTBACK_LEROBOT:-/mnt/vepfs02/output/kevin_wang/memorywam/data/rmbench_lerobot_v21_rgb_v3/put_back_block}"
export MEMORYWAM_PUTBACK_STATS="${MEMORYWAM_PUTBACK_STATS:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_stats_rgb_v3/dataset_stats.json}"
export MEMORYWAM_PUTBACK_CONTINUOUS_LATENTS="${MEMORYWAM_PUTBACK_CONTINUOUS_LATENTS:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_continuous_episode_stride16_v4}"
export MEMORYWAM_PUTBACK_TEXT_CACHE="${MEMORYWAM_PUTBACK_TEXT_CACHE:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_text_cache_rgb_v3}"
export FASTWAM_ACTION_DIT_INIT="${FASTWAM_ACTION_DIT_INIT:-/mnt/vepfs02/output/kevin_wang/memorywam/model_assets/fastwam_fullkv_official_init/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
export LATENT_KERNEL_REGIME_MANIFEST="${manifest_root}"
export PUTBACK_LATENT_KERNEL_OUTPUT="${output_dir}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1

for required in \
  "${FASTWAM_ACTION_DIT_INIT}" \
  "${RMBENCH_PUTBACK_LEROBOT}/meta/info.json" \
  "${MEMORYWAM_PUTBACK_STATS}" \
  "${MEMORYWAM_PUTBACK_CONTINUOUS_LATENTS}/manifest.json" \
  "${manifest_root}/manifest.json"; do
  [[ -s "${required}" ]] || die "Missing or empty artifact: ${required}"
done
[[ -d "${MEMORYWAM_PUTBACK_TEXT_CACHE}" ]] || die "Missing text cache"

cd "${repo_root}"
manifest_summary="$(${runtime_bin}/python - "${manifest_root}" <<'PY'
import collections
import json
import sys
from fastwam.memory.latent_kernel_regime import LatentKernelManifestStore

store = LatentKernelManifestStore(sys.argv[1], expected_episode_count=50)
lengths = collections.Counter()
reasons = collections.Counter()
for episode in range(50):
    rows = store.segments_for_episode(episode)
    lengths.update(row.length for row in rows)
    reasons.update(row.reason for row in rows)
if len(lengths) < 4:
    raise SystemExit(f"insufficient dynamic lengths: {dict(lengths)}")
if not reasons["multires_latent_kernel_regime"] or not reasons["stable_max_length"]:
    raise SystemExit(f"selector collapsed: {dict(reasons)}")
ratio = float(store.metadata.get("event_to_candidate_score_ratio", 0.0))
if ratio < 1.5:
    raise SystemExit(f"weak visual selection ratio: {ratio}")
print(json.dumps({"lengths": dict(sorted(lengths.items())), "reasons": dict(sorted(reasons.items())), "event_score_ratio": ratio}))
PY
)"

resolved_config="$(${runtime_bin}/python scripts/train.py "task=${task_name}" --cfg job --resolve)"
RESOLVED_CONFIG="${resolved_config}" OUTPUT_EXPECTED="${output_dir}" \
  ACTION_INIT_EXPECTED="${FASTWAM_ACTION_DIT_INIT}" MANIFEST_EXPECTED="${manifest_root}" \
  "${runtime_bin}/python" - <<'PY' || die "Resolved initialization-training contract mismatch"
import os
from omegaconf import OmegaConf

cfg = OmegaConf.create(os.environ["RESOLVED_CONFIG"])
expected = {
    "output_dir": os.environ["OUTPUT_EXPECTED"],
    "initial_weights": None,
    "resume": None,
    "learning_rate": 2e-4,
    "max_steps": 20000,
    "save_every": 0,
    "save_steps": [5000, 10000, 15000, 16000, 17000, 18000, 19000, 20000],
    "save_training_state": True,
    "save_training_state_every": 10000,
    "batch_size": 1,
    "num_gpus": 8,
    "gradient_accumulation_steps": 1,
    "mixed_precision": "bf16",
    "seed": 42,
    "native_cache_train_mode": "full",
    "data.train._target_": "fastwam.datasets.lerobot.latent_kernel_regime_dataset.LatentKernelRegimeRobotVideoDataset",
    "data.train.latent_kernel_manifest_path": os.environ["MANIFEST_EXPECTED"],
    "data.train.memory_tokens": 8,
    "data.train.minimum_history_frames": 1,
    "model.native_cache.mode": "layerwise",
    "model.native_cache.memory_tokens": 8,
    "model.native_cache.anchor_frames": 2,
    "model.native_cache.recent_frames": 4,
    "model.native_cache.recursive": False,
    "model.loss.lambda_video": 1.0,
    "model.loss.lambda_action": 1.0,
    "model.action_dit_pretrained_path": os.environ["ACTION_INIT_EXPECTED"],
    "model.skip_dit_load_from_pretrain": False,
}
for path, want in expected.items():
    got = OmegaConf.select(cfg, path)
    if got != want:
        raise SystemExit(f"{path}: expected {want!r}, got {got!r}")
text = os.environ["RESOLVED_CONFIG"].lower()
if "robotwin_uncond_3cam_384.pt" in text:
    raise SystemExit("RobotWin pretrained checkpoint leaked into config")
PY

echo "task=put_back_block selector=train_free_multires_latent_kernel memory=fixed_K8"
echo "selector_inputs=causal_vae_latents_only action=false proprio=false fitted=false"
echo "initialization=generic_wan22_plus_official_actiondit robotwin_pretrained=false"
echo "joint=reader_video_action_and_compressor steps=0_to_20000 lr=2e-4"
echo "gpus=8 backend=deepspeed_zero2 sharing_allowed=$((1-require_idle_gpus))"
echo "anchor=2 recent=4 min_L=4 max_L=8 minimum_history_frames=1"
echo "checkpoint_steps=5000,10000,15000,16000,17000,18000,19000,20000"
echo "manifest=${manifest_root} summary=${manifest_summary}"
echo "output_dir=${output_dir}"

if [[ "${preflight_only}" == 1 ]]; then
  echo "preflight=ok"
  exit 0
fi
[[ ! -e "${output_dir}" ]] || die "Refusing to overwrite existing output: ${output_dir}"
IFS=',' read -r -a visible_gpus <<<"${CUDA_VISIBLE_DEVICES}"
[[ ${#visible_gpus[@]} -eq 8 ]] || die "Formal training requires exactly eight visible GPUs"
if [[ "${require_idle_gpus}" == 1 ]]; then
  active_compute_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -u)"
  [[ -z "${active_compute_pids}" ]] || die "GPU compute processes are active"
fi

mkdir -p "${output_dir}/contracts"
printf '%s\n' "${resolved_config}" >"${output_dir}/contracts/train_20k_resolved.yaml"
printf '%s\n' "${manifest_summary}" >"${output_dir}/contracts/latent_kernel_manifest_summary.json"
exec > >(tee -a "${output_dir}/train.log") 2>&1
accelerate launch \
  --config_file "${zero2_config}" \
  --num_processes 8 \
  scripts/train.py "task=${task_name}"

for step in 5000 10000 15000 16000 17000 18000 19000 20000; do
  checkpoint="${output_dir}/checkpoints/weights/step_$(printf '%06d' "${step}").pt"
  [[ -s "${checkpoint}" ]] || die "Missing expected checkpoint: ${checkpoint}"
done
echo "complete output_dir=${output_dir}"
