#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/vepfs02/output/kevin.wang/memorywam/code/fastwam_event_conditioned_rate}"
runtime_bin="${RUNTIME_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin}"
memorywam_root="${MEMORYWAM_ROOT:-/mnt/vepfs02/output/kevin.wang/memorywam}"
battery_root="${BATTERY_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_fullkv_remaining8_rgb_v3/battery_try}"

dataset_root="${RMBENCH_BATTERY_LEROBOT:-${battery_root}/lerobot/battery_try}"
latent_cache="${MEMORYWAM_BATTERY_CONTINUOUS_LATENTS:-${battery_root}/temporal_fullkv_continuous_episode_stride16_v4}"
text_cache="${MEMORYWAM_BATTERY_TEXT_CACHE:-${battery_root}/text_cache}"
model_base="${DIFFSYNTH_MODEL_BASE_PATH:-${memorywam_root}/model_assets/diffsynth_official}"
action_init="${FASTWAM_ACTION_DIT_INIT:-${memorywam_root}/model_assets/fastwam_fullkv_official_init/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"

# This immutable reference identifies the exact initialization used to build the
# frozen v3 selector. Its asset paths may be historical, but all asset hashes are
# checked against the new kevin.wang copies supplied above.
init_reference="${INIT_REFERENCE:-/mnt/vepfs02/output/kevin_wang/memorywam/analysis/putback_wam_embedding_surprise_init_v1/initialization_manifest.json}"
selector_root="${FROZEN_SELECTOR_ROOT:-${memorywam_root}/analysis/putback_control_information_selector_v3}"
runtime_lock="${RUNTIME_LOCK:-${selector_root}/selector_runtime/locked_selector.json}"
contextual_statistics="${CONTEXTUAL_STATISTICS:-${selector_root}/selector_candidate/contextual_statistics.pt}"
pca="${FROZEN_PCA:-${memorywam_root}/analysis/putback_four_phase_wam_pca64_v1.pt}"
feature_predictor="${FROZEN_FEATURE_PREDICTOR:-${memorywam_root}/analysis/putback_feature_predictors_pca64_v1/visual_action/visual_action.fpbin}"
control_probe="${FROZEN_CONTROL_PROBE:-${memorywam_root}/analysis/putback_control_information_selector_v1/wam_proprio/wam_proprio.cipbin}"
selector_normalization="${FROZEN_SELECTOR_NORMALIZATION:-${memorywam_root}/analysis/putback_init_per_episode_noise_2ep_20260813_r3/.work/rank_0/dataset_stats.json}"

analysis_root="${ANALYSIS_ROOT:-${memorywam_root}/analysis/battery_control_information_selector_v3_transfer}"
phase0_bank="${PHASE0_BANK:-${analysis_root}/initialization_wam_phase0_bank}"
four_phase_latents="${FOUR_PHASE_LATENTS:-${analysis_root}/four_phase_latents}"
feature_bank="${FEATURE_BANK:-${analysis_root}/four_phase_multilayer_features}"
probe_dataset="${PROBE_DATASET:-${analysis_root}/battery_probe_dataset.pt}"
trace_output="${TRACE_OUTPUT:-${analysis_root}/frozen_replay_causal_online}"
manifest_root="${MANIFEST_ROOT:-${memorywam_root}/data/battery_control_information_planning_segments_v3_causal_online}"
preflight_only="${PREFLIGHT_ONLY:-0}"

die() { echo "ERROR: $*" >&2; exit 2; }
[[ "${preflight_only}" == "0" || "${preflight_only}" == "1" ]] || die "PREFLIGHT_ONLY must be 0 or 1"
[[ -d "${repo_root}" && -x "${runtime_bin}/python" && -x "${runtime_bin}/torchrun" ]] || die "repository/runtime missing"

for required in \
  "${repo_root}/scripts/extract_putback_init_wam_embedding_bank.py" \
  "${repo_root}/scripts/build_putback_four_phase_latents.py" \
  "${repo_root}/scripts/extract_putback_four_phase_wam_features.py" \
  "${repo_root}/scripts/build_putback_control_information_dataset.py" \
  "${repo_root}/scripts/prepare_battery_control_information_manifest_v3.py" \
  "${dataset_root}/meta/info.json" \
  "${latent_cache}/manifest.json" \
  "${action_init}" \
  "${init_reference}" \
  "${runtime_lock}" \
  "${contextual_statistics}" \
  "${pca}" \
  "${feature_predictor}" \
  "${control_probe}" \
  "${selector_normalization}" \
  "${model_base}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors" \
  "${model_base}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors" \
  "${model_base}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors"; do
  [[ -s "${required}" ]] || die "missing required artifact: ${required}"
done
[[ -d "${text_cache}" ]] || die "missing Battery text cache: ${text_cache}"
vae_path="${VAE_PATH:-${model_base}/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors}"
[[ -s "${vae_path}" ]] || die "missing Wan2.2 VAE: ${vae_path}"

"${runtime_bin}/python" - "${latent_cache}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
manifest = json.loads((root / "manifest.json").read_text())
metadata = manifest.get("metadata", {})
assert metadata.get("complete") is True
assert metadata.get("replan_stride") == 16
assert int(metadata.get("episode_count", -1)) == 50
episodes = manifest.get("episodes", {})
assert isinstance(episodes, dict)
for episode in range(50):
    suffix = f"/{episode}"
    matches = [
        value
        for key, value in episodes.items()
        if (str(key) == str(episode) or str(key).endswith(suffix))
        and isinstance(value, str)
    ]
    assert len(matches) == 1, (episode, matches)
    path = root / matches[0]
    assert path.is_file() and path.stat().st_size > 0, path
print("battery_latent_manifest_status=ok")
PY

"${runtime_bin}/python" - "${dataset_root}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
rows = [
    json.loads(line)
    for line in (root / "meta" / "episodes.jsonl").read_text().splitlines()
    if line.strip()
]
assert len(rows) == 50
assert sorted(int(row["episode_index"]) for row in rows) == list(range(50))
for row in rows:
    raw = Path(row["raw_file_name"])
    assert raw.is_file() and raw.stat().st_size > 0, raw
    assert int(row["length"]) > 16
print("battery_raw_episode_contract_status=ok")
PY

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a gpu_ids <<<"${CUDA_VISIBLE_DEVICES}"
[[ "${#gpu_ids[@]}" -eq 8 ]] || die "exactly eight visible GPUs are required"
[[ "$(printf '%s\n' "${gpu_ids[@]}" | sort -u | wc -l | tr -d ' ')" -eq 8 ]] || die "GPU indices must be unique"

echo "method=v6_event_full_forced_half"
echo "selector_transfer=putback_locked_v3_to_battery_zero_refit"
echo "selector_model=initialization_wam_frozen"
echo "episodes=50 detector_stride=4 replan_stride=16"
echo "analysis_root=${analysis_root}"
echo "four_phase_latents=${four_phase_latents}"
echo "manifest_root=${manifest_root}"
echo "gpus=${CUDA_VISIBLE_DEVICES}"
if [[ "${preflight_only}" == "1" ]]; then
  echo "preflight_status=ok"
  exit 0
fi

export PATH="${runtime_bin}:${PATH}"
export PYTHONPATH="${repo_root}/src:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFSYNTH_MODEL_BASE_PATH="${model_base}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export FASTWAM_ACTION_DIT_INIT="${action_init}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
cd "${repo_root}"

mkdir -p "${analysis_root}"
if [[ ! -s "${phase0_bank}/bank_manifest.json" ]]; then
  "${runtime_bin}/torchrun" --standalone --nproc_per_node=8 \
    scripts/extract_putback_init_wam_embedding_bank.py \
    --repo "${repo_root}" \
    --output "${phase0_bank}" \
    --latent-cache "${latent_cache}" \
    --text-cache "${text_cache}" \
    --dataset-root "${dataset_root}" \
    --action-init "${action_init}" \
    --model-base "${model_base}" \
    --initialization-reference "${init_reference}" \
    --episodes 0-49 \
    --feature-window 8 \
    --feature-layer -1 \
    --seed 42 \
    2>&1 | tee -a "${analysis_root}/phase0_bank.log"
fi

if [[ ! -s "${four_phase_latents}/manifest.json" ]]; then
  "${runtime_bin}/torchrun" --standalone --nproc_per_node=8 \
    scripts/build_putback_four_phase_latents.py \
    --lerobot-root "${dataset_root}" \
    --vae-path "${vae_path}" \
    --output "${four_phase_latents}" \
    --episodes 0-49 \
    2>&1 | tee -a "${analysis_root}/four_phase_latents.log"
fi

"${runtime_bin}/python" - "${four_phase_latents}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
manifest = json.loads((root / "manifest.json").read_text())
assert manifest.get("complete") is True
assert manifest.get("phase0_equivalence_pending") is True
assert int(manifest.get("episode_count", -1)) == 50
assert manifest.get("phase_offsets") == [0, 4, 8, 12]
rows = manifest.get("episodes", [])
assert [int(row["episode"]) for row in rows] == list(range(50))
for row in rows:
    path = root / row["relative_path"]
    assert path.is_file() and path.stat().st_size > 0, path
print("battery_four_phase_latent_status=ok")
PY

if [[ ! -s "${feature_bank}/bank_manifest.json" ]]; then
  "${runtime_bin}/torchrun" --standalone --nproc_per_node=8 \
    scripts/extract_putback_four_phase_wam_features.py \
    --repo "${repo_root}" \
    --output "${feature_bank}" \
    --latent-cache "${four_phase_latents}" \
    --text-cache "${text_cache}" \
    --dataset-root "${dataset_root}" \
    --action-init "${action_init}" \
    --model-base "${model_base}" \
    --initialization-reference "${init_reference}" \
    --phase0-bank "${phase0_bank}" \
    --episodes 0-49 \
    --feature-window 8 \
    --seed 42 \
    2>&1 | tee -a "${analysis_root}/four_phase_features.log"
fi

if [[ ! -s "${probe_dataset}" ]]; then
  "${runtime_bin}/python" scripts/build_putback_control_information_dataset.py \
    --feature-bank "${feature_bank}" \
    --pca "${pca}" \
    --dataset-root "${dataset_root}" \
    --normalization-stats "${selector_normalization}" \
    --output "${probe_dataset}" \
    --episodes 0-49 \
    --horizon 16 \
    2>&1 | tee -a "${analysis_root}/probe_dataset.log"
fi

if [[ ! -s "${manifest_root}/manifest.json" ]]; then
  "${runtime_bin}/python" scripts/prepare_battery_control_information_manifest_v3.py \
    --feature-bank "${feature_bank}" \
    --pca "${pca}" \
    --dataset-root "${dataset_root}" \
    --probe-dataset "${probe_dataset}" \
    --feature-predictor "${feature_predictor}" \
    --control-probe "${control_probe}" \
    --contextual-statistics "${contextual_statistics}" \
    --runtime-lock "${runtime_lock}" \
    --analysis-output "${trace_output}" \
    --manifest-output "${manifest_root}" \
    --device cuda:0 \
    --max-history 8 \
    2>&1 | tee -a "${analysis_root}/frozen_replay.log"
fi

echo "battery_v6_manifest_status=complete"
echo "manifest=${manifest_root}/manifest.json"
