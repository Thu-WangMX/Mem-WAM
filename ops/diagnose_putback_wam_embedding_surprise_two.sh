#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_wam_embedding_surprise}"
runtime_bin="${RUNTIME_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin}"
output_dir="${OUTPUT_DIR:-/mnt/vepfs02/output/kevin_wang/memorywam/analysis/putback_wam_embedding_surprise_init_v1}"
dataset_root="${DATASET_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/data/rmbench_lerobot_v21_rgb_v3/put_back_block}"
latent_cache="${LATENT_CACHE:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_continuous_episode_stride16_v4}"
text_cache="${TEXT_CACHE:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_text_cache_rgb_v3}"
action_init="${ACTION_INIT:-/mnt/vepfs02/output/kevin_wang/memorywam/model_assets/fastwam_fullkv_official_init/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
model_base="${MODEL_BASE:-/mnt/vepfs02/datasets/kevin_wang/code/projects/FastWAM/checkpoints}"
preflight_only="${PREFLIGHT_ONLY:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PATH="${runtime_bin}:${PATH}"
export PYTHONPATH="${repo_root}/src:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFSYNTH_MODEL_BASE_PATH="${model_base}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export FASTWAM_ACTION_DIT_INIT="${action_init}"
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8

die() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ "${preflight_only}" == "0" || "${preflight_only}" == "1" ]] || die "PREFLIGHT_ONLY must be 0 or 1"
[[ -d "${repo_root}" ]] || die "repository not found: ${repo_root}"
[[ -x "${runtime_bin}/python" && -x "${runtime_bin}/torchrun" ]] || die "FastWAM runtime is incomplete: ${runtime_bin}"
[[ -f "${repo_root}/scripts/diagnose_putback_wam_embedding_surprise.py" ]] || die "diagnostic runner is missing"
[[ -f "${repo_root}/scripts/render_putback_wam_embedding_surprise.py" ]] || die "video renderer is missing"
[[ -s "${latent_cache}/manifest.json" ]] || die "latent cache is missing"
[[ -d "${text_cache}" ]] || die "text cache is missing"
[[ -s "${action_init}" ]] || die "ActionDiT initialization asset is missing"
[[ -d "${model_base}/Wan-AI/Wan2.2-TI2V-5B" ]] || die "Wan2.2 initialization assets are missing"
[[ -d "${dataset_root}/videos/chunk-000" ]] || die "PutBack camera videos are missing"
[[ ! -e "${output_dir}" ]] || die "refusing to overwrite output: ${output_dir}"

IFS=',' read -r -a gpu_ids <<<"${CUDA_VISIBLE_DEVICES}"
[[ "${#gpu_ids[@]}" -eq 2 ]] || die "exactly two visible GPUs are required"
[[ "${gpu_ids[0]}" != "${gpu_ids[1]}" ]] || die "GPU ids must be unique"

echo "host=$(hostname)"
echo "repo_root=${repo_root}"
echo "output_dir=${output_dir}"
echo "episodes=40,41"
echo "model_source=initialization"
echo "policy_checkpoint=null"
echo "feature_layer=-1"
echo "feature_window=8"
echo "statistics_window=8"
echo "threshold_window=8"
echo "min_history=4"
echo "gamma=1.0"
echo "nms_distance=2"
echo "segment_range=2..8"
echo "gpus=${CUDA_VISIBLE_DEVICES}"

if [[ "${preflight_only}" == "1" ]]; then
  echo "preflight_status=ok"
  exit 0
fi

command -v ffmpeg >/dev/null 2>&1 || die "ffmpeg is unavailable"
command -v ffprobe >/dev/null 2>&1 || die "ffprobe is unavailable"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"
physical_gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
[[ "${physical_gpu_count}" -ge 2 ]] || die "host exposes fewer than two GPUs"

log_file="${output_dir}.log"
cd "${repo_root}"
"${runtime_bin}/torchrun" \
  --standalone \
  --nproc_per_node=2 \
  scripts/diagnose_putback_wam_embedding_surprise.py \
  --repo "${repo_root}" \
  --output "${output_dir}" \
  --latent-cache "${latent_cache}" \
  --text-cache "${text_cache}" \
  --dataset-root "${dataset_root}" \
  --action-init "${action_init}" \
  --model-base "${model_base}" \
  --episodes 40,41 \
  --feature-window 8 \
  --feature-layer -1 \
  --statistics-window 8 \
  --threshold-window 8 \
  --min-history 4 \
  --gamma 1.0 \
  --nms-distance 2 \
  --min-segment 2 \
  --max-segment 8 \
  --seed 42 \
  2>&1 | tee "${log_file}"

mkdir -p "${output_dir}/videos"
for episode in 40 41; do
  video="${output_dir}/videos/episode_$(printf '%03d' "${episode}")_wam_embedding_surprise.mp4"
  "${runtime_bin}/python" scripts/render_putback_wam_embedding_surprise.py \
    --analysis-root "${output_dir}" \
    --dataset-root "${dataset_root}" \
    --episode "${episode}" \
    --output "${video}"
  codec="$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=nw=1:nk=1 "${video}")"
  duration="$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "${video}")"
  [[ "${codec}" == "h264" ]] || die "episode ${episode} video codec is ${codec}, expected h264"
  "${runtime_bin}/python" - "${duration}" <<'PY'
import sys
if float(sys.argv[1]) <= 0:
    raise SystemExit("video duration must be positive")
PY
  echo "video_ok episode=${episode} codec=${codec} duration=${duration} path=${video}"
done

[[ -s "${output_dir}/aggregate.json" && -s "${output_dir}/initialization_manifest.json" ]] || die "diagnostic reports are incomplete"
echo "diagnostic_status=complete"
