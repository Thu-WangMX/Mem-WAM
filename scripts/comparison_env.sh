#!/usr/bin/env bash
# Source this file; shared data and pretrained weights are read-only inputs.
COMPARISON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export COMPARISON_ROOT
export COMPARISON_VENV="${COMPARISON_VENV:-/mnt/vepfs01/output/spidy.wang/runtime/starwam-libero/venv}"
export PATH="${COMPARISON_VENV}/bin:${PATH}"
export PYTHONPATH="${COMPARISON_ROOT}/runtime/python:${COMPARISON_ROOT}/src:${COMPARISON_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="${COMPARISON_ROOT}/runtime/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_EXTENSIONS_DIR="${COMPARISON_ROOT}/runtime/torch_extensions"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/vepfs02/datasets/kevin_wang/code/projects/FastWAM/checkpoints}"
export FASTWAM_ACTION_DIT_INIT="${FASTWAM_ACTION_DIT_INIT:-/mnt/vepfs02/output/kevin_wang/memorywam/model_assets/fastwam_fullkv_official_init/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
export RMBENCH_TASK="${RMBENCH_TASK:-rearrange_blocks}"
if [[ ! "${RMBENCH_TASK}" =~ ^[A-Za-z0-9_]+$ ]]; then
  echo 'Invalid RMBENCH_TASK' >&2; return 1
fi

# Keep this mapping aligned with Kevin's validated nine-task training spec:
# configs/data/ninetask_anchor1_event_spec.json.  The historical data tree is
# split across three layouts, so a single remaining8 default is insufficient.
case "${RMBENCH_TASK}" in
  battery_try|blocks_ranking_try|press_button|rearrange_blocks|swap_T|swap_blocks)
    default_task_root="/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_fullkv_remaining8_rgb_v3/${RMBENCH_TASK}"
    ;;
  cover_blocks)
    default_task_root="/mnt/vepfs02/output/kevin_wang/memorywam/data/memorywam_fullattention_cover_blocks_rgb_v3"
    ;;
  observe_and_pickup)
    default_task_root="/mnt/vepfs02/output/kevin_wang/memorywam/data/memorywam_fullattention_observe_and_pickup_rgb_v3"
    ;;
  put_back_block)
    default_task_root=""
    ;;
  *)
    echo "Unsupported RMBench task: ${RMBENCH_TASK}" >&2; return 1
    ;;
esac

RMBENCH_TASK_ROOT="${RMBENCH_TASK_ROOT:-${default_task_root}}"
# These historical env names are retained by the colleague's common dataset config.
if [[ "${RMBENCH_TASK}" == put_back_block ]]; then
  export RMBENCH_COVER_BLOCKS_LEROBOT="${RMBENCH_COVER_BLOCKS_LEROBOT:-/mnt/vepfs02/output/kevin_wang/memorywam/data/rmbench_lerobot_v21_rgb_v3/put_back_block}"
  export MEMORYWAM_COVER_BLOCKS_STATS="${MEMORYWAM_COVER_BLOCKS_STATS:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_stats_rgb_v3/dataset_stats.json}"
  export MEMORYWAM_COVER_BLOCKS_CONTINUOUS_LATENTS="${MEMORYWAM_COVER_BLOCKS_CONTINUOUS_LATENTS:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_continuous_episode_stride16_v4}"
  export MEMORYWAM_COVER_BLOCKS_TEXT_CACHE="${MEMORYWAM_COVER_BLOCKS_TEXT_CACHE:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_text_cache_rgb_v3}"
else
  export RMBENCH_COVER_BLOCKS_LEROBOT="${RMBENCH_COVER_BLOCKS_LEROBOT:-${RMBENCH_TASK_ROOT}/lerobot/${RMBENCH_TASK}}"
  export MEMORYWAM_COVER_BLOCKS_STATS="${MEMORYWAM_COVER_BLOCKS_STATS:-${RMBENCH_TASK_ROOT}/stats/dataset_stats.json}"
  export MEMORYWAM_COVER_BLOCKS_CONTINUOUS_LATENTS="${MEMORYWAM_COVER_BLOCKS_CONTINUOUS_LATENTS:-${RMBENCH_TASK_ROOT}/temporal_fullkv_continuous_episode_stride16_v4}"
  export MEMORYWAM_COVER_BLOCKS_TEXT_CACHE="${MEMORYWAM_COVER_BLOCKS_TEXT_CACHE:-${RMBENCH_TASK_ROOT}/text_cache}"
fi
for input_path in "${RMBENCH_COVER_BLOCKS_LEROBOT}/meta/info.json" "${MEMORYWAM_COVER_BLOCKS_STATS}" "${MEMORYWAM_COVER_BLOCKS_CONTINUOUS_LATENTS}/manifest.json" "${MEMORYWAM_COVER_BLOCKS_TEXT_CACHE}"; do
  [[ -r "${input_path}" ]] || { echo "Missing read-only input: ${input_path}" >&2; return 1; }
done
mkdir -p "${HF_HOME}" "${TORCH_EXTENSIONS_DIR}" "${COMPARISON_ROOT}/runs"
