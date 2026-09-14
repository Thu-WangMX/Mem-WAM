#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_memory_fullkv}"
python_bin="${PYTHON_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin/python}"
dataset="${RMBENCH_PUTBACK_LEROBOT:-/mnt/vepfs02/output/kevin_wang/memorywam/data/rmbench_lerobot_v21_rgb_v3/put_back_block}"
dataset_manifest="${RMBENCH_DATASET_MANIFEST:-$(dirname "${dataset}")/_build_manifest.json}"
rgb_report="${RMBENCH_RGB_REPORT:-$(dirname "${dataset}")/_build_logs/rgb_contract.json}"
cache="${FASTWAM_FULL_KV_LATENTS:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_temporal_fullkv_rgb_v3_stride16}"
stats_root="${FASTWAM_STATS_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_stats_rgb_v3}"
text_cache="${FASTWAM_PUTBACK_TEXT_CACHE:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_text_cache_rgb_v3}"
action_init="${FASTWAM_ACTION_DIT_INIT:-/mnt/vepfs02/output/kevin_wang/memorywam/model_assets/fastwam_fullkv_official_init/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"

case "${dataset}" in
  */rmbench_lerobot_v21/put_back_block)
    echo "Refusing legacy red/blue-swapped RMBench dataset: ${dataset}" >&2
    exit 2
    ;;
esac

for required in \
  "${dataset}/meta/info.json" \
  "${dataset}/meta/episodes.jsonl" \
  "${dataset_manifest}" \
  "${rgb_report}" \
  "${action_init}"; do
  if [[ ! -s "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 2
  fi
done

"${python_bin}" - "${dataset_manifest}" "${rgb_report}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
if payload.get("color_contract") != "simulator_rgb_preserved":
    raise SystemExit(f"{path}: invalid color contract: {payload.get('color_contract')!r}")
if int(payload.get("episodes_per_task", -1)) != 50:
    raise SystemExit(f"{path}: expected 50 episodes")
print(f"validated dataset manifest: {path}")
rgb_path = Path(sys.argv[2])
rgb = json.loads(rgb_path.read_text())
if rgb.get("passed") is not True:
    raise SystemExit(f"{rgb_path}: RGB validation did not pass")
print(f"validated RGB report: {rgb_path}")
PY

export RMBENCH_PUTBACK_LEROBOT="${dataset}"
export FASTWAM_FULL_KV_LATENTS="${cache}"
export FASTWAM_STATS_ROOT="${stats_root}"
export FASTWAM_FULL_KV_OUTPUT="${stats_root}"
export FASTWAM_PUTBACK_STATS="${stats_root}/dataset_stats.json"
export FASTWAM_PUTBACK_TEXT_CACHE="${text_cache}"
export FASTWAM_ACTION_DIT_INIT="${action_init}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/vepfs02/datasets/kevin_wang/code/projects/FastWAM/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUTF8=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

mkdir -p "${stats_root}" "${text_cache}"
cd "${repo_root}"

if [[ ! -s "${stats_root}/dataset_stats.json" ]]; then
  "${python_bin}" scripts/compute_lerobot_stats.py \
    task=rmbench_putback_memorywam_fullkv_5k \
    "output_dir=${stats_root}"
fi

if [[ ! -s "${cache}/manifest.json" ]]; then
  "${python_bin}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node=8 \
    scripts/precompute_full_kv_observation_latents.py \
    --lerobot-root "${dataset}" \
    --output "${cache}" \
    --replan-stride 16 \
    --temporal-subframes 4 \
    --batch-size 8
fi

"${python_bin}" - "${cache}/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
metadata = json.loads(path.read_text())["metadata"]
expected = {
    "schema_version": "fastwam_full_kv_temporal_observation_latents_v3",
    "complete": True,
    "episode_count": 50,
    "replan_stride": 16,
    "temporal_subframes": 4,
    "temporal_subframe_stride": 4,
    "mosaic": "wrists_top_head_bottom_384x320",
    "color_contract": "simulator_rgb_preserved",
}
mismatch = {
    key: (value, metadata.get(key))
    for key, value in expected.items()
    if metadata.get(key) != value
}
if mismatch:
    raise SystemExit(f"{path}: incompatible cache: {mismatch}")
print(f"validated full-KV cache manifest: {path}")
PY

"${python_bin}" scripts/precompute_text_embeds.py \
  task=rmbench_putback_memorywam_fullkv_5k

echo "MemoryWAM-aligned full-KV Put Back assets are ready."
echo "dataset=${dataset}"
echo "stats=${stats_root}/dataset_stats.json"
echo "cache=${cache}"
echo "text_cache=${text_cache}"
