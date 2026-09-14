#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_dynamic_multiframe_selection}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/vepfs02/output/kevin_wang/memorywam/analysis/putback_four_phase_latents_v1}
LEROBOT_ROOT=${LEROBOT_ROOT:?LEROBOT_ROOT is required}
VAE_PATH=${VAE_PATH:?VAE_PATH is required}
RUNTIME_BIN=${RUNTIME_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
MIN_FREE_GIB=${MIN_FREE_GIB:-100}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}

fail() {
  printf '%s\n' "$1" >&2
  exit 1
}

IFS=',' read -r -a gpu_array <<< "$CUDA_VISIBLE_DEVICES"
[ "${#gpu_array[@]}" -eq 8 ] || fail "CUDA_VISIBLE_DEVICES must contain exactly eight GPUs"
[ "$(printf '%s\n' "${gpu_array[@]}" | sort -u | wc -l | tr -d ' ')" -eq 8 ] || fail "CUDA_VISIBLE_DEVICES must contain exactly eight unique GPUs"
[ -d "$REPO_ROOT" ] || fail "missing repository: $REPO_ROOT"
[ -f "$REPO_ROOT/scripts/build_putback_four_phase_latents.py" ] || fail "missing four-phase builder"
[ -f "$LEROBOT_ROOT/meta/episodes.jsonl" ] || fail "missing dataset metadata"
[ -f "$VAE_PATH" ] || fail "missing VAE_PATH: $VAE_PATH"
[ -x "$RUNTIME_BIN/python" ] || fail "missing runtime python"
[ -x "$RUNTIME_BIN/torchrun" ] || fail "missing runtime torchrun"

if [ -f "$OUTPUT_DIR/manifest.json" ] && "$RUNTIME_BIN/python" - "$OUTPUT_DIR/manifest.json" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get("complete") is True else 1)
PY
then
  fail "refusing completed output root: $OUTPUT_DIR"
fi

output_parent=$(dirname "$OUTPUT_DIR")
mkdir -p "$output_parent"
free_kib=$(df -Pk "$output_parent" | awk 'NR==2 {print $4}')
required_kib=$((MIN_FREE_GIB * 1024 * 1024))
[ "$free_kib" -ge "$required_kib" ] || fail "insufficient free space: require ${MIN_FREE_GIB} GiB"

printf 'preflight_status=ok\n'
printf 'gpus=%s\n' "$CUDA_VISIBLE_DEVICES"
printf 'phase_offsets=0,4,8,12\n'
printf 'video_expert_frame_stride=16\n'
printf 'policy_checkpoint=null\n'

if [ "$PREFLIGHT_ONLY" = "1" ]; then
  exit 0
fi

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT"
cd "$REPO_ROOT"
exec "$RUNTIME_BIN/torchrun" --standalone --nproc_per_node=8 \
  scripts/build_putback_four_phase_latents.py \
  --lerobot-root "$LEROBOT_ROOT" \
  --vae-path "$VAE_PATH" \
  --episodes 0-49 \
  --output "$OUTPUT_DIR"
