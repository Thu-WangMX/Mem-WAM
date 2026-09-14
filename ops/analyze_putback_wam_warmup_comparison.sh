#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_wam_embedding_surprise}"
bank_root="${BANK_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/analysis/putback_init_wam_embedding_bank_v1}"
dataset_root="${DATASET_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/data/rmbench_lerobot_v21_rgb_v3/put_back_block}"
output_dir="${OUTPUT_DIR:-/mnt/vepfs02/output/kevin_wang/memorywam/analysis/putback_wam_warmup_comparison_v1}"
runtime_bin="${RUNTIME_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin}"
preflight_only="${PREFLIGHT_ONLY:-0}"

die() { echo "ERROR: $*" >&2; exit 2; }
[[ "$preflight_only" == 0 || "$preflight_only" == 1 ]] || die "PREFLIGHT_ONLY must be 0 or 1"
[[ -x "$runtime_bin/python" ]] || die "python runtime missing"
[[ -f "$repo_root/scripts/analyze_putback_wam_warmup_comparison.py" ]] || die "comparison analyzer missing"
[[ -f "$repo_root/scripts/render_putback_wam_warmup_comparison.py" ]] || die "comparison renderer missing"
[[ -s "$bank_root/bank_manifest.json" ]] || die "complete feature-bank manifest missing"
[[ ! -e "$output_dir" ]] || die "refusing to overwrite warmup comparison"
for episode in $(seq 40 49); do
  [[ -s "$dataset_root/data/chunk-000/episode_$(printf '%06d' "$episode").parquet" ]] || die "held-out parquet missing: episode $episode"
  for camera in observation.images.cam_high observation.images.cam_left_wrist observation.images.cam_right_wrist; do
    [[ -s "$dataset_root/videos/chunk-000/$camera/episode_$(printf '%06d' "$episode").mp4" ]] || die "held-out camera video missing: episode $episode $camera"
  done
done

echo "bank_root=$bank_root"
echo "output_dir=$output_dir"
echo "calibration_episodes=0-39"
echo "heldout_episodes=40-49"
echo "policy_checkpoint=null"
if [[ "$preflight_only" == 1 ]]; then
  echo "preflight_status=ok"
  exit 0
fi

command -v ffmpeg >/dev/null || die "ffmpeg unavailable"
command -v ffprobe >/dev/null || die "ffprobe unavailable"
export PYTHONPATH="$repo_root/src:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_root"
"$runtime_bin/python" scripts/analyze_putback_wam_warmup_comparison.py --bank "$bank_root" --dataset-root "$dataset_root" --output "$output_dir"
mkdir -p "$output_dir/videos"
for episode in $(seq 40 49); do
  video="$output_dir/videos/episode_$(printf '%03d' "$episode")_warmup_comparison.mp4"
  "$runtime_bin/python" scripts/render_putback_wam_warmup_comparison.py --comparison-root "$output_dir" --dataset-root "$dataset_root" --episode "$episode" --output "$video"
  codec="$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=nw=1:nk=1 "$video")"
  duration="$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$video")"
  [[ "$codec" == h264 ]] || die "episode $episode codec is $codec"
  "$runtime_bin/python" - "$duration" <<'PY'
import sys
if float(sys.argv[1]) <= 0:
    raise SystemExit("duration must be positive")
PY
  echo "video_ok episode=$episode codec=$codec duration=$duration path=$video"
done
echo "warmup_comparison_status=complete"
