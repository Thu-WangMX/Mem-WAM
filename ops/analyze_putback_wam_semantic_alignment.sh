#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_wam_embedding_surprise}"
analysis_root="${ANALYSIS_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/analysis/putback_wam_embedding_surprise_init_v1}"
dataset_root="${DATASET_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/data/rmbench_lerobot_v21_rgb_v3/put_back_block}"
runtime_bin="${RUNTIME_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin}"
preflight_only="${PREFLIGHT_ONLY:-0}"

die() { echo "ERROR: $*" >&2; exit 2; }
[[ "$preflight_only" == 0 || "$preflight_only" == 1 ]] || die "PREFLIGHT_ONLY must be 0 or 1"
[[ -x "$runtime_bin/python" ]] || die "python runtime missing: $runtime_bin/python"
[[ -f "$repo_root/scripts/analyze_putback_wam_semantic_alignment.py" ]] || die "analysis CLI missing"
[[ -f "$repo_root/scripts/render_putback_wam_semantic_alignment.py" ]] || die "renderer missing"
for episode in 40 41; do
  [[ -s "$analysis_root/episodes/episode_$(printf '%03d' "$episode").json" ]] || die "source trace missing for episode $episode"
  [[ -s "$dataset_root/data/chunk-000/episode_$(printf '%06d' "$episode").parquet" ]] || die "parquet missing for episode $episode"
done
[[ ! -e "$analysis_root/semantic_alignment.json" ]] || die "refusing to overwrite semantic report"

echo "analysis_root=$analysis_root"
echo "episodes=40,41"
echo "policy_checkpoint=null"
if [[ "$preflight_only" == 1 ]]; then
  echo "preflight_status=ok"
  exit 0
fi

command -v ffmpeg >/dev/null || die "ffmpeg unavailable"
command -v ffprobe >/dev/null || die "ffprobe unavailable"
export PYTHONPATH="$repo_root/src:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_root"
"$runtime_bin/python" scripts/analyze_putback_wam_semantic_alignment.py --analysis-root "$analysis_root" --dataset-root "$dataset_root" --episodes 40,41 --stride 16 --tolerance 1.0 --warmup-end 8
mkdir -p "$analysis_root/semantic_alignment_videos"
for episode in 40 41; do
  output="$analysis_root/semantic_alignment_videos/episode_$(printf '%03d' "$episode")_semantic_alignment.mp4"
  "$runtime_bin/python" scripts/render_putback_wam_semantic_alignment.py --analysis-root "$analysis_root" --dataset-root "$dataset_root" --episode "$episode" --output "$output"
  codec="$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=nw=1:nk=1 "$output")"
  duration="$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$output")"
  [[ "$codec" == h264 ]] || die "episode $episode codec is $codec"
  "$runtime_bin/python" - "$duration" <<'PY'
import sys
if float(sys.argv[1]) <= 0:
    raise SystemExit("duration must be positive")
PY
  echo "video_ok episode=$episode codec=$codec duration=$duration path=$output"
done
echo "semantic_alignment_status=complete"
