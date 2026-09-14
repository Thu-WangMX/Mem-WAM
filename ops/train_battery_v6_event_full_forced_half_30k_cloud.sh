#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/vepfs02/output/kevin.wang/memorywam/code/fastwam_event_conditioned_rate}"
manifest_root="${MANIFEST_ROOT:-/mnt/vepfs02/output/kevin.wang/memorywam/data/battery_control_information_planning_segments_v3_causal_online}"
preflight_only="${PREFLIGHT_ONLY:-0}"

prepare_script="${repo_root}/ops/prepare_battery_control_information_v6_8gpu.sh"
train_script="${repo_root}/ops/train_battery_control_information_event_rate_30k.sh"

[[ -x "${prepare_script}" ]] || { echo "missing prepare script: ${prepare_script}" >&2; exit 2; }
[[ -x "${train_script}" ]] || { echo "missing train script: ${train_script}" >&2; exit 2; }

if [[ "${preflight_only}" == "1" ]]; then
  PREFLIGHT_ONLY=1 MANIFEST_ROOT="${manifest_root}" bash "${prepare_script}"
  if [[ -s "${manifest_root}/manifest.json" ]]; then
    PREFLIGHT_ONLY=1 MANIFEST_ROOT="${manifest_root}" bash "${train_script}"
  else
    echo "training_preflight=pending_battery_manifest_generation"
  fi
  echo "cloud_entry_preflight_status=ok"
  exit 0
fi

if [[ ! -s "${manifest_root}/manifest.json" ]]; then
  PREFLIGHT_ONLY=0 MANIFEST_ROOT="${manifest_root}" bash "${prepare_script}"
fi

exec env PREFLIGHT_ONLY=0 MANIFEST_ROOT="${manifest_root}" bash "${train_script}"
