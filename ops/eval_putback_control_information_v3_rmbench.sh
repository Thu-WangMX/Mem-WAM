#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/vepfs02/output/kevin.wang/memorywam/code/fastwam_control_information_anchor_recent}"
python_bin="${PYTHON_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin/python}"
train_root_default="/mnt/vepfs02/output/kevin.wang/memorywam/train/control_information_v4_anchor_recent_k8_putback_e2e_40k_seed42"
checkpoint="${CHECKPOINT:-${train_root_default}/checkpoints/weights/step_005000.pt}"
manifest_root="${MANIFEST_ROOT:-/mnt/vepfs02/output/kevin.wang/memorywam/data/putback_control_information_planning_segments_v3}"
runtime_lock="${RUNTIME_LOCK:-/mnt/vepfs02/output/kevin.wang/memorywam/analysis/putback_control_information_selector_v3/selector_runtime/locked_selector.json}"
run_tag="${RUN_TAG:-control_information_v4_anchor_recent_rmbench4_$(date +%Y%m%d_%H%M%S)}"
log_root="${LOG_ROOT:-/mnt/vepfs02/output/kevin.wang/memorywam/eval_runs/${run_tag}}"
artifact_root="${ARTIFACT_ROOT:-${log_root}/artifacts}"
gpus_text="${GPUS:-0,1,2,3}"
indices_text="${PAIR_INDICES:-0,1,2,3}"
preflight_only="${PREFLIGHT_ONLY:-0}"

scene_seeds=(100000 200000 1300000 1400000)
policy_seeds=(1000 1002 1008 1010)
read -r -a gpus <<<"${gpus_text//,/ }"
read -r -a indices <<<"${indices_text//,/ }"

die() { echo "ERROR: $*" >&2; exit 2; }
[[ "${preflight_only}" == "0" || "${preflight_only}" == "1" ]] || die "PREFLIGHT_ONLY must be 0 or 1"
[[ -d "${repo_root}" ]] || die "repository not found: ${repo_root}"
[[ -x "${python_bin}" ]] || die "python not executable: ${python_bin}"
[[ -s "${checkpoint}" ]] || die "checkpoint not found or empty: ${checkpoint}"
[[ -s "${manifest_root}/manifest.json" ]] || die "v3 manifest missing: ${manifest_root}"
[[ -s "${runtime_lock}" ]] || die "v3 runtime lock missing: ${runtime_lock}"
[[ ${#gpus[@]} -eq ${#indices[@]} && ${#gpus[@]} -gt 0 ]] || die "GPUS and PAIR_INDICES must have equal nonzero length"
[[ $(printf '%s\n' "${gpus[@]}" | sort -u | wc -l | tr -d ' ') -eq ${#gpus[@]} ]] || die "GPU ids must be unique"
for index in "${indices[@]}"; do
  [[ "${index}" =~ ^[0-3]$ ]] || die "pair index must be 0-3: ${index}"
done

export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/vepfs02/output/kevin.wang/memorywam/model_assets/diffsynth_official}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${repo_root}/src:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8

cd "${repo_root}"
preflight_report="$("${python_bin}" - "${checkpoint}" "${manifest_root}" "${runtime_lock}" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


checkpoint = Path(sys.argv[1]).resolve()
manifest_root = Path(sys.argv[2]).resolve()
runtime_lock = Path(sys.argv[3]).resolve()
expected_runtime_sha = "c27e63084f3b579f1d04d6ca62126ca8caa8f6032ae94286cab196402c6cfb96"

match = re.fullmatch(r"step_(\d{6})\.pt", checkpoint.name)
assert match is not None, f"checkpoint must be named step_NNNNNN.pt: {checkpoint}"
checkpoint_step = int(match.group(1))
assert 5000 <= checkpoint_step <= 40000 and checkpoint_step % 5000 == 0
training_root = checkpoint.parents[2]
training_config = training_root / "config.yaml"
assert training_config.is_file(), f"training config missing: {training_config}"

cfg = OmegaConf.load(training_config)
expected_training = {
    "max_steps": 40000,
    "save_every": 5000,
    "save_steps": [],
    "seed": 42,
    "batch_size": 1,
    "gradient_accumulation_steps": 1,
    "learning_rate": 2e-4,
    "mixed_precision": "bf16",
    "native_cache_train_mode": "full",
    "data.train.minimum_history_frames": 1,
    "model.native_cache.enabled": True,
    "model.native_cache.mode": "layerwise",
    "model.native_cache.memory_tokens": 8,
    "model.native_cache.group_size": 4,
    "model.native_cache.anchor_frames": 2,
    "model.native_cache.recent_frames": 4,
    "model.native_cache.recursive": False,
}
for key, expected in expected_training.items():
    actual = OmegaConf.select(cfg, key)
    assert actual == expected, f"training config mismatch {key}: {actual!r} != {expected!r}"
configured_manifest = Path(OmegaConf.select(cfg, "data.train.surprise_manifest_path")).resolve()
assert configured_manifest == manifest_root, (
    f"checkpoint was trained with {configured_manifest}, requested {manifest_root}"
)

manifest = json.loads((manifest_root / "manifest.json").read_text())
metadata = manifest["metadata"]
assert metadata["complete"] is True
assert metadata["selector"] == "locked_segment_relative_control_information_v3"
assert metadata["runtime_lock_sha256"] == expected_runtime_sha
assert metadata["episode_count"] == 50
assert metadata["retroactive_boundary_count"] == 0
assert metadata["detector_stride"] == 4
assert metadata["replan_stride"] == 16
assert metadata["memory_tokens_per_group"] == 8

actual_runtime_sha = hashlib.sha256(runtime_lock.read_bytes()).hexdigest()
assert actual_runtime_sha == expected_runtime_sha
lock = json.loads(runtime_lock.read_text())
assert lock["schema_version"] == "putback_locked_control_information_runtime_v3"
assert lock["method"] == "segment_relative_counterfactual_control_information"
selector = lock["candidate"]["candidate"]["selector_config"]
assert selector["detector_stride"] == 4
assert selector["min_units"] == 8
assert selector["max_units"] == 24
assert selector["minimum_consecutive_evidence"] == 2

with initialize_config_dir(version_base="1.3", config_dir=str(Path.cwd() / "configs")):
    eval_cfg = compose(config_name="sim_robotwin_control_information_v3")
assert eval_cfg.model.native_cache.enabled is True
assert eval_cfg.model.native_cache.mode == "layerwise"
assert eval_cfg.model.native_cache.memory_tokens == 8
assert eval_cfg.EVALUATION.replan_steps == 16
assert eval_cfg.EVALUATION.dynamic_surprise_online is False
assert eval_cfg.EVALUATION.wrist_event_online is False
assert eval_cfg.EVALUATION.embodied_information_online is False
assert eval_cfg.EVALUATION.control_information_online is True
assert Path(eval_cfg.EVALUATION.control_information_lock).resolve() == runtime_lock

before_size = checkpoint.stat().st_size
with zipfile.ZipFile(checkpoint) as archive:
    names = archive.namelist()
    assert any(name.endswith("/data.pkl") for name in names), "checkpoint lacks data.pkl"
    assert any(name.endswith("/version") for name in names), "checkpoint lacks version"
    assert any("/data/" in name for name in names), "checkpoint lacks tensor payloads"
after_size = checkpoint.stat().st_size
assert before_size == after_size and before_size > 0, "checkpoint changed during preflight"

print("selector=segment_relative_counterfactual_control_information")
print("selector_runtime=v3")
print("selector_source=frozen_initialization_wam")
print("strict_online=true forward_alignment=true")
print("old_dynamic_surprise=false gripper_hard_trigger=false")
print("runtime_lock_sha256=" + actual_runtime_sha)
print("memory_tokens=8 detector_stride=4 replan_steps=16")
print("segment_frame_range=32..96")
print(f"checkpoint_step={checkpoint_step} portable_checkpoint=true")
PY
)"
printf '%s\n' "${preflight_report}"

selected_pairs=()
for index in "${indices[@]}"; do
  selected_pairs+=("${scene_seeds[$index]}:${policy_seeds[$index]}")
done
pairs_csv="$(IFS=,; echo "${selected_pairs[*]}")"
echo "scene_policy_pairs=${pairs_csv}"

if [[ "${preflight_only}" == "1" ]]; then
  echo "preflight_status=ok"
  exit 0
fi
[[ ! -e "${log_root}" ]] || die "refusing to overwrite evaluation root: ${log_root}"
mkdir -p "${log_root}" "${artifact_root}"

pids=()
labels=()
for position in "${!indices[@]}"; do
  index="${indices[$position]}"
  gpu="${gpus[$position]}"
  scene_seed="${scene_seeds[$index]}"
  policy_seed="${policy_seeds[$index]}"
  label="scene${scene_seed}_policy${policy_seed}"
  labels+=("${label}")
  "${python_bin}" -u experiments/robotwin/eval_robotwin_single.py \
    --config-name sim_robotwin_control_information_v3 \
    "ckpt=${checkpoint}" \
    "gpu_id=${gpu}" \
    "seed=${policy_seed}" \
    "EVALUATION.start_scene_seed=${scene_seed}" \
  "EVALUATION.task_name=put_back_block" \
  "EVALUATION.task_config=demo_clean" \
    "EVALUATION.policy_name=${POLICY_NAME:-fastwam_control_information_v4_anchor_recent_policy}" \
    "EVALUATION.eval_num_episodes=1" \
    "EVALUATION.output_dir=${log_root}/${label}" \
    "EVALUATION.artifact_root=${artifact_root}/${label}" \
    >"${log_root}/${label}.log" 2>&1 &
  pids+=("$!")
  echo "launched index=${index} scene=${scene_seed} policy=${policy_seed} gpu=${gpu} pid=$!"
done

failed=0
for position in "${!pids[@]}"; do
  if wait "${pids[$position]}"; then
    echo "${labels[$position]} status=ok"
  else
    echo "${labels[$position]} status=failed"
    failed=$((failed + 1))
  fi
done

summary="${log_root}/summary.txt"
{
  echo "checkpoint=${checkpoint}"
  echo "selector=segment_relative_counterfactual_control_information runtime=v3 initialization_wam=frozen strict_online=true"
  for label in "${labels[@]}"; do
    log="${log_root}/${label}.log"
    result="$(grep -a 'Success rate:' "${log}" | tail -n 1 || true)"
    audit="$("${python_bin}" - "${log}" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys


log = Path(sys.argv[1])
selector = None
events = []
arrivals = []
for raw in log.read_text(errors="replace").splitlines():
    for tag, target in (
        ("FASTWAM_CONTROL_INFORMATION_SELECTOR ", "selector"),
        ("FASTWAM_CONTROL_INFORMATION_DETECTOR ", "detector"),
        ("FASTWAM_CONTROL_INFORMATION_PLANNING_ARRIVAL ", "arrival"),
    ):
        if tag not in raw:
            continue
        payload = json.loads(raw.split(tag, 1)[1])
        if target == "selector":
            selector = payload
        elif target == "detector" and payload.get("event") is not None:
            events.append(payload["event"])
        elif target == "arrival":
            arrivals.append(payload)

assert selector is not None, "missing v3 selector record"
assert selector["runtime_version"] == "v3"
assert selector["source"] == "frozen_initialization_wam"
assert selector["strict_online"] is True
assert selector["forward_alignment"] is True
assert selector["uses_old_dynamic_surprise"] is False
assert selector["uses_gripper_hard_trigger"] is False
assert events, "no closed event segments"
previous_end = 0
learned = forced = 0
for event in events:
    start = int(event["group_start"])
    end = int(event["group_end"])
    confirmation = int(event["confirmation_frame"])
    reason = event["reason"]
    assert start == previous_end, (start, previous_end)
    assert end == confirmation, (end, confirmation)
    assert 32 <= end - start <= 96, (start, end)
    assert end % 4 == 0
    if reason == "forced_maximum":
        forced += 1
        assert end - start == 96
    else:
        learned += 1
        assert reason == "segment_relative_control_information"
    expected_arrival = ((confirmation + 15) // 16) * 16
    matching = [
        arrival for arrival in arrivals
        if int(arrival["frame"]) == expected_arrival
        and arrival.get("close_range") is not None
    ]
    assert matching, (confirmation, expected_arrival)
    previous_end = end
print(f"online_audit=ok events={len(events)} learned={learned} forced={forced}")
PY
)" || {
      echo "${label} online_audit=failed"
      failed=$((failed + 1))
      continue
    }
    echo "${label} ${result:-missing_result} ${audit}"
  done
  echo "failed_processes_or_audits=${failed}"
} >"${summary}"
cat "${summary}"
exit "${failed}"
