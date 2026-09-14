#!/usr/bin/env bash
set -euo pipefail

# Strict 100-episode RM-Bench PutBack evaluation for the v6 event-conditioned
# memory checkpoint.  This is intentionally separate from the older fixed-K
# and residual-pyramid launchers.

die() {
  echo "ERROR: $*" >&2
  exit 2
}

memorywam_root="${MEMORYWAM_ROOT:-/mnt/vepfs02/output/kevin.wang/memorywam}"
repo_root="${REPO_ROOT:-${memorywam_root}/code/fastwam_event_conditioned_rate}"
runtime_bin="${RUNTIME_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin}"
python_bin="${PYTHON_BIN:-${runtime_bin}/python}"

train_root="${TRAIN_ROOT:-/mnt/vepfs01/output/kevin.wang/memorywam/train/control_information_v6_event_full_forced_half_putback_e2e_30k_seed42}"
checkpoint="${CHECKPOINT:-${train_root}/checkpoints/weights/step_030000.pt}"
dataset_stats="${DATASET_STATS:-${train_root}/dataset_stats.json}"
train_config="${TRAIN_CONFIG:-${train_root}/config.yaml}"
manifest_root="${MANIFEST_ROOT:-${memorywam_root}/data/putback_control_information_planning_segments_v3}"
runtime_lock="${RUNTIME_LOCK:-${memorywam_root}/analysis/putback_control_information_selector_v3/selector_runtime/locked_selector.json}"
eval_config_name="${EVAL_CONFIG_NAME:-sim_robotwin_control_information_rate_event}"
policy_name="${POLICY_NAME:-fastwam_rate_compare_policy}"

episodes="${EVAL_NUM_EPISODES:-100}"
policy_seed="${POLICY_SEED:-0}"
scene_seed_start="${SCENE_SEED_START:-7000001}"
scene_seed_stride="${SCENE_SEED_STRIDE:-100000}"
gpus_text="${GPUS:-0,1,2,3,4,5,6,7}"
task_config="${TASK_CONFIG:-demo_clean}"
preflight_only="${PREFLIGHT_ONLY:-0}"
require_idle_gpus="${REQUIRE_IDLE_GPUS:-0}"
run_tag="${RUN_TAG:-v6_event_full_forced_half_30k_putback100_policy${policy_seed}_$(date +%Y%m%d_%H%M%S)}"
log_root="${LOG_ROOT:-${memorywam_root}/eval_runs/${run_tag}}"
artifact_root="${ARTIFACT_ROOT:-${log_root}/artifacts}"

[[ -d "${repo_root}" ]] || die "repository not found: ${repo_root}"
[[ -x "${python_bin}" ]] || die "python is not executable: ${python_bin}"
[[ -s "${checkpoint}" ]] || die "checkpoint is missing or empty: ${checkpoint}"
[[ -s "${dataset_stats}" ]] || die "dataset stats are missing or empty: ${dataset_stats}"
[[ -s "${train_config}" ]] || die "training config is missing or empty: ${train_config}"
[[ -s "${manifest_root}/manifest.json" ]] || die "selector manifest is missing: ${manifest_root}/manifest.json"
[[ -s "${runtime_lock}" ]] || die "selector runtime lock is missing: ${runtime_lock}"
[[ "${episodes}" =~ ^[1-9][0-9]*$ ]] || die "EVAL_NUM_EPISODES must be positive"
[[ "${policy_seed}" =~ ^[0-9]+$ ]] || die "POLICY_SEED must be non-negative"
[[ "${scene_seed_start}" =~ ^[0-9]+$ ]] || die "SCENE_SEED_START must be non-negative"
[[ "${scene_seed_stride}" =~ ^[1-9][0-9]*$ ]] || die "SCENE_SEED_STRIDE must be positive"
[[ "${scene_seed_stride}" -gt "${episodes}" ]] || die "SCENE_SEED_STRIDE must exceed EVAL_NUM_EPISODES"
[[ "${preflight_only}" == 0 || "${preflight_only}" == 1 ]] || die "PREFLIGHT_ONLY must be 0 or 1"
[[ "${require_idle_gpus}" == 0 || "${require_idle_gpus}" == 1 ]] || die "REQUIRE_IDLE_GPUS must be 0 or 1"

read -r -a gpus <<<"${gpus_text//,/ }"
[[ ${#gpus[@]} -gt 0 ]] || die "GPUS must contain at least one GPU"
declare -A seen_gpus=()
for gpu in "${gpus[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || die "GPU identifier must be numeric: ${gpu}"
  [[ -z "${seen_gpus[$gpu]:-}" ]] || die "GPU index is repeated: ${gpu}"
  seen_gpus[$gpu]=1
done

export PATH="${runtime_bin}:${PATH}"
export PYTHONPATH="${repo_root}/src:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUTF8=1
export PYTHONUNBUFFERED=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${memorywam_root}/model_assets/diffsynth_official}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export FASTWAM_ADVANCE_POLICY_SEED=false

if [[ "${require_idle_gpus}" == 1 ]]; then
  command -v nvidia-smi >/dev/null 2>&1 || die "REQUIRE_IDLE_GPUS=1 needs nvidia-smi"
  while IFS=, read -r index used _; do
    index="${index//[[:space:]]/}"
    used="${used//[[:space:]]/}"
    used="${used//MiB/}"
    for gpu in "${gpus[@]}"; do
      if [[ "${index}" == "${gpu}" && "${used}" =~ ^[0-9]+$ && "${used}" -gt 1024 ]]; then
        die "selected GPU ${gpu} is not idle (${used} MiB used); set REQUIRE_IDLE_GPUS=0 to share"
      fi
    done
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
fi

cd "${repo_root}"

# Validate the training/evaluation contract without loading the 12-GB tensor
# payload into RAM.  The checkpoint is checked as a portable torch archive;
# all rate/allocation fields are checked against the v6 training config and
# the rate-event inference config.
preflight_report="$(${python_bin} - "${checkpoint}" "${train_config}" "${manifest_root}" "${runtime_lock}" "${repo_root}" "${eval_config_name}" "${policy_name}" <<'PY'
from __future__ import annotations

import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


checkpoint = Path(sys.argv[1]).resolve()
train_config = Path(sys.argv[2]).resolve()
manifest_root = Path(sys.argv[3]).resolve()
runtime_lock = Path(sys.argv[4]).resolve()
repo_root = Path(sys.argv[5]).resolve()
eval_config_name = sys.argv[6]
policy_name = sys.argv[7]

match = re.fullmatch(r"step_(\d{6})\.pt", checkpoint.name)
assert match is not None, f"checkpoint must be named step_NNNNNN.pt: {checkpoint}"
checkpoint_step = int(match.group(1))
assert checkpoint_step == 30000, f"expected v6 step_030000.pt, got step_{checkpoint_step:06d}"
assert checkpoint.stat().st_size > 0

cfg = OmegaConf.load(train_config)
expected_training = {
    "max_steps": 30000,
    "save_every": 10000,
    "save_steps": [5000],
    "seed": 42,
    "batch_size": 1,
    "gradient_accumulation_steps": 1,
    "learning_rate": 2.0e-4,
    "mixed_precision": "bf16",
    "native_cache_train_mode": "full",
    "memory_allocation_rule": "event_full_forced_half",
    "data.train.minimum_history_frames": 1,
    "model.native_cache.enabled": True,
    "model.native_cache.mode": "layerwise",
    "model.native_cache.memory_tokens": 48,
    "model.native_cache.dynamic_tokens_per_frame": 8,
    "model.native_cache.allocation_mode": "event_full_forced_half",
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
assert metadata["runtime_lock_sha256"] == "c27e63084f3b579f1d04d6ca62126ca8caa8f6032ae94286cab196402c6cfb96"
assert metadata["episode_count"] == 50
assert metadata["retroactive_boundary_count"] == 0
assert metadata["detector_stride"] == 4
assert metadata["replan_stride"] == 16
assert metadata["memory_tokens_per_group"] == 8

runtime_sha = hashlib.sha256(runtime_lock.read_bytes()).hexdigest()
assert runtime_sha == "c27e63084f3b579f1d04d6ca62126ca8caa8f6032ae94286cab196402c6cfb96"
lock = json.loads(runtime_lock.read_text())
assert lock["schema_version"] == "putback_locked_control_information_runtime_v3"
assert lock["method"] == "segment_relative_counterfactual_control_information"
selector = lock["candidate"]["candidate"]["selector_config"]
assert selector["detector_stride"] == 4
assert selector["min_units"] == 8
assert selector["max_units"] == 24
assert selector["minimum_consecutive_evidence"] == 2

with initialize_config_dir(version_base="1.3", config_dir=str(repo_root / "configs")):
    eval_cfg = compose(config_name=eval_config_name)
assert eval_cfg.model.native_cache.enabled is True
assert eval_cfg.model.native_cache.mode == "layerwise"
assert eval_cfg.model.native_cache.memory_tokens == 48
assert eval_cfg.model.native_cache.dynamic_tokens_per_frame == 8
assert eval_cfg.model.native_cache.allocation_mode == "event_full_forced_half"
assert eval_cfg.model.native_cache.group_size == 4
assert eval_cfg.model.native_cache.anchor_frames == 2
assert eval_cfg.model.native_cache.recent_frames == 4
assert eval_cfg.model.native_cache.recursive is False
assert eval_cfg.EVALUATION.replan_steps == 16
assert eval_cfg.EVALUATION.num_inference_steps == 50
assert eval_cfg.EVALUATION.dynamic_surprise_online is False
assert eval_cfg.EVALUATION.wrist_event_online is False
assert eval_cfg.EVALUATION.embodied_information_online is False
assert eval_cfg.EVALUATION.control_information_online is True
assert Path(eval_cfg.EVALUATION.control_information_lock).resolve() == runtime_lock

with zipfile.ZipFile(checkpoint) as archive:
    names = archive.namelist()
    assert any(name.endswith("/data.pkl") for name in names), "checkpoint lacks data.pkl"
    assert any(name.endswith("/version") for name in names), "checkpoint lacks version"
    assert any("/data/" in name for name in names), "checkpoint lacks tensor payloads"

policy_source = (repo_root / "experiments" / "robotwin" / "fastwam_policy").resolve()
robotwin_root = Path(str(eval_cfg.EVALUATION.robotwin_root)).expanduser().resolve()
policy_link = robotwin_root / "policy" / policy_name
if policy_link.exists() or policy_link.is_symlink():
    assert policy_link.resolve() == policy_source, (
        f"policy alias {policy_link} points to {policy_link.resolve()}, expected {policy_source}"
    )

print("selector=segment_relative_counterfactual_control_information")
print("selector_runtime=v3")
print("selector_source=frozen_initialization_wam")
print("strict_online=true forward_alignment=true")
print("old_dynamic_surprise=false gripper_hard_trigger=false")
print("runtime_lock_sha256=" + runtime_sha)
print("memory_tokens=48 dynamic_tokens_per_frame=8 allocation_mode=event_full_forced_half")
print("group_size=4 anchor_frames=2 recent_frames=4 recursive=false replan_steps=16")
print(f"checkpoint_step={checkpoint_step} portable_checkpoint=true")
print(f"policy_alias={policy_name}")
PY
)"
printf '%s\n' "${preflight_report}"

if [[ "${preflight_only}" == 1 ]]; then
  echo "preflight_status=ok episodes=${episodes} workers=${#gpus[@]}"
  exit 0
fi

[[ ! -e "${log_root}" ]] || die "LOG_ROOT already exists: ${log_root}"
mkdir -p "${log_root}" "${artifact_root}"

worker_count=${#gpus[@]}
base_count=$((episodes / worker_count))
extra_count=$((episodes % worker_count))
worker_rows=()
for index in "${!gpus[@]}"; do
  count="${base_count}"
  if [[ "${index}" -lt "${extra_count}" ]]; then count=$((count + 1)); fi
  [[ "${count}" -gt 0 ]] || continue
  gpu="${gpus[$index]}"
  worker_scene_start=$((scene_seed_start + index * scene_seed_stride))
  worker_tag="${run_tag}_worker$(printf '%02d' "${index}")_gpu${gpu}_scene${worker_scene_start}"
  worker_rows+=("${index}|${gpu}|${count}|${worker_scene_start}|${worker_tag}")
done

printf 'worker\tgpu\tepisodes\tstart_scene_seed\tworker_tag\tlauncher_log\n' >"${log_root}/workers.tsv"
pids=()
worker_ids=()
for row in "${worker_rows[@]}"; do
  IFS='|' read -r index gpu count worker_scene_start worker_tag <<<"${row}"
  launcher_log="${log_root}/worker$(printf '%02d' "${index}").launcher.log"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${index}" "${gpu}" "${count}" "${worker_scene_start}" "${worker_tag}" "${launcher_log}" \
    >>"${log_root}/workers.tsv"
  "${python_bin}" -u experiments/robotwin/eval_robotwin_single.py \
    --config-name "${eval_config_name}" \
    "ckpt=${checkpoint}" \
    "gpu_id=${gpu}" \
    "seed=${policy_seed}" \
    "EVALUATION.start_scene_seed=${worker_scene_start}" \
    "EVALUATION.task_name=put_back_block" \
    "EVALUATION.task_config=${task_config}" \
    "EVALUATION.policy_name=${policy_name}" \
    "EVALUATION.eval_num_episodes=${count}" \
    "EVALUATION.output_dir=${log_root}/${worker_tag}" \
    "EVALUATION.artifact_root=${artifact_root}" \
    "EVALUATION.dataset_stats_path=${dataset_stats}" \
    "EVALUATION.skip_get_obs_within_replan=false" \
    >"${launcher_log}" 2>&1 &
  pid=$!
  pids+=("${pid}")
  worker_ids+=("${index}")
  echo "launched worker=${index} gpu=${gpu} episodes=${count} scene_start=${worker_scene_start} pid=${pid}"
done

failed_workers=0
for position in "${!pids[@]}"; do
  if wait "${pids[$position]}"; then
    echo "worker=${worker_ids[$position]} status=ok"
  else
    echo "worker=${worker_ids[$position]} status=failed"
    failed_workers=$((failed_workers + 1))
  fi
done

"${python_bin}" - "${log_root}" "${artifact_root}" "${checkpoint}" "${episodes}" "${policy_name}" <<'PY'
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


log_root = Path(sys.argv[1])
artifact_root = Path(sys.argv[2])
checkpoint = Path(sys.argv[3]).resolve()
expected = int(sys.argv[4])
policy_name = sys.argv[5]
ansi = re.compile(r"\x1b\[[0-9;]*m")
error_patterns = [
    r"Traceback",
    r"OutOfMemory",
    r"CUDA out of memory",
    r"NCCL.*ERROR",
    r"Cannot detect model type",
]

workers = []
for log in sorted(log_root.glob("worker*.launcher.log")):
    text = log.read_text(errors="replace")
    clean = ansi.sub("", text)
    rates = re.findall(r"Success rate:\s*(\d+)/(\d+)", clean)
    selector = None
    events = []
    arrivals = []
    for raw in text.splitlines():
        for tag, kind in (
            ("FASTWAM_CONTROL_INFORMATION_SELECTOR ", "selector"),
            ("FASTWAM_CONTROL_INFORMATION_DETECTOR ", "detector"),
            ("FASTWAM_CONTROL_INFORMATION_PLANNING_ARRIVAL ", "arrival"),
        ):
            if tag not in raw:
                continue
            try:
                payload = json.loads(raw.split(tag, 1)[1])
            except Exception:
                continue
            if kind == "selector":
                selector = payload
            elif kind == "detector" and payload.get("event") is not None:
                events.append(payload["event"])
            elif kind == "arrival":
                arrivals.append(payload)

    errors = [pat for pat in error_patterns if re.search(pat, text, re.I)]
    selector_ok = bool(
        selector
        and selector.get("runtime_version") == "v3"
        and selector.get("source") == "frozen_initialization_wam"
        and selector.get("strict_online") is True
        and selector.get("forward_alignment") is True
        and selector.get("uses_old_dynamic_surprise") is False
        and selector.get("uses_gripper_hard_trigger") is False
    )
    event_ok = bool(events)
    for event in events:
        start = int(event["group_start"])
        end = int(event["group_end"])
        confirmation = int(event["confirmation_frame"])
        reason = event["reason"]
        event_ok &= end == confirmation and end >= start and end % 4 == 0
        event_ok &= reason in {"segment_relative_control_information", "forced_maximum"}

    close_arrivals = [a for a in arrivals if a.get("close_range") is not None]
    allocation_ok = bool(close_arrivals)
    for arrival in close_arrivals:
        left, right = map(int, arrival["close_range"])
        span = right - left
        reason = arrival.get("close_reason")
        tokens = arrival.get("close_token_count")
        allocation_ok &= 2 <= span <= 8
        allocation_ok &= arrival.get("allocation_mode") == "event_full_forced_half"
        allocation_ok &= reason in {"segment_relative_control_information", "forced_maximum"}
        allocation_ok &= isinstance(tokens, int) and 0 < tokens <= 48 and tokens % 8 == 0
        if reason == "segment_relative_control_information":
            allocation_ok &= tokens == span * 8
        elif reason == "forced_maximum":
            allocation_ok &= tokens == ((span + 1) // 2) * 8

    result = rates[-1] if rates else None
    video_root = artifact_root / checkpoint.stem
    # The evaluator's run_ts is the full worker tag, while the wrapper log is
    # intentionally shortened to workerNN.launcher.log.  Resolve the full tag
    # by worker index instead of assuming the two names are identical.
    worker_match = re.search(r"worker(\d+)", log.name)
    worker_video_root = None
    if worker_match and video_root.exists():
        candidates = sorted(video_root.glob(f"*worker{worker_match.group(1)}_*"))
        if len(candidates) == 1:
            worker_video_root = candidates[0]
    videos = len(list(worker_video_root.rglob("*.mp4"))) if worker_video_root is not None and worker_video_root.exists() else 0
    videos = len(list(worker_video_root.rglob("*.mp4"))) if worker_video_root.exists() else 0
    workers.append({
        "log": str(log),
        "result": None if result is None else {"successes": int(result[0]), "episodes": int(result[1])},
        "events": len(events),
        "close_arrivals": len(close_arrivals),
        "selector_audit": selector_ok,
        "event_audit": event_ok,
        "allocation_audit": allocation_ok,
        "videos": videos,
        "errors": errors,
    })

observed = sum((w["result"] or {}).get("episodes", 0) for w in workers)
successes = sum((w["result"] or {}).get("successes", 0) for w in workers)
valid = (
    len(workers) > 0
    and observed == expected
    and all(w["result"] is not None for w in workers)
    and all(w["selector_audit"] and w["event_audit"] and w["allocation_audit"] for w in workers)
    and all(not w["errors"] for w in workers)
    and all(w["videos"] == w["result"]["episodes"] for w in workers if w["result"] is not None)
)
summary = {
    "valid": bool(valid),
    "checkpoint": str(checkpoint),
    "policy_name": policy_name,
    "allocation_mode": "event_full_forced_half",
    "expected_episodes": expected,
    "observed_episodes": observed,
    "successes": successes,
    "success_rate": (successes / observed) if observed else 0.0,
    "workers": workers,
}
(log_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
with (log_root / "summary.txt").open("w") as out:
    out.write(f"checkpoint={checkpoint}\n")
    out.write("selector=segment_relative_control_information_v3 frozen_initialization_wam strict_online=true forward_alignment=true\n")
    out.write("allocation_mode=event_full_forced_half memory_tokens=48 dynamic_tokens_per_frame=8\n")
    for worker in workers:
        out.write(json.dumps(worker, sort_keys=True) + "\n")
    out.write(f"success_rate={successes}/{observed}\n")
    out.write(f"valid={str(bool(valid)).lower()}\n")
print(f"SUCCESS_RATE {successes}/{observed} = {(successes / observed if observed else 0.0):.4%}")
print(f"SUMMARY {log_root / 'summary.json'}")
if not valid:
    raise SystemExit("batch audit failed; inspect summary.txt and worker logs")
PY

[[ "${failed_workers}" -eq 0 ]] || die "${failed_workers} evaluation workers failed; inspect ${log_root}"
echo "complete summary=${log_root}/summary.json videos=${artifact_root}"
