#!/usr/bin/env bash
set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 2; }

memorywam_root="${MEMORYWAM_ROOT:-/mnt/vepfs02/output/kevin.wang/memorywam}"
repo_root="${REPO_ROOT:-${memorywam_root}/code/fastwam_event_conditioned_rate}"
runtime_bin="${RUNTIME_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin}"
python_bin="${PYTHON_BIN:-${runtime_bin}/python}"
train_root="${TRAIN_ROOT:-/mnt/vepfs01/output/kevin.wang/memorywam/train/control_information_v6_event_full_forced_half_battery_e2e_50k_seed42_resume30k_weightonly}"
step="${STEP:-}"
gpus_text="${GPUS:-0,1,2,3}"
scene_seed_start="${SCENE_SEED_START:-6207001}"
scene_seed_stride="${SCENE_SEED_STRIDE:-10}"
policy_seed="${POLICY_SEED:-0}"
policy_name="${POLICY_NAME:-fastwam_event_conditioned_rate_v6_policy_30k}"
preflight_only="${PREFLIGHT_ONLY:-0}"

case "${step}" in
  40000|45000|50000) ;;
  *) die "STEP must be one of 40000, 45000, 50000" ;;
esac

step_tag="step_$(printf '%06d' "${step}")"
checkpoint="${CHECKPOINT:-${train_root}/checkpoints/weights/${step_tag}.pt}"
dataset_stats="${DATASET_STATS:-${train_root}/dataset_stats.json}"
train_config="${TRAIN_CONFIG:-${train_root}/config.yaml}"
selector_normalization="${CONTROL_INFORMATION_NORMALIZATION:-${memorywam_root}/analysis/putback_init_per_episode_noise_2ep_20260813_r3/.work/rank_0/dataset_stats.json}"
runtime_lock="${RUNTIME_LOCK:-${memorywam_root}/analysis/putback_control_information_selector_v3/selector_runtime/locked_selector.json}"
eval_config_name="${EVAL_CONFIG_NAME:-sim_robotwin_control_information_rate_event}"
run_tag="${RUN_TAG:-battery_v6_resume30k_${step_tag}_same4_$(date +%Y%m%d_%H%M%S)}"
log_root="${LOG_ROOT:-${memorywam_root}/eval_runs/${run_tag}}"
artifact_root="${ARTIFACT_ROOT:-${log_root}/artifacts}"

[[ -d "${repo_root}" ]] || die "repository missing: ${repo_root}"
[[ -x "${python_bin}" ]] || die "python missing: ${python_bin}"
for required in "${checkpoint}" "${dataset_stats}" "${train_config}" "${selector_normalization}" "${runtime_lock}"; do
  [[ -s "${required}" ]] || die "required artifact missing: ${required}"
done
[[ "${preflight_only}" == "0" || "${preflight_only}" == "1" ]] || die "PREFLIGHT_ONLY must be 0 or 1"
[[ "${scene_seed_start}" =~ ^[0-9]+$ ]] || die "SCENE_SEED_START must be non-negative"
[[ "${scene_seed_stride}" =~ ^[1-9][0-9]*$ ]] || die "SCENE_SEED_STRIDE must be positive"

read -r -a gpus <<<"${gpus_text//,/ }"
[[ "${#gpus[@]}" -eq 4 ]] || die "this paired evaluation requires exactly four GPUs"
[[ "$(printf '%s\n' "${gpus[@]}" | sort -u | wc -l | tr -d ' ')" -eq 4 ]] || die "GPU indices must be unique"
for gpu in "${gpus[@]}"; do [[ "${gpu}" =~ ^[0-9]+$ ]] || die "invalid GPU: ${gpu}"; done

export PATH="${runtime_bin}:${PATH}"
export PYTHONPATH="${repo_root}/src:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${memorywam_root}/model_assets/diffsynth_official}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export FASTWAM_ADVANCE_POLICY_SEED=false

cd "${repo_root}"
"${python_bin}" - "${checkpoint}" "${step}" "${train_config}" "${dataset_stats}" "${runtime_lock}" "${selector_normalization}" <<'PY'
import hashlib
import sys
import zipfile
from pathlib import Path

from omegaconf import OmegaConf

checkpoint = Path(sys.argv[1]).resolve()
expected_step = int(sys.argv[2])
config = OmegaConf.load(sys.argv[3])
dataset_stats = Path(sys.argv[4]).resolve()
runtime_lock = Path(sys.argv[5]).resolve()
selector_normalization = Path(sys.argv[6]).resolve()

assert checkpoint.name == f"step_{expected_step:06d}.pt"
assert checkpoint.stat().st_size == 12042343818, checkpoint.stat().st_size
assert int(config.max_steps) == 50000
assert int(config.weight_only_resume_step) == 30000
assert int(config.save_every) == 0
assert list(config.save_steps) == [40000, 45000, 50000]
assert config.memory_allocation_rule == "event_full_forced_half"
assert config.model.native_cache.allocation_mode == "event_full_forced_half"
assert int(config.model.native_cache.memory_tokens) == 48
assert int(config.model.native_cache.dynamic_tokens_per_frame) == 8
assert int(config.model.native_cache.anchor_frames) == 2
assert int(config.model.native_cache.recent_frames) == 4
assert config.model.native_cache.recursive is False
assert dataset_stats.stat().st_size > 0
assert selector_normalization.stat().st_size > 0
assert hashlib.sha256(runtime_lock.read_bytes()).hexdigest() == "c27e63084f3b579f1d04d6ca62126ca8caa8f6032ae94286cab196402c6cfb96"
with zipfile.ZipFile(checkpoint) as archive:
    names = archive.namelist()
    assert any(name.endswith("/data.pkl") for name in names)
    assert any("/data/" in name for name in names)
print(f"checkpoint_preflight=pass step={expected_step}")
PY

if [[ "${preflight_only}" == "1" ]]; then
  echo "preflight_status=ok step=${step} scenes=${scene_seed_start},$((scene_seed_start+scene_seed_stride)),$((scene_seed_start+2*scene_seed_stride)),$((scene_seed_start+3*scene_seed_stride))"
  exit 0
fi

[[ ! -e "${log_root}" ]] || die "LOG_ROOT already exists: ${log_root}"
mkdir -p "${log_root}" "${artifact_root}"
printf 'worker\tgpu\tscene_seed\tlauncher_log\n' >"${log_root}/workers.tsv"

pids=()
workers=()
for index in 0 1 2 3; do
  gpu="${gpus[$index]}"
  scene_seed=$((scene_seed_start + index * scene_seed_stride))
  worker="worker$(printf '%02d' "${index}")"
  launcher_log="${log_root}/${worker}.launcher.log"
  printf '%s\t%s\t%s\t%s\n' "${worker}" "${gpu}" "${scene_seed}" "${launcher_log}" >>"${log_root}/workers.tsv"
  "${python_bin}" -u experiments/robotwin/eval_robotwin_single.py \
    --config-name "${eval_config_name}" \
    "ckpt=${checkpoint}" \
    "gpu_id=${gpu}" \
    "seed=${policy_seed}" \
    "EVALUATION.start_scene_seed=${scene_seed}" \
    "EVALUATION.task_name=battery_try" \
    "EVALUATION.task_config=demo_clean" \
    "EVALUATION.policy_name=${policy_name}" \
    "EVALUATION.eval_num_episodes=1" \
    "EVALUATION.output_dir=${log_root}/${worker}" \
    "EVALUATION.artifact_root=${artifact_root}" \
    "EVALUATION.dataset_stats_path=${dataset_stats}" \
    "EVALUATION.skip_get_obs_within_replan=false" \
    "+EVALUATION.control_information_normalization=${selector_normalization}" \
    >"${launcher_log}" 2>&1 &
  pids+=("$!")
  workers+=("${worker}")
  echo "launched step=${step} worker=${worker} gpu=${gpu} scene=${scene_seed} pid=$!"
done

failed=0
for index in 0 1 2 3; do
  if wait "${pids[$index]}"; then
    echo "${workers[$index]} status=ok"
  else
    echo "${workers[$index]} status=failed"
    failed=$((failed + 1))
  fi
done

"${python_bin}" - "${log_root}" "${artifact_root}" "${checkpoint}" <<'PY'
import json
import re
import subprocess
import sys
from pathlib import Path

log_root = Path(sys.argv[1])
artifact_root = Path(sys.argv[2])
checkpoint = Path(sys.argv[3])
ansi = re.compile(r"\x1b\[[0-9;]*m")
rows = []
for worker in range(4):
    log = log_root / f"worker{worker:02d}.launcher.log"
    text = log.read_text(errors="replace")
    clean = ansi.sub("", text)
    diagnostics = re.findall(r"RMBENCH_EVAL_DIAGNOSTICS (\{.*\})", clean)
    selector = re.findall(r"FASTWAM_CONTROL_INFORMATION_SELECTOR (\{.*\})", clean)
    contract = re.findall(r"FASTWAM_EVAL_CONTRACT (\{.*\})", clean)
    errors = [p for p in ("Traceback", "OutOfMemory", "CUDA out of memory", "NCCL ERROR", "nan") if p.lower() in clean.lower()]
    assert len(diagnostics) == 1, (log, len(diagnostics))
    assert len(selector) == 1, (log, len(selector))
    assert len(contract) == 1, (log, len(contract))
    d = json.loads(diagnostics[0])
    s = json.loads(selector[0])
    c = json.loads(contract[0])
    assert not errors, (log, errors)
    assert Path(c["checkpoint"]).resolve() == checkpoint.resolve()
    assert s["runtime_version"] == "v3"
    assert s["source"] == "frozen_initialization_wam"
    assert s["strict_online"] is True and s["forward_alignment"] is True
    assert s["uses_old_dynamic_surprise"] is False
    assert s["uses_gripper_hard_trigger"] is False
    rows.append(d)

videos = sorted(artifact_root.rglob("*.mp4"))
assert len(videos) == 4, [str(p) for p in videos]
for video in videos:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,nb_frames", "-of", "json", str(video)],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream["codec_name"] == "h264"
    assert int(stream["width"]) == 320 and int(stream["height"]) == 240
    assert int(stream["nb_frames"]) > 0

successes = sum(bool(row["official_success"]) for row in rows)
summary = {
    "checkpoint": str(checkpoint.resolve()),
    "episodes": 4,
    "successes": successes,
    "success_rate": successes / 4,
    "diagnostics": rows,
    "videos": [str(p) for p in videos],
    "strict_online_selector_audit": True,
}
(log_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(log_root / "summary.txt").write_text(
    f"checkpoint={checkpoint.resolve()}\nepisodes=4\nsuccess={successes}/4\nsuccess_rate={successes/4:.2%}\nstrict_online_selector_audit=true\n"
)
print(json.dumps(summary, indent=2))
PY

[[ "${failed}" -eq 0 ]] || die "${failed} worker(s) failed"
echo "evaluation_status=ok summary=${log_root}/summary.txt"
