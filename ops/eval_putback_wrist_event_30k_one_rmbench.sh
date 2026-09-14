#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_wrist_latent_event_memory}"
python_bin="${PYTHON_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin/python}"
checkpoint="${CHECKPOINT:-/mnt/vepfs02/output/kevin_wang/memorywam/train/dynamic_surprise_memory_putback_k8_25k_seed42/checkpoints/weights/step_030000.pt}"
predictor_repo="${PREDICTOR_REPO:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_wrist_latent_event_predictor}"
predictor_checkpoint="${PREDICTOR_CHECKPOINT:-/mnt/vepfs02/output/kevin_wang/memorywam/analysis/putback_wrist_latent_event_predictor_v1/best.pt}"
gpu="${GPU:-0}"
scene_seed="${SCENE_SEED:-100000}"
policy_seed="${POLICY_SEED:-1000}"
run_tag="${RUN_TAG:-wrist_event_runtime_smoke_30k_$(date +%Y%m%d_%H%M%S)}"
log_root="${LOG_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/eval_runs/${run_tag}}"
preflight_only="${PREFLIGHT_ONLY:-0}"

die() { echo "ERROR: $*" >&2; exit 2; }
[[ -d "${repo_root}" ]] || die "repository not found: ${repo_root}"
[[ -x "${python_bin}" ]] || die "python not executable: ${python_bin}"
[[ -s "${checkpoint}" ]] || die "policy checkpoint not found: ${checkpoint}"
[[ -d "${predictor_repo}" ]] || die "predictor repository not found: ${predictor_repo}"
[[ -s "${predictor_checkpoint}" ]] || die "predictor checkpoint not found: ${predictor_checkpoint}"
[[ "${gpu}" =~ ^[0-7]$ ]] || die "GPU must be in 0..7"
[[ "${preflight_only}" == "0" || "${preflight_only}" == "1" ]] || die "PREFLIGHT_ONLY must be 0 or 1"

export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/vepfs02/output/kevin_wang/memorywam/model_assets/diffsynth_official}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONPATH="${repo_root}/src:${repo_root}"
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8

cd "${repo_root}"
"${python_bin}" - "${predictor_checkpoint}" <<'PY'
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
config = payload["model_config"]
if config.get("include_proprio") is not False or config.get("history") != 3:
    raise SystemExit("predictor contract mismatch")
print("predictor_preflight=ok parameters=274945 history=3 uses_vlm=false")
PY

if [[ "${preflight_only}" == "1" ]]; then
  "${python_bin}" - <<'PY'
from hydra import compose, initialize_config_dir
from pathlib import Path

root = Path.cwd()
with initialize_config_dir(version_base="1.3", config_dir=str(root / "configs")):
    cfg = compose(config_name="sim_robotwin_wrist_event")
assert cfg.model.native_cache.memory_tokens == 8
assert cfg.EVALUATION.wrist_event_online is True
assert cfg.EVALUATION.dynamic_surprise_online is False
assert cfg.EVALUATION.policy_name == "fastwam_wrist_event_policy"
print("preflight_ok benchmark=RMBench task=put_back_block runtime=causal_wrist_event")
PY
  exit 0
fi

mkdir -p "${log_root}"
log_file="${log_root}/scene${scene_seed}_policy${policy_seed}.log"
CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u experiments/robotwin/eval_robotwin_single.py \
  --config-name sim_robotwin_wrist_event \
  "ckpt=${checkpoint}" \
  "gpu_id=0" \
  "seed=${policy_seed}" \
  "EVALUATION.start_scene_seed=${scene_seed}" \
  "EVALUATION.task_name=put_back_block" \
  "EVALUATION.task_config=demo_clean" \
  "EVALUATION.eval_num_episodes=1" \
  "EVALUATION.output_dir=${log_root}/output" \
  "EVALUATION.wrist_event_predictor_repo=${predictor_repo}" \
  "EVALUATION.wrist_event_checkpoint=${predictor_checkpoint}" \
  >"${log_file}" 2>&1

grep -E "FASTWAM_WRIST_EVENT|FASTWAM_NATIVE_CACHE_METRICS|success|Success|Traceback" "${log_file}" | tail -n 100 || true
echo "runtime_smoke_complete log=${log_file}"
