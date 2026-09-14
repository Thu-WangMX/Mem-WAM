# Native Cache RoboTwin Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a native-cache-specific RoboTwin evaluation entry that is preflight-validated, measurable, and runnable from actionmem-01.

**Architecture:** Inherit the verified FullKV evaluation configuration and switch only the model selection to native cache. Keep checkpoint inspection in a lightweight standalone script, expose cache telemetry through a pure helper plus the existing policy, and use one environment-parameterized shell launcher for smoke and formal runs.

**Tech Stack:** Python 3.10, PyTorch, Hydra/OmegaConf, pytest, Bash, RoboTwin/RMBench.

## Global Constraints

- Work only in `/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_native_cache_consolidation` and its local mirror.
- Reuse shared code and assets; do not copy model, dataset, policy, or RoboTwin implementations.
- Match `sim_robotwin_full_kv.yaml` for action horizon 16, replan steps 16, 50 inference steps, seen instructions, and unprojected actions.
- Prepare on shared storage and run future rollout evaluation from actionmem-01.
- Do not interrupt training on actionmem-02 or existing workloads on actionmem-01.

---

### Task 1: Native evaluation configuration

**Files:**
- Create: `configs/task/robotwin_native_cache_eval.yaml`
- Create: `configs/sim_robotwin_native_cache.yaml`
- Create: `tests/test_native_cache_eval_contract.py`

**Interfaces:**
- Consumes: Hydra config groups `robotwin_full_kv_eval`, `fastwam_native_cache_consolidation`, and `sim_robotwin_full_kv`.
- Produces: composable config name `sim_robotwin_native_cache` with `model.native_cache.enabled=true` and the unchanged FullKV evaluation contract.

- [ ] Write a test that composes `sim_robotwin_native_cache` and asserts native model selection plus the exact evaluation values.
- [ ] Run `PYTHONPATH=src pytest -q tests/test_native_cache_eval_contract.py` and verify it fails because the config is absent.
- [ ] Add the task and simulator configs using Hydra defaults inheritance.
- [ ] Re-run the focused test and verify it passes.

### Task 2: Lightweight checkpoint preflight

**Files:**
- Create: `scripts/validate_native_cache_checkpoint.py`
- Create: `tests/test_validate_native_cache_checkpoint.py`

**Interfaces:**
- Consumes: `validate_checkpoint(path: Path) -> dict[str, object]`.
- Produces: a JSON summary containing checkpoint path, step, and parameter-group tensor counts; exits nonzero with a precise error for missing required groups.

- [ ] Write tests using temporary `torch.save` payloads for one valid payload and missing native-cache weights.
- [ ] Run `PYTHONPATH=src pytest -q tests/test_validate_native_cache_checkpoint.py` and verify the import fails because the validator is absent.
- [ ] Implement CPU-only validation of `native_cache_compressor`, `dit`, `action_expert`, and `proprio_encoder` mappings.
- [ ] Re-run the focused tests and verify they pass.

### Task 3: Native-cache evaluation telemetry

**Files:**
- Modify: `src/fastwam/memory/native_cache.py`
- Modify: `experiments/robotwin/fastwam_policy/deploy_policy.py`
- Modify: `tests/test_native_cache.py`

**Interfaces:**
- Consumes: `summarize_native_cache_state(state: NativeCacheState) -> dict[str, object]`.
- Produces: stable cache counts and `FASTWAM_NATIVE_CACHE_METRICS` JSON log lines after successful replans.

- [ ] Write a synthetic-state unit test asserting retained blocks/tokens/span, level histogram, layer count, and K/V tokens.
- [ ] Run the focused test and verify it fails because the helper is absent.
- [ ] Implement the pure summary helper and invoke it only after inference state commit.
- [ ] Add CUDA peak allocated/reserved bytes and reset peak statistics at episode reset when CUDA is active.
- [ ] Re-run `tests/test_native_cache.py` and verify it passes.

### Task 4: actionmem-01 launcher and end-to-end preflight

**Files:**
- Create: `ops/eval_putback_native_cache.sh`
- Modify: `NATIVE_CACHE_CONSOLIDATION.md`

**Interfaces:**
- Consumes: `CHECKPOINT`, `SEEDS`, `GPUS`, `EVAL_NUM_EPISODES`, `RUN_TAG`, `LOG_ROOT`, and `PREFLIGHT_ONLY` environment variables.
- Produces: validated per-seed RoboTwin workers using `--config-name sim_robotwin_native_cache`, or a no-GPU preflight report.

- [ ] Add a guarded launcher that validates paths, one-to-one seed/GPU mapping, checkpoint payload, unique output location, and shell arguments.
- [ ] Run `bash -n ops/eval_putback_native_cache.sh`.
- [ ] Run `PREFLIGHT_ONLY=1` on actionmem-01 against a synthetic structurally valid checkpoint and verify no simulator process starts.
- [ ] Run the full maintained test suite and record the exact result before reporting completion.

