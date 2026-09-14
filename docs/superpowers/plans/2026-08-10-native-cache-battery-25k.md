# Native-Cache Battery 25k Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a directly runnable Volcano Cloud script for a 25k native-cache Battery Try experiment aligned with PutBack.

**Architecture:** Add one Battery-specific FullKV dataset config and one task config while reusing the existing model, trainer, cache dataset, and FSDP launcher. A strict shell preflight validates assets and composes the Hydra job before distributed launch.

**Tech Stack:** Bash, Hydra/OmegaConf YAML, PyTorch Accelerate FSDP, pytest.

## Global Constraints

- Formal training uses all 50 Battery Try episodes and exactly eight visible GPUs.
- Formal schedule is 25k steps, save at step 1k and every 5k, seed 42.
- Formal initialization is the official Wan2.2/ActionDiT initialization, not a task checkpoint.
- Smoke training, if needed, runs only on actionmem-01.
- Existing output directories are never overwritten.

---

### Task 1: Battery training contract

**Files:**
- Create: `tests/test_battery_training_contract.py`
- Create: `configs/data/rmbench_battery_fullattention.yaml`
- Create: `configs/task/rmbench_battery_native_cache_25k.yaml`
- Create: `ops/train_battery_native_cache_25k.sh`

**Interfaces:**
- Consumes: existing Battery RGB-v3 assets and `scripts/train.py` Hydra entrypoint.
- Produces: `PREFLIGHT_ONLY=1 bash ops/train_battery_native_cache_25k.sh` and the same command without the variable for formal launch.

- [ ] Write behavioral tests that invoke the absent launcher and therefore fail.
- [ ] Run the focused tests and confirm the failure is caused by the missing Battery entry.
- [ ] Add the minimal data config, task config, and strict launcher.
- [ ] Run the focused tests and the complete test suite.
- [ ] Run shell syntax checking and real preflight on actionmem-01.

