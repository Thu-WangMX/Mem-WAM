# Native Cache 25k End-to-End Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Launch one reproducible native-cache end-to-end PutBack run from common initialization to step 25,000 with checkpoints at 1k and every 5k.

**Architecture:** Extend the trainer's periodic checkpoint predicate with explicit one-off save steps, then add a FullKV-matched Hydra task and guarded eight-GPU launcher. Preserve one uninterrupted optimizer/scheduler trajectory rather than staging two runs.

**Tech Stack:** Python 3.10, PyTorch, Hydra/OmegaConf, Accelerate FSDP, pytest, Bash.

## Global Constraints

- Work only in `/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_native_cache_consolidation`.
- Use the full 50-episode PutBack dataset and V4 continuous latent cache.
- Start from Wan2.2 plus official ActionDiT initialization; do not load a PutBack-trained checkpoint.
- Match the verified FullKV optimizer, timestep sampling, batch, seed, and trainable backbone policy.
- Train to 25,000 steps and save at 1k, 5k, 10k, 15k, 20k, and 25k.

---

### Task 1: Explicit checkpoint milestone

**Files:**
- Modify: `src/fastwam/trainer.py`
- Create: `tests/test_trainer_checkpoint_schedule.py`

**Interfaces:**
- Consumes: `step: int`, `save_every: int`, and `save_steps: Collection[int]`.
- Produces: `_is_checkpoint_step(...) -> bool`, used by the training loop.

- [ ] Write tests asserting `[1000] + every 5000` selects exactly `{1000, 5000, 10000, 15000, 20000, 25000}` through step 25k and rejects non-positive explicit steps.
- [ ] Run `PYTHONPATH=src pytest -q tests/test_trainer_checkpoint_schedule.py` and confirm failure because the predicate is absent.
- [ ] Add `_is_checkpoint_step`, parse and validate `cfg.save_steps`, and replace the inline modulo-only condition.
- [ ] Re-run the focused test to green.

### Task 2: FullKV-matched formal task and launcher

**Files:**
- Create: `configs/task/rmbench_putback_native_cache_25k.yaml`
- Create: `ops/train_putback_native_cache_25k.sh`

**Interfaces:**
- Consumes: the existing FullKV data/model configs and required environment paths.
- Produces: a fresh eight-GPU run with `resume=null`, full train mode, 25k steps, and the required checkpoint schedule.

- [ ] Add a Hydra task that overrides the model with `fastwam_native_cache_consolidation`, retains logit-normal video/action sampling, sets `minimum_history_frames=5`, and matches the verified FullKV optimization values.
- [ ] Add a launcher that validates all artifacts, exactly 50 episodes, exactly eight visible GPUs, and a nonexistent output directory before FSDP launch.
- [ ] Compose the config and assert the formal-run contract.
- [ ] Run `bash -n ops/train_putback_native_cache_25k.sh`.

### Task 3: Verification and launch

**Files:**
- Modify: `NATIVE_CACHE_CONSOLIDATION.md`

**Interfaces:**
- Consumes: the formal task and launcher.
- Produces: passing tests, a fresh-init real smoke, and a detached 25k training process with a monitored log.

- [ ] Run `PYTHONPATH=src pytest -q tests` and record the exact pass count.
- [ ] Run a one-step fresh-initialization eight-GPU smoke with final checkpoint disabled and verify finite video/action loss.
- [ ] Document the formal training contract and exact output path.
- [ ] Launch the 25k job, verify all workers are active, and inspect initial finite loss plus ETA.

