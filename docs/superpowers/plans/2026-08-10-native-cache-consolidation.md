# Native Cache Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tested, checkpointable, online-compatible native `4 -> 1` cache consolidation experiment to the verified FullKV continuous-VAE codebase.

**Architecture:** A block-0-initialized anchor-global compressor consumes four pre-DiT native token blocks and emits one same-shaped block. Training compresses completed clean-history groups before MoT; inference retains native block metadata alongside K/V and rewrites a four-block suffix into one rematerialized all-layer cache block.

**Tech Stack:** Python 3.10, PyTorch, Hydra/OmegaConf, Accelerate/FSDP, pytest, existing FastWAM Wan2.2 MoT implementation.

## Global Constraints

- Work only in `/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_native_cache_consolidation` after copying from the verified continuous-VAE FullKV source.
- Do not derive implementation code from `fastwam_full_history_stage1_v1`; reuse only its native-attention-copy and checkpoint-registration patterns.
- Do not average or edit post-RoPE K/V and do not synthesize compressed VAE latents.
- Keep output tokens native-shaped and preserve `endpoint`, `span`, and `level` metadata for later recursion.
- Phase one merges only four level-0 raw blocks and trains only compressor parameters.

---

### Task 1: Isolated project and clean baseline

**Files:**
- Create: the destination project by copying source files while excluding `.pytest_cache`, `__pycache__`, logs, runs, evaluation artifacts, and generated checkpoints.
- Create: `docs/superpowers/specs/2026-08-10-native-cache-consolidation-design.md`
- Create: `docs/superpowers/plans/2026-08-10-native-cache-consolidation.md`

**Interfaces:**
- Consumes: verified source tree and its Python environment.
- Produces: an isolated long-lived project with the same imports and baseline behavior.

- [ ] Copy the source tree with `rsync -a` and explicit generated-artifact exclusions.
- [ ] Copy this design and plan into the destination `docs/superpowers` tree.
- [ ] Run `PYTHONPATH=src pytest -q tests/test_full_kv_contract.py tests/test_fastwam_registration.py tests/test_compact_checkpoint.py`.
- [ ] Confirm the result is exactly 18 passing tests before feature edits.

### Task 2: Native block compressor

**Files:**
- Create: `src/fastwam/memory/__init__.py`
- Create: `src/fastwam/memory/native_cache.py`
- Create: `tests/test_native_cache.py`

**Interfaces:**
- Consumes: `video_block.self_attn`, `video_block.norm1`, source tokens `[B,4,N,D]`, source RoPE frequencies, levels `[B,4]`, and spans `[B,4]`.
- Produces: `NativeBlock(tokens, endpoint, span, level)` and `NativeBlockCompressor.forward(block_tokens, block_freqs, levels, spans) -> Tensor[B,N,D]`.

- [ ] Write a failing test that constructs a tiny VideoDiT block and asserts a zero-gate compressor returns the fourth input block bit-for-bit with shape `[B,N,D]`.
- [ ] Run `PYTHONPATH=src pytest -q tests/test_native_cache.py::test_zero_gate_is_exact_last_block_identity` and confirm failure because the module is absent.
- [ ] Implement dataclasses, strict shape validation, deep-copied native projections, 3D-RoPE Q/K, metadata embeddings, and the bounded zero-initialized residual gate.
- [ ] Re-run the identity test and confirm it passes.
- [ ] Write failing tests proving copied parameters do not alias block 0, nonzero gate changes output, early source blocks receive gradients, invalid spans/levels are rejected, and four raw blocks produce endpoint 3/span 4/level 1.
- [ ] Run those tests and confirm the expected behavioral failures.
- [ ] Implement the minimal metadata/consolidation methods, then run the complete `tests/test_native_cache.py` file to green.

### Task 3: Model construction and checkpoint contract

**Files:**
- Modify: `src/fastwam/runtime.py`
- Modify: `src/fastwam/models/wan22/fastwam.py`
- Modify: `src/fastwam/utils/compact_checkpoint.py`
- Modify: `tests/test_compact_checkpoint.py`
- Modify: `tests/test_fastwam_registration.py`

**Interfaces:**
- Consumes: Hydra `native_cache` mapping with `enabled`, `group_size`, `alpha_max`, and `max_levels`.
- Produces: registered `model.native_cache_compressor`, FullKV-compatible initialization, and portable payload key `native_cache_compressor`.

- [ ] Write failing registration and checkpoint tests: compressor parameters appear once outside `mot`; legacy FullKV load refreshes its copied attention from loaded VideoDiT block 0; native checkpoint save/load round-trips the compressor; portable export preserves its state.
- [ ] Run the focused tests and confirm failures at the missing construction/checkpoint boundaries.
- [ ] Thread `native_cache` through `create_fastwam`, `from_wan22_pretrained`, and `FastWAM.__init__`.
- [ ] Extend save/load and portable checkpoint helpers without changing legacy FullKV payload acceptance.
- [ ] Run the focused registration/checkpoint tests to green.

### Task 4: Compressed training forward

**Files:**
- Modify: `src/fastwam/models/wan22/fastwam.py`
- Modify: `src/fastwam/trainer.py`
- Modify: `tests/test_native_cache.py`
- Create: `configs/model/fastwam_native_cache_consolidation.yaml`
- Create: `configs/task/rmbench_putback_native_cache_smoke.yaml`

**Interfaces:**
- Consumes: `video_pre` plus `clean_frame_count` and `noisy_frame_count`.
- Produces: compressed video tokens/frequencies/modulation/context masks, retained clean-unit count, and future-only post-DiT state.

- [ ] Write failing tests for history lengths 1, 4, 5, and 9, asserting that the newest clean frame remains raw and only completed groups before it are compressed.
- [ ] Write a failing tiny-model test that runs `_training_loss_full_kv`, asserts finite video/action losses, calls backward, and observes compressor gradients while backbone gradients remain absent in compressor-only mode.
- [ ] Run the focused tests and confirm failures occur at the missing grouping/training path.
- [ ] Implement grouping and future-only decoding by reusing `pre_dit`, MoT, existing masks, schedulers, and loss functions.
- [ ] Add the compressor-only trainer mode and optimizer parameter selection.
- [ ] Add model/task configs inheriting the existing FullKV configs and changing only native-cache and smoke-specific fields.
- [ ] Run focused training tests to green.

### Task 5: Online native memory state and cache rewrite

**Files:**
- Modify: `src/fastwam/memory/native_cache.py`
- Modify: `src/fastwam/models/wan22/fastwam.py`
- Modify: `experiments/robotwin/fastwam_policy/deploy_policy.py`
- Modify: `tests/test_native_cache.py`

**Interfaces:**
- Consumes: optional `NativeCacheState`, current pre-DiT block, current endpoint, and current all-layer cache.
- Produces: action output and updated state whose `blocks` correspond exactly to its K/V sequence.

- [ ] Write a failing tiny-MoT integration test that appends four raw observations, verifies each is available to action prediction before rewrite, then verifies returned state contains one level-1/span-4 block and every cache layer shrinks from `4N` tokens to `N`.
- [ ] Write a failing test that a fifth observation appends after the summary and produces `2N` cached tokens without mutating the earlier state object.
- [ ] Run focused tests and confirm failures occur at the missing online state path.
- [ ] Implement append-then-consolidate using a sliced immutable cache prefix and existing `MoT.prefill_video_cache`; do not recompute or average old K/V.
- [ ] Update `infer_action` and RoboTwin policy reset/commit logic to pass one `native_cache_state` object atomically.
- [ ] Run the online-state tests to green.

### Task 6: Full verification and real smoke

**Files:**
- Create: `ops/train_putback_native_cache_smoke.sh`
- Create: `NATIVE_CACHE_CONSOLIDATION.md`

**Interfaces:**
- Consumes: verified step-24k FullKV checkpoint, v4 PutBack continuous-episode cache, and the new smoke config.
- Produces: reproducible command, one-step run directory, finite metrics, and a concise experimental contract.

- [ ] Add a shell launcher that validates every required path before invoking the existing `scripts/train.py` through the existing FSDP accelerate config.
- [ ] Run `PYTHONPATH=src pytest -q` and record pass/fail counts.
- [ ] Run a tiny CUDA forward/backward test and record peak allocation plus finite gradient status.
- [ ] Run the real one-step 8-GPU PutBack smoke from step 24,000 and inspect the log for finite total/video/action loss, a nonzero compressor gradient or gate update, and successful process exit.
- [ ] Re-run `PYTHONPATH=src pytest -q` after the smoke and compare the destination against the source to confirm only intended feature/config/document files changed.
- [ ] Document the source path, checkpoint, data contract, exact launch command, observed smoke metrics, and the explicit fact that recursive level-1 carry is not enabled in this phase.
