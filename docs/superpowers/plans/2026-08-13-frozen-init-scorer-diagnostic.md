# Frozen Initialization Scorer Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate the 30k PutBack policy while a separately loaded, frozen official initialization model selects online surprise boundaries.

**Architecture:** The policy model remains responsible for action generation and memory. A second model is optional and used only by `score_transition`; initialization scoring uses full observed history and no task checkpoint. The existing policy scorer remains the default.

**Tech Stack:** Python, PyTorch, Hydra, pytest, Bash, RMBench.

## Global Constraints

- Do not train or mutate checkpoints.
- Preserve sigma 1.0, gamma 1.5, window 5, min segment 2, max segment 8, 50 denoising steps, 16-step replanning, and eight memory tokens.
- Fail explicitly if the initialization scorer cannot load; never fall back silently.
- Evaluate scenes 100000, 200000, 300000, and 400000 with policy seeds 1000, 1002, 1004, and 1006.

---

### Task 1: Frozen scorer source selection

**Files:**
- Modify: `experiments/robotwin/fastwam_policy/deploy_policy.py`
- Modify: `experiments/robotwin/eval_robotwin_single.py`
- Modify: `configs/sim_robotwin_dynamic_surprise.yaml`
- Test: `tests/test_dynamic_surprise_frozen_init_scorer.py`

**Interfaces:**
- Consumes: the existing `score_transition(model, ...)` API.
- Produces: `dynamic_surprise_scorer_source` with values `policy` and `initialization`, plus `self._surprise_scoring_model`.

- [ ] **Step 1: Write a failing test** that constructs policy configuration and verifies initialization mode selects a distinct frozen model while policy mode selects `self.model`.
- [ ] **Step 2: Run** `PYTHONPATH=src:. python -m pytest -q tests/test_dynamic_surprise_frozen_init_scorer.py` and confirm failure because the source option is absent.
- [ ] **Step 3: Implement minimal selection** by instantiating the same model config with `load_text_encoder=False`, `skip_dit_load_from_pretrain=False`, no task checkpoint, moving it to the policy device, and setting `.eval()` with all parameters `requires_grad_(False)`.
- [ ] **Step 4: Route scoring** through `self._surprise_scoring_model`; use `memory_groups=None` for initialization mode and preserve current groups for policy mode.
- [ ] **Step 5: Propagate Hydra option** through the simulation config and evaluator override list, defaulting to `policy`.
- [ ] **Step 6: Run** `PYTHONPATH=src:. python -m pytest -q tests/test_dynamic_surprise_frozen_init_scorer.py tests/test_dynamic_surprise_scorer.py tests/test_dynamic_surprise_online.py tests/test_dynamic_layerwise_online.py` and require all tests to pass.

### Task 2: Four-scene isolated launcher

**Files:**
- Create: `ops/diagnose_putback_30k_frozen_init_scorer_four.sh`
- Test: `tests/test_frozen_init_scorer_launcher.py`

**Interfaces:**
- Consumes: `EVALUATION.dynamic_surprise_scorer_source=initialization`.
- Produces: one isolated evaluation directory containing four logs and RMBench diagnostics.

- [ ] **Step 1: Write a failing launcher test** that invokes `PREFLIGHT_ONLY=1` and requires the resolved scorer source to be `initialization`, checkpoint step to be 30000, and exactly four unique scene/policy pairs.
- [ ] **Step 2: Run** `PYTHONPATH=src:. python -m pytest -q tests/test_frozen_init_scorer_launcher.py` and confirm failure because the launcher is absent.
- [ ] **Step 3: Implement the launcher** with the four exact scene/policy pairs, GPUs 0-3, 30k checkpoint, unique output directory, and a preflight mode that performs no model load.
- [ ] **Step 4: Run** `bash -n ops/diagnose_putback_30k_frozen_init_scorer_four.sh` and both new test files.
- [ ] **Step 5: Launch on actionmem-02**, monitor until all four enter planning, and collect success, stage, center, press count, segment lengths, and trigger reasons.

### Task 3: Paired result analysis

**Files:**
- Create: `scripts/summarize_frozen_init_scorer_ab.py`
- Test: `tests/test_summarize_frozen_init_scorer_ab.py`

**Interfaces:**
- Consumes: baseline self-scorer and frozen-init log directories.
- Produces: JSON with per-scene paired outcomes and aggregate trigger statistics.

- [ ] **Step 1: Write a failing parser test** using two literal synthetic logs and hand-derived expected segment lengths and RMBench fields.
- [ ] **Step 2: Run** `PYTHONPATH=src:. python -m pytest -q tests/test_summarize_frozen_init_scorer_ab.py` and confirm failure because the summarizer is absent.
- [ ] **Step 3: Implement strict parsing** that rejects missing scenes, duplicate scenes, missing diagnostics, and mismatched policy seeds.
- [ ] **Step 4: Run the parser test and complete relevant test suite**, then summarize the actual four paired scenes.
