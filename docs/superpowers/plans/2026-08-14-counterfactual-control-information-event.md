# Counterfactual Control-Information Event Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a frozen-initialization-WAM selector that closes a dynamic PutBack memory segment online only when an unpredictable WAM state changes the predicted next action chunk.

**Architecture:** Reuse the locked multi-layer initialization-WAM feature bank and existing causal feature predictor as the counterfactual dynamics model. Train a small future-action probe on frozen WAM features, score the action shift between predicted and observed WAM states, feed that score through one strict-causal CUSUM state machine shared by offline manifest generation and online RM-Bench, and compress every resulting variable-length group to K=8 tokens.

**Tech Stack:** Python 3.10, PyTorch, pytest, Hydra/OmegaConf, existing FastWAM Wan2.2/ActionDiT runtime, RoboTwin/RM-Bench.

## Global Constraints

- All authoritative code edits and execution occur on `actionmem-02` under `/mnt/vepfs02/output/kevin.wang/memorywam/code/fastwam_dynamic_multiframe_selection`.
- New code/artifacts/checkpoints use `/mnt/vepfs02/output/kevin.wang`; only the still-unmoved runtime `/root/kevin_wang/envs/fastwam_guidemem` and dataset roots under `/mnt/vepfs02/output/kevin_wang` retain the underscore spelling.
- Do not overwrite or silently reuse the old dynamic-surprise, wrist-event, or embodied-information selector modules or artifacts.
- The selector uses frozen official Wan2.2 video-WAM initialization and never loads a task-trained policy checkpoint for boundary decisions.
- No VLM, no future observation, no frame index/progress input, no gripper hard trigger, and no retroactive boundary.
- Detector stride is four simulator frames; policy planning stride is sixteen; confirmations align only forward with `ceil(frame / 16)`.
- Every variable-length completed group is compressed to exactly eight memory tokens.
- Offline training-manifest generation and online evaluation import the same scorer, state machine, and planning aligner.
- A full 40k policy training run is forbidden until the selector quality, replay parity, and online latency gates pass.
- Formal policy checkpoints save every 5k steps, never at 1k.
- The authoritative target directory is not a Git repository. Instead of the commit steps normally required by this skill, every task ends with an atomic remote write, focused test log, and SHA-256 inventory under `analysis/putback_control_information_selector_v1/development_audit/`.

---

## File Map

- `src/fastwam/memory/control_information_dataset.py`: build strict-causal future-action-probe examples and masks.
- `src/fastwam/memory/control_information_probe.py`: future-action probe, masked loss, compact serialization, and counterfactual action-shift scorer.
- `src/fastwam/memory/control_information_boundary.py`: contextual robust calibration and causal CUSUM boundary state.
- `src/fastwam/evaluation/control_information_online.py`: locked artifact loading and strict-online feature/scorer/runtime path.
- `scripts/build_putback_control_information_dataset.py`: immutable 50-episode probe dataset creation.
- `scripts/train_putback_control_information_probe.py`: episode-disjoint probe and proprio-only baseline training.
- `scripts/trace_putback_control_information.py`: produce causal counterfactual score traces without opening held-out labels.
- `scripts/calibrate_putback_control_information_selector.py`: fit train statistics, sweep calibration-only CUSUM settings, and lock one selector.
- `scripts/freeze_putback_control_information_manifest.py`: generate planning-aligned dynamic groups using the shared state machine.
- `scripts/evaluate_putback_control_information_selector.py`: shortcut, dynamic-pattern, semantic, replay, and latency gates.
- `scripts/render_putback_control_information_selector.py`: detector/group overlays for visual review.
- `configs/sim_robotwin_control_information.yaml` and `configs/task/robotwin_control_information_eval.yaml`: strict-online RM-Bench configuration.
- `configs/task/rmbench_putback_control_information_k8_40k.yaml`: PutBack policy-training configuration.
- `ops/prepare_putback_control_information_selector.sh`: reproducible selector build/lock/gate pipeline.
- `ops/train_putback_control_information_k8_40k.sh`: gated eight-GPU formal training entry point.

### Task 1: Future-action probe dataset contract

**Files:**
- Create: `src/fastwam/memory/control_information_dataset.py`
- Create: `tests/test_control_information_dataset.py`
- Create: `scripts/build_putback_control_information_dataset.py`
- Test: `tests/test_control_information_dataset.py`

**Interfaces:**
- Consumes: projected phase rows `{frame_indices, features, warmup}`, raw `actions[T,14]`, raw `proprio[T,14]`, and locked dataset normalization statistics.
- Produces: `build_control_probe_examples(...) -> list[dict]`, where each row contains `episode`, `phase`, `frame`, `feature[1280]`, `proprio[14]`, `target_actions[16,14]`, and `target_mask[16]`.

- [ ] **Step 1: Write failing dataset tests**

```python
def test_control_probe_example_uses_only_current_state_and_future_labels():
    rows = build_control_probe_examples(
        episode=3,
        projected_phases=_four_phase_features(),
        actions=torch.arange(40 * 14).reshape(40, 14).float(),
        proprio=torch.zeros(40, 14),
        action_mean=torch.zeros(14),
        action_std=torch.ones(14),
        proprio_mean=torch.zeros(14),
        proprio_std=torch.ones(14),
        horizon=16,
    )
    row = next(value for value in rows if value["frame"] == 16)
    assert set(row) == {
        "episode", "phase", "frame", "feature", "proprio",
        "target_actions", "target_mask",
    }
    assert row["target_actions"].shape == (16, 14)
    assert row["target_mask"].dtype == torch.bool
    assert row["target_mask"].all()

def test_terminal_action_targets_are_padded_and_masked_without_future_observation():
    row = _build_short_episode_row(frame=28, episode_length=35)
    assert row["target_mask"].sum().item() == 7
    assert torch.count_nonzero(row["target_actions"][7:]).item() == 0
```

- [ ] **Step 2: Run tests and verify the missing-module failure**

Run:

```bash
cd /mnt/vepfs02/output/kevin.wang/memorywam/code/fastwam_dynamic_multiframe_selection
PYTHONPATH=src:. /root/kevin_wang/envs/fastwam_guidemem/bin/python -m pytest tests/test_control_information_dataset.py -q
```

Expected: collection fails because `fastwam.memory.control_information_dataset` does not exist.

- [ ] **Step 3: Implement normalized, masked examples**

```python
ACTION_DIM = 14
PROPRIO_DIM = 14

def build_control_probe_examples(*, episode, projected_phases, actions, proprio,
                                 action_mean, action_std, proprio_mean, proprio_std,
                                 horizon=16):
    actions = normalize_controls(actions, action_mean, action_std, "actions")
    proprio = normalize_controls(proprio, proprio_mean, proprio_std, "proprio")
    rows = []
    for phase in (0, 4, 8, 12):
        stream = projected_phases[str(phase)]
        for frame, feature in zip(stream["frame_indices"], stream["features"]):
            frame = int(frame)
            target = torch.zeros(horizon, ACTION_DIM)
            valid = min(horizon, len(actions) - frame)
            if valid <= 0:
                continue
            target[:valid] = actions[frame:frame + valid]
            mask = torch.arange(horizon) < valid
            rows.append({"episode": int(episode), "phase": phase, "frame": frame,
                         "feature": feature.clone(), "proprio": proprio[frame].clone(),
                         "target_actions": target, "target_mask": mask})
    return sorted(rows, key=lambda row: row["frame"])
```

The builder script must validate the existing feature-bank/PCA hashes, use episodes 0-49, store the exact normalization tensors and split roles, write to a `.tmp` path, then atomically rename.

- [ ] **Step 4: Run focused tests**

Run the command from Step 2. Expected: all tests pass.

- [ ] **Step 5: Record task audit**

Run:

```bash
mkdir -p /mnt/vepfs02/output/kevin.wang/memorywam/analysis/putback_control_information_selector_v1/development_audit
sha256sum src/fastwam/memory/control_information_dataset.py scripts/build_putback_control_information_dataset.py tests/test_control_information_dataset.py > /mnt/vepfs02/output/kevin.wang/memorywam/analysis/putback_control_information_selector_v1/development_audit/task01.sha256
```

### Task 2: Future-action probe and baseline training

**Files:**
- Create: `src/fastwam/memory/control_information_probe.py`
- Create: `tests/test_control_information_probe.py`
- Create: `scripts/train_putback_control_information_probe.py`
- Test: `tests/test_control_information_probe.py`

**Interfaces:**
- Consumes: Task 1 rows.
- Produces: `FutureActionProbe(feature_dim=1280, proprio_dim=14, horizon=16, action_dim=14)`, `masked_action_huber_loss`, `save_compact_control_probe`, and `load_compact_control_probe`.

- [ ] **Step 1: Write failing model, masking, and serialization tests**

```python
def test_probe_output_and_masked_loss_ignore_terminal_padding(tmp_path):
    model = FutureActionProbe(feature_dim=8, proprio_dim=3, hidden_dim=16,
                              horizon=4, action_dim=2)
    prediction = model(torch.randn(2, 8), torch.randn(2, 3))
    assert prediction.shape == (2, 4, 2)
    target = prediction.clone()
    target[0, 3] += 1000
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
    assert masked_action_huber_loss(prediction, target, mask).item() == 0

def test_compact_probe_round_trip_is_exact(tmp_path):
    source = _probe_with_deterministic_weights()
    path = tmp_path / "probe.cipbin"
    save_compact_control_probe(path, source, config=_config(), report={"ok": True})
    loaded, metadata = load_compact_control_probe(path, device=torch.device("cpu"))
    for key, value in source.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[key], rtol=0, atol=0)
    assert metadata["schema_version"] == "putback_control_information_probe_v1"
```

- [ ] **Step 2: Run tests and verify they fail on the missing module**

```bash
PYTHONPATH=src:. /root/kevin_wang/envs/fastwam_guidemem/bin/python -m pytest tests/test_control_information_probe.py -q
```

- [ ] **Step 3: Implement the probe and deterministic trainer**

```python
class FutureActionProbe(nn.Module):
    def __init__(self, *, feature_dim=1280, proprio_dim=14, hidden_dim=512,
                 horizon=16, action_dim=14):
        super().__init__()
        self.feature = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.SiLU())
        self.proprio = nn.Sequential(nn.Linear(proprio_dim, 128), nn.SiLU())
        self.fusion = nn.Sequential(nn.Linear(hidden_dim + 128, hidden_dim), nn.SiLU(),
                                    nn.Linear(hidden_dim, horizon * action_dim))
        self.horizon, self.action_dim = horizon, action_dim

    def forward(self, feature, proprio):
        fused = torch.cat([self.feature(feature), self.proprio(proprio)], dim=-1)
        return self.fusion(fused).reshape(len(feature), self.horizon, self.action_dim)
```

The training script must train `wam_proprio` and a matched `proprio_only` model with feature input zeroed, use episodes 0-25 for training and 26-29 for validation, deterministic seed 42, AdamW `3e-4`, early stopping patience 15, and emit masked MAE/RMSE plus compact frozen artifacts.

- [ ] **Step 4: Run focused tests and a tiny synthetic training test**

Expected: round-trip is bit-exact and a four-example synthetic dataset decreases validation loss.

- [ ] **Step 5: Record task audit**

```bash
sha256sum src/fastwam/memory/control_information_probe.py scripts/train_putback_control_information_probe.py tests/test_control_information_probe.py > /mnt/vepfs02/output/kevin.wang/memorywam/analysis/putback_control_information_selector_v1/development_audit/task02.sha256
```

### Task 3: Counterfactual action-shift scorer

**Files:**
- Modify: `src/fastwam/memory/control_information_probe.py`
- Create: `tests/test_counterfactual_control_scorer.py`
- Test: `tests/test_counterfactual_control_scorer.py`

**Interfaces:**
- Consumes: existing `FeaturePredictor`, frozen `FutureActionProbe`, causal feature history, last sixteen executed actions, current proprio, and observed projected WAM feature.
- Produces: `CounterfactualControlScore(information, predicted_feature, prior_action, posterior_action)` and `score_counterfactual_control_information(...)`.

- [ ] **Step 1: Write failing counterfactual tests**

```python
def test_information_is_zero_when_observed_feature_matches_prediction():
    result = score_counterfactual_control_information(
        feature_predictor=_copy_last_feature_predictor(),
        control_probe=_linear_probe_sensitive_to_first_feature(),
        history=torch.tensor([[1.0, 2.0]]),
        dynamics_condition=torch.zeros(3),
        observed_feature=torch.tensor([1.0, 2.0]),
        normalized_proprio=torch.zeros(1),
    )
    assert result.information == 0.0

def test_information_ignores_unpredictable_change_in_control_nullspace():
    sensitive = _score(observed=torch.tensor([2.0, 2.0]))
    nullspace = _score(observed=torch.tensor([1.0, 20.0]))
    assert sensitive.information > 0
    assert nullspace.information == 0
```

- [ ] **Step 2: Run tests and verify missing-interface failures**

```bash
PYTHONPATH=src:. /root/kevin_wang/envs/fastwam_guidemem/bin/python -m pytest tests/test_counterfactual_control_scorer.py -q
```

- [ ] **Step 3: Implement the two-pass frozen probe score**

```python
@torch.inference_mode()
def score_counterfactual_control_information(*, feature_predictor, control_probe,
                                             history, dynamics_condition,
                                             observed_feature, normalized_proprio):
    predicted = feature_predictor(history[None], dynamics_condition[None],
                                  lengths=torch.tensor([len(history)]))[0]
    prior = control_probe(predicted[None], normalized_proprio[None])[0]
    posterior = control_probe(observed_feature[None], normalized_proprio[None])[0]
    information = torch.sqrt(torch.mean((posterior - prior).square()))
    return CounterfactualControlScore(float(information), predicted, prior, posterior)
```

Require both models to be in evaluation mode with all parameters `requires_grad=False`; reject non-finite inputs and outputs.

- [ ] **Step 4: Run focused tests**

Expected: action-sensitive residual passes and control-nullspace residual is rejected.

- [ ] **Step 5: Record task audit**

```bash
sha256sum src/fastwam/memory/control_information_probe.py tests/test_counterfactual_control_scorer.py > /mnt/vepfs02/output/kevin.wang/memorywam/analysis/putback_control_information_selector_v1/development_audit/task03.sha256
```

### Task 4: Strict-causal CUSUM boundary state

**Files:**
- Create: `src/fastwam/memory/control_information_boundary.py`
- Create: `tests/test_control_information_boundary.py`
- Test: `tests/test_control_information_boundary.py`

**Interfaces:**
- Consumes: one scalar counterfactual information value per post-warm-up detector frame and phase/history-depth robust statistics.
- Produces: `fit_control_information_statistics`, `contextual_information_z`, `ControlInformationBoundaryEvent`, and `ControlInformationBoundaryState.update/finalize`.

- [ ] **Step 1: Write failing causality/state tests**

```python
def test_cusum_confirms_at_current_frame_without_peak_backdating():
    state = ControlInformationBoundaryState(
        statistics=_unit_statistics(), threshold=2.0, drift=0.5, decay=1.0,
        detector_stride=4, min_units=4, max_units=24, initial_group_start=0,
    )
    for frame in (0, 4, 8, 12):
        assert state.update(frame=frame, information=None) is None
    assert state.update(frame=16, information=3.0).confirmation_frame == 16
    assert state.retroactive_boundary_count == 0

def test_no_gripper_or_proprio_argument_exists():
    assert "proprio" not in inspect.signature(ControlInformationBoundaryState.update).parameters
    assert "gripper" not in inspect.signature(ControlInformationBoundaryState.update).parameters

def test_maximum_and_terminal_events_are_separate_reasons():
    assert _force_max().reason == "forced_maximum"
    assert _terminal_tail().reason == "terminal_tail"
```

- [ ] **Step 2: Run tests and verify the module is absent**

```bash
PYTHONPATH=src:. /root/kevin_wang/envs/fastwam_guidemem/bin/python -m pytest tests/test_control_information_boundary.py -q
```

- [ ] **Step 3: Implement contextual robust z-score and leaky CUSUM**

```python
z = max(0.0, (information - median[phase, depth]) / mad_scale[phase, depth])
self.cusum = max(0.0, self.decay * self.cusum + max(0.0, z - self.drift))
if units >= self.max_units:
    return self._emit(frame, "forced_maximum", information, z)
if units >= self.min_units and self.cusum >= self.threshold:
    return self._emit(frame, "counterfactual_control_information", information, z)
```

Enforce contiguous stride-four observations, four residual-free warmups, finite positive MAD, strictly increasing groups, and reset CUSUM after every emitted event.

- [ ] **Step 4: Run focused tests**

Expected: all causality, reset, min/max, and tail tests pass.

- [ ] **Step 5: Record task audit**

```bash
sha256sum src/fastwam/memory/control_information_boundary.py tests/test_control_information_boundary.py > /mnt/vepfs02/output/kevin.wang/memorywam/analysis/putback_control_information_selector_v1/development_audit/task04.sha256
```

### Task 5: Trace, calibration, and immutable selector lock

**Files:**
- Create: `scripts/trace_putback_control_information.py`
- Create: `scripts/calibrate_putback_control_information_selector.py`
- Create: `tests/test_control_information_calibration.py`
- Test: `tests/test_control_information_calibration.py`

**Interfaces:**
- Consumes: feature bank, PCA, existing frozen dynamics predictor, Task 2 probe, dataset controls, and split roles.
- Produces: score traces for episodes 0-39, contextual statistics from 0-29, calibration report from 30-39, and `locked_candidate.json` with immutable selector parameters and every dependency available before online-runtime integration.

- [ ] **Step 1: Write failing split, determinism, and lock tests**

```python
def test_calibration_never_reads_heldout_episodes():
    assert locked_split()["trace_episodes"] == list(range(40))
    assert locked_split()["statistics_episodes"] == list(range(30))
    assert locked_split()["calibration_episodes"] == list(range(30, 40))
    assert locked_split()["heldout_episodes"] == list(range(40, 50))

def test_locked_candidate_hashes_every_selector_dependency(tmp_path):
    lock = lock_candidate(tmp_path / "locked_candidate.json", candidate=_candidate(),
                          dependency_paths=_dependencies())
    assert set(lock["hashes"]) == {
        "initialization_manifest", "feature_bank_manifest", "pca",
        "feature_predictor", "control_probe", "contextual_statistics",
        "normalization", "boundary_source",
    }
```

- [ ] **Step 2: Run tests and verify missing scripts/interfaces**

```bash
PYTHONPATH=src:. /root/kevin_wang/envs/fastwam_guidemem/bin/python -m pytest tests/test_control_information_calibration.py -q
```

- [ ] **Step 3: Implement causal traces and calibration-only selection**

Generate scores in increasing detector-frame order with the same phase histories and last-sixteen-action condition as online runtime. Fit median/MAD by detector phase and capped history depth on episodes 0-29. Sweep this finite locked grid on episodes 30-39:

```python
THRESHOLDS = (2.0, 3.0, 4.0, 5.0, 6.0)
DRIFTS = (0.25, 0.5, 1.0)
DECAYS = (0.8, 0.9, 1.0)
MIN_UNITS = (4, 8)
MAX_UNITS = (16, 24)
```

Reject candidates whose mean group count differs by more than 10% from fixed group-size-four memory or whose learned-event fraction is below 0.5. Rank remaining candidates by descending fraction of total excess counterfactual information captured within one detector unit of a learned boundary, then lower forced-boundary fraction, then lower p95 group length, then lexicographic parameters. Write artifacts atomically and refuse overwrite.

- [ ] **Step 4: Run focused tests and deterministic synthetic calibration twice**

Expected: both runs produce byte-identical candidate JSON and no held-out access marker.

- [ ] **Step 5: Record task audit**

```bash
sha256sum scripts/trace_putback_control_information.py scripts/calibrate_putback_control_information_selector.py tests/test_control_information_calibration.py > /mnt/vepfs02/output/kevin.wang/memorywam/analysis/putback_control_information_selector_v1/development_audit/task05.sha256
```

### Task 6: Shared offline manifest and strict-online runtime

**Files:**
- Create: `src/fastwam/evaluation/control_information_online.py`
- Create: `scripts/freeze_putback_control_information_manifest.py`
- Create: `tests/test_control_information_online.py`
- Create: `tests/test_control_information_replay_parity.py`
- Test: both new test files.

**Interfaces:**
- Consumes: hash-locked artifacts from Task 5, initialization-WAM features or online images, executed actions, proprioception, and existing `OnlinePlanningBoundaryAligner`.
- Produces: `ControlInformationOnlineRuntime.observe`, `arrive_planning`, a final `locked_selector.json` extending the Task 5 candidate with the online-runtime hash, identical offline episode partitions, and `putback_control_information_planning_segments_v1`.

- [ ] **Step 1: Write failing online and replay tests**

```python
def test_online_confirmation_is_only_applied_at_forward_planning_arrival():
    runtime = _fake_runtime(event_at_frame=20)
    _observe_through(runtime, frame=20)
    assert runtime.arrive_planning(frame=16) is None
    assert runtime.arrive_planning(frame=32) == (0, 2)
    assert runtime.aligner.retroactive_boundary_count == 0

def test_offline_and_online_prefix_replay_are_identical():
    online = replay_online(_trace())
    offline = replay_offline(_trace())
    assert online["boundaries"] == offline["boundaries"]
    assert online["reasons"] == offline["reasons"]
```

- [ ] **Step 2: Run tests and verify missing runtime/manifest failures**

```bash
PYTHONPATH=src:. /root/kevin_wang/envs/fastwam_guidemem/bin/python -m pytest tests/test_control_information_online.py tests/test_control_information_replay_parity.py -q
```

- [ ] **Step 3: Implement one shared scorer/state path**

`ControlInformationOnlineRuntime.observe` must encode the current stride-four image prefix, capture and project the frozen initialization-WAM feature, call `score_counterfactual_control_information`, update `ControlInformationBoundaryState`, and queue any confirmation in `OnlinePlanningBoundaryAligner`. The offline freezer must call the same boundary-state helper over immutable score traces and `align_detector_events`; it must not reproduce the CUSUM logic locally.

The frozen manifest metadata must include `selector="locked_counterfactual_control_information_v1"`, `retroactive_boundary_count=0`, exact dependency hashes, dynamic segment histogram, learned/forced counts, and `memory_tokens_per_group=8`.

Create the final runtime lock only after `control_information_online.py` exists.
It must embed the Task 5 candidate SHA-256 unchanged and add hashes for
`online_source`, `planning_aligner_source`, and the offline freezer. Refuse to
open held-out traces unless this final lock already exists.

- [ ] **Step 4: Run focused tests plus all existing dynamic-layout tests**

```bash
PYTHONPATH=src:. /root/kevin_wang/envs/fastwam_guidemem/bin/python -m pytest \
  tests/test_control_information_online.py \
  tests/test_control_information_replay_parity.py \
  tests/test_dynamic_groups_decode.py \
  tests/test_dynamic_layerwise_layout.py \
  tests/test_dynamic_layerwise_online.py -q
```

- [ ] **Step 5: Record task audit**

```bash
sha256sum src/fastwam/evaluation/control_information_online.py scripts/freeze_putback_control_information_manifest.py tests/test_control_information_online.py tests/test_control_information_replay_parity.py > /mnt/vepfs02/output/kevin.wang/memorywam/analysis/putback_control_information_selector_v1/development_audit/task06.sha256
```

### Task 7: RM-Bench deployment integration with no fallback

**Files:**
- Modify: `experiments/robotwin/fastwam_policy/deploy_policy.py`
- Create: `configs/sim_robotwin_control_information.yaml`
- Create: `configs/task/robotwin_control_information_eval.yaml`
- Create: `tests/test_control_information_eval_plumbing.py`
- Test: `tests/test_control_information_eval_plumbing.py`

**Interfaces:**
- Consumes: Task 6 runtime and policy observations/actions.
- Produces: `control_information_online=true` evaluation mode that is mutually exclusive with all old dynamic selectors and resets cleanly per episode.

- [ ] **Step 1: Write failing configuration and policy-plumbing tests**

```python
def test_new_selector_is_mutually_exclusive_with_old_selectors():
    with pytest.raises(ValueError, match="exactly one online selector"):
        _make_policy(control_information_online=True, embodied_information_online=True)

def test_missing_locked_artifact_aborts_without_fixed_group_fallback():
    with pytest.raises(FileNotFoundError):
        _make_policy(control_information_online=True,
                     control_information_lock="/missing/lock.json")

def test_episode_reset_clears_selector_feature_and_cusum_state():
    policy = _make_fake_policy()
    policy.reset()
    assert policy._control_information_runtime.boundary_state.last_frame is None
```

- [ ] **Step 2: Run tests and verify the new configuration is unsupported**

```bash
PYTHONPATH=src:. /root/kevin_wang/envs/fastwam_guidemem/bin/python -m pytest tests/test_control_information_eval_plumbing.py -q
```

- [ ] **Step 3: Add explicit constructor, inference, and reset plumbing**

Parse only these new arguments: `control_information_online`, `control_information_lock`, `control_information_pca`, `control_information_feature_predictor`, `control_information_control_probe`, `control_information_statistics`, and `control_information_init_action_dit_path`. Validate all files and hashes before the first environment reset. Observe every fourth simulator frame, pass actually executed actions, and call `arrive_planning` only on stride-sixteen policy decisions. Never select the old scorer or fixed grouping on failure.

- [ ] **Step 4: Run focused and existing selector-plumbing tests**

```bash
PYTHONPATH=src:. /root/kevin_wang/envs/fastwam_guidemem/bin/python -m pytest \
  tests/test_control_information_eval_plumbing.py \
  tests/test_embodied_information_eval_plumbing.py \
  tests/test_wrist_event_eval_contract.py -q
```

- [ ] **Step 5: Record task audit**

```bash
sha256sum experiments/robotwin/fastwam_policy/deploy_policy.py configs/sim_robotwin_control_information.yaml configs/task/robotwin_control_information_eval.yaml tests/test_control_information_eval_plumbing.py > /mnt/vepfs02/output/kevin.wang/memorywam/analysis/putback_control_information_selector_v1/development_audit/task07.sha256
```

### Task 8: Selector quality, visual, replay, and latency gates

**Files:**
- Create: `scripts/evaluate_putback_control_information_selector.py`
- Create: `scripts/render_putback_control_information_selector.py`
- Create: `tests/test_control_information_quality_gate.py`
- Create: `ops/prepare_putback_control_information_selector.sh`
- Test: `tests/test_control_information_quality_gate.py`

**Interfaces:**
- Consumes: locked selector, held-out traces, reviewed phase annotations, offline/online replay reports, and measured latency.
- Produces: one immutable gate report with explicit pass/fail clauses and review videos.

- [ ] **Step 1: Write failing quality-gate tests**

```python
def test_gate_rejects_schedule_shortcut_and_forced_boundary_dominance():
    report = _passing_report()
    report["shortcut"]["wam_probe_mae"] = report["shortcut"]["proprio_only_mae"]
    assert evaluate_gate(report)["pass"] is False
    report = _passing_report()
    report["segments"]["learned_fraction"] = 0.49
    assert evaluate_gate(report)["pass"] is False

def test_gate_requires_full_replay_parity_and_zero_retroactive_boundaries():
    report = _passing_report()
    report["replay"]["matching_episodes"] = 49
    assert evaluate_gate(report)["pass"] is False
```

- [ ] **Step 2: Run tests and verify gate interfaces are absent**

```bash
PYTHONPATH=src:. /root/kevin_wang/envs/fastwam_guidemem/bin/python -m pytest tests/test_control_information_quality_gate.py -q
```

- [ ] **Step 3: Implement immutable gate and renderer**

The gate requires all clauses simultaneously:

```python
clauses = {
    "probe_beats_proprio": wam_probe_mae < 0.95 * proprio_only_mae,
    "probe_beats_schedule": wam_probe_mae < 0.95 * schedule_baseline_mae,
    "dynamic_patterns": distinct_segment_patterns >= 5,
    "learned_majority": learned_boundary_fraction >= 0.50,
    "no_retroactive": retroactive_boundary_count == 0,
    "replay_all_episodes": matching_replay_episodes == 50,
    "semantic_better_than_raw": semantic_f1 >= raw_residual_f1 + 0.05,
    "semantic_not_worse_than_gripper": semantic_f1 >= gripper_selector_f1 - 0.05,
    "online_latency": selector_latency_p95_ms <= locked_latency_limit_ms,
}
```

The renderer creates MP4s with frame, planning decision, counterfactual score, z-score, CUSUM, boundary reason, and group index. Render episodes 30 and 36 before lock review; after the lock is immutable, open episodes 40-49 once and render the four selected held-out episodes recorded in the lock.

- [ ] **Step 4: Run focused tests and pipeline preflight**

Run the focused pytest, then run `PRECHECK_ONLY=1 bash ops/prepare_putback_control_information_selector.sh`. Expected: all paths resolve, no held-out marker is created, and no old selector artifact appears in the resolved command.

- [ ] **Step 5: Record task audit**

```bash
sha256sum scripts/evaluate_putback_control_information_selector.py scripts/render_putback_control_information_selector.py ops/prepare_putback_control_information_selector.sh tests/test_control_information_quality_gate.py > /mnt/vepfs02/output/kevin.wang/memorywam/analysis/putback_control_information_selector_v1/development_audit/task08.sha256
```

### Task 9: Gated K8 policy-training integration and smoke test

**Files:**
- Create: `configs/task/rmbench_putback_control_information_k8_40k.yaml`
- Create: `ops/train_putback_control_information_k8_40k.sh`
- Create: `tests/test_control_information_training_contract.py`
- Modify only if the regression test fails: `src/fastwam/models/wan22/fastwam.py`
- Test: `tests/test_control_information_training_contract.py`

**Interfaces:**
- Consumes: passing Task 8 gate report and Task 6 planning manifest.
- Produces: a preflighted eight-GPU PutBack launcher and one-step smoke evidence that dynamic `memory_groups` reach the layerwise memory transform.

- [ ] **Step 1: Write failing training-contract tests**

```python
def test_launcher_requires_passing_selector_gate_before_output_creation():
    text = LAUNCHER.read_text()
    assert "selector_gate_report" in text
    assert 'assert report["pass"] is True' in text

def test_training_schedule_is_40k_every5k_without1k():
    cfg = OmegaConf.load(TASK_CONFIG)
    assert cfg.max_steps == 40000
    assert cfg.save_every == 5000
    assert cfg.save_steps == []
    assert cfg.model.native_cache.memory_tokens == 8

def test_dynamic_groups_are_forwarded_to_layerwise_transform():
    source = FASTWAM_SOURCE.read_text()
    assert "memory_groups=dynamic_memory_groups" in source
```

- [ ] **Step 2: Run tests and verify launcher/config are missing**

```bash
PYTHONPATH=src:. /root/kevin_wang/envs/fastwam_guidemem/bin/python -m pytest tests/test_control_information_training_contract.py -q
```

- [ ] **Step 3: Implement a fail-closed formal launcher**

Use new output `/mnt/vepfs02/output/kevin.wang/memorywam/train/control_information_k8_putback_e2e_40k_seed42`, new manifest `/mnt/vepfs02/output/kevin.wang/memorywam/data/putback_control_information_planning_segments_v1`, official model assets under `/mnt/vepfs02/output/kevin.wang`, runtime under `/root/kevin_wang/envs/fastwam_guidemem`, and dataset/text/latent/stat paths only where they still physically exist under `/mnt/vepfs02/output/kevin_wang`. Require exactly eight visible GPUs, passing immutable gate report, 50 complete episode manifests, dynamic length histogram, zero retroactive boundaries, K=8, seed 42, LR `2e-4`, bf16, and checkpoint schedule 5k through 40k. Refuse an existing output directory.

- [ ] **Step 4: Run tests, preflight, and one-step isolated smoke**

```bash
PYTHONPATH=src:. /root/kevin_wang/envs/fastwam_guidemem/bin/python -m pytest \
  tests/test_control_information_training_contract.py \
  tests/test_dynamic_groups_decode.py \
  tests/test_dynamic_layerwise_layout.py -q
PREFLIGHT_ONLY=1 bash ops/train_putback_control_information_k8_40k.sh
```

Then run a separate one-step smoke output with `max_steps=1`, portable final checkpoint disabled, and exactly one batch. Inspect the captured layerwise layout assertion; do not start the 40k run.

- [ ] **Step 5: Record final implementation audit**

```bash
find src/fastwam/memory src/fastwam/evaluation scripts configs ops tests -type f -newer docs/superpowers/specs/2026-08-14-counterfactual-control-information-event-design.md -print0 | sort -z | xargs -0 sha256sum > /mnt/vepfs02/output/kevin.wang/memorywam/analysis/putback_control_information_selector_v1/development_audit/implementation.sha256
PYTHONPATH=src:. /root/kevin_wang/envs/fastwam_guidemem/bin/python -m pytest tests/test_control_information_*.py -q | tee /mnt/vepfs02/output/kevin.wang/memorywam/analysis/putback_control_information_selector_v1/development_audit/final_pytest.log
```

The implementation is not complete until the selector artifacts are locked,
the gate report passes, all 50 expert episodes have replay parity, the online
runtime has been exercised on actionmem-02 without fallback, and the isolated
dynamic-layout smoke confirms that the training transformer receives the new
manifest's groups.
