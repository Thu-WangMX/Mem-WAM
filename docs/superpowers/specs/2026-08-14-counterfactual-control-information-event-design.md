# Counterfactual Control-Information Event Segmentation

## Goal

Build a PutBack event selector that uses a frozen initialization-WAM to close a
memory segment only when a newly observed interaction state is both
unpredictable from the causal past and consequential for future control. The
selector must be strict-online, use no VLM or task-trained policy checkpoint,
and produce exactly the same planning-aligned groups during offline training
manifest generation and online RM-Bench rollout.

## Why the current selector is insufficient

The current embodied-information selector hard-triggers on gripper threshold
crossings. In the 50-episode training manifest, 234 of 245 non-terminal
boundaries are gripper transitions and only 11 come from WAM information. It is
therefore a control-rule segmenter with a WAM fallback, not a WAM-native event
selector. The earlier adjacent-latent and predictive-residual selectors also do
not directly test whether a visual surprise changes future control.

The official initialization ActionDiT cannot supply this signal directly. Its
transformer backbone is pretrained, but `action_encoder` and `head` are
explicitly excluded from the pretrained payload and remain random. The design
therefore uses the initialized video-WAM as a frozen representation and trains
small probes on top of those frozen features. The probes are selector artifacts,
not the action policy, and are frozen before policy training and evaluation.

## Selected method

At detector frame `t`, with simulator stride four:

1. Encode all observations available through `t` with the same causal VAE path
   used by policy evaluation.
2. Run the frozen initialization video-WAM on the matching phase stream and
   capture the locked multi-layer, multi-camera feature `h_t`.
3. Predict the feature that should have been observed from causal feature
   history, the last sixteen executed actions, and current proprioception:

       h_hat_t = F(h_<t, a_<t, p_t)

4. Run one frozen future-control probe twice, holding every input except the
   newest feature fixed:

       u_prior = G(h_<t, h_hat_t, p_t)
       u_post  = G(h_<t, h_t,     p_t)

   `G` predicts the normalized next sixteen expert actions. It receives no
   frame index or episode progress.
5. Define counterfactual control information as the robustly normalized action
   shift between `u_prior` and `u_post`. This quantity is small when the new
   observation is predictable, and also small when an unpredictable visual
   change is irrelevant to future control. It is large only when the
   unpredictable component of the WAM state changes the future action chunk.

The probe predicts actions in the dataset's normalized action space, so the
primary online scalar is:

    I_t = RMS(u_post - u_prior)

Calibration statistics are conditioned on detector phase and causal history
depth, using only the designated calibration episodes. No gripper transition is
a hard boundary. Gripper information can affect `I_t` only through the frozen
WAM representation, proprioception, and the probe's predicted control change.

## Frozen components and trainable selector artifacts

The following are immutable and hash-locked:

- official Wan2.2 TI2V video weights;
- official preprocessed ActionDiT backbone initialization;
- random initialization seed used for excluded ActionDiT modules, even though
  their direct outputs are not used by the selector;
- causal VAE and text embedding cache;
- multi-layer feature taps, camera regions, PCA transform, and action/proprio
  normalization statistics.

Two small artifacts are trained before policy training and then frozen:

- `F`, the existing action-conditioned causal WAM-feature predictor;
- `G`, a future-action probe operating on the PCA-projected frozen WAM feature
  and current proprioception.

`F` and `G` train only on demonstrations. They never receive a trained PutBack
policy checkpoint, rollout success labels, VLM labels, frame index, or future
observation. Future expert actions are labels for `G`, not online inputs.

## Boundary state machine

The shared boundary state consumes one `I_t` every four simulator frames.

- The first four detector observations (frames 0, 4, 8, and 12) are causal
  warm-up anchors and cannot use a prediction residual.
- Each post-warm-up `I_t` is converted to a one-sided robust z-score using
  phase/history-depth statistics.
- A leaky one-sided CUSUM accumulates only excess control information:

      C_t = max(0, decay * C_(t-1) + max(0, z_t - drift))

- A boundary is confirmed at the current frame when `C_t` reaches the locked
  threshold and the minimum segment length has elapsed.
- The boundary is never moved backward to an earlier peak. CUSUM, cooldown, and
  a minimum segment length suppress adjacent detections.
- A maximum segment length is a safety/compute fallback and is logged separately
  from learned events.
- Confirmation at detector frame `t` is aligned forward with `ceil(t / 16)` to
  the next policy planning decision. The current planning observation begins the
  next group; only the completed past group is compressed.
- Every completed variable-length group is compressed to eight memory tokens.

The exact same state-machine implementation is imported by offline manifest
generation and online evaluation. There is no offline backdating or inference
approximation.

## Data split and leakage controls

Use episode-disjoint roles:

- episodes 0-25: train `F` and `G`;
- episodes 26-29: model selection/early stopping;
- episodes 30-39: selector calibration and threshold selection;
- episodes 40-49: held-out boundary-quality evaluation opened only after the
  selector artifacts and thresholds are hash-locked.

The feature bank and PCA already follow the initialization-WAM contract. The
control-probe dataset must add, for every valid detector frame, the current
projected WAM feature, current proprioception, and next sixteen normalized
actions plus a validity mask for terminal tails.

Required shortcut diagnostics are:

- no frame index, episode length, or normalized progress in model inputs;
- compare against proprio-only and mean-by-progress baselines;
- shuffled-feature counterfactual must destroy event alignment;
- visual-action dynamics prediction must outperform its visual-only counterpart;
- held-out event patterns must not collapse to one fixed schedule.

## Artifact and parity contract

A locked selector manifest records SHA-256 hashes of:

- initialization manifest;
- feature-bank manifest and PCA;
- feature predictor and control probe;
- contextual score statistics;
- selector configuration;
- state-machine source;
- action/proprio normalization statistics.

Offline group manifests record detector confirmation frame, forward-aligned
planning boundary, event score, CUSUM value, boundary reason, and group length.
Online rollout logs the same fields. Replaying an episode prefix through both
paths must produce identical boundaries and reasons.

## Evaluation gates before policy training

The selector does not advance to a 40k policy run until all of these pass:

1. Unit tests prove causal prefix invariance, no retroactive boundary, reset,
   minimum/maximum length, terminal tail, and planning alignment.
2. Offline and online replay produce identical boundaries for all 50 expert
   episodes.
3. Event segmentation is genuinely dynamic: multiple segment-length patterns
   appear and learned events are not dominated by forced maximum boundaries.
4. The future-action probe beats proprio-only and schedule baselines on held-out
   next-action prediction.
5. Counterfactual event scores separate reviewed task transitions from
   background better than raw WAM residual, adjacent-latent distance, and the
   old gripper-hard-trigger selector at matched memory count.
6. Two rendered development videos are visually reviewed before the held-out set
   is opened; after locking, four held-out videos are rendered once.
7. Online selector latency and GPU memory fit alongside RM-Bench inference with
   no silent fallback.

## Policy-training integration

After the selector passes its gates, generate a new PutBack-only planning-group
manifest in a new artifact directory. Training starts from the same official
Wan2.2 and ActionDiT initialization as the fixed-K8 baseline, uses K=8 memory
tokens per dynamic group, saves portable policy checkpoints every 5k steps, and
does not save a 1k checkpoint. The trainer must pass decoded dynamic
`memory_groups` into the layerwise memory transform; a regression test prevents
the previously discovered fixed-layout omission.

Evaluation uses the policy checkpoint for action generation and the separately
hash-locked initialization-WAM selector for event decisions. The paper
comparison reports RM-Bench success, average/percentile group length, memory
group count, token count, inference latency, learned/forced boundary counts, and
failure stage against fixed K8, MemoryWAM-style single-frame K8, and FullKV.

## Failure behavior

Any fingerprint mismatch, missing selector artifact, non-contiguous detector
frame, missing executed-action history, non-finite score, or incompatible
planning alignment aborts loudly. Evaluation must never fall back to the old
dynamic-surprise scorer, gripper thresholds, fixed grouping, or a task-trained
boundary model.
