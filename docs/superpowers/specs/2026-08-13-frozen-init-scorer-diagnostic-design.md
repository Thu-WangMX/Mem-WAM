# Frozen Initialization Scorer Diagnostic

## Goal

Determine whether dynamic PutBack segmentation degrades because the task-trained
30k scorer co-adapts and drifts, or because online policy rollouts and predicted
actions differ from offline expert conditioning. This is a diagnostic only: no
training and no checkpoint mutation.

## Chosen design

Keep the 30k dynamic-surprise model as the sole action and memory model. Load a
second, frozen initialization model from the same official Wan2.2 video and
ActionDiT initialization used by `boundary_init_seed42`. The frozen model scores
each observed transition using the 30k policy's normalized predicted action and
full observed latent history (`memory_groups=None`). Its boundary decision alone
controls cache consolidation. Log both the selected frozen-init score and its
boundary decision.

This dual-model design is preferred over offline-only precomputation, which
cannot cover autonomous future observations, and over an action-free metric,
which would change the scorer definition rather than isolate checkpoint drift.

## Evaluation contract

Run the existing 30k checkpoint on the same four RMBench PutBack pairs:

- scene 100000 / policy 1000
- scene 200000 / policy 1002
- scene 300000 / policy 1004
- scene 400000 / policy 1006

Preserve sigma 1.0, gamma 1.5, window 5, min segment 2, max segment 8,
50 denoising steps, 16-step replanning, and eight memory tokens. Compare against
the already completed 30k self-scorer run using official success, stage, button
press count, center placement, segment lengths, and trigger reasons.

## Safety and validation

The scorer source is explicit and defaults to the existing policy scorer. A
preflight validates that initialization mode cannot accept a task checkpoint.
Unit tests cover source selection and episode reset. The four-scene launcher
uses an isolated output directory. If the second model exceeds memory or fails
to load, stop without falling back silently.
