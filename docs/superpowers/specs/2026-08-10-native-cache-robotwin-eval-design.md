# Native Cache RoboTwin Evaluation Design

## Goal

Prepare a reproducible evaluation path for native-cache checkpoints that can be launched from actionmem-01 while reusing the shared isolated project, RMBench installation, model assets, dataset statistics, and FullKV evaluation contract.

## Configuration boundary

- Add a native-cache evaluation task that selects `fastwam_native_cache_consolidation` instead of `fastwam_full_kv`.
- Add a native-cache simulator config that inherits the verified FullKV simulator settings without changing action horizon, replan interval, denoising steps, language contract, observation refresh, or action projection.
- Do not duplicate model, policy, dataset, or RoboTwin code.

## Checkpoint and launch contract

- Reject a missing or malformed checkpoint before allocating a simulator or GPU-heavy model.
- Require the checkpoint payload to contain `native_cache_compressor`, `dit`, `action_expert`, and `proprio_encoder` state dictionaries.
- Provide one parameterized launcher for actionmem-01. Its defaults run a one-seed, one-episode smoke; environment overrides select seeds, episode count, checkpoint, GPUs, and output root for formal evaluation.
- Keep every output in a unique run directory and record the resolved checkpoint, config, seeds, episodes, and GPU mapping.

## Evaluation telemetry

- Preserve the existing rollout inference/simulator timing.
- When native cache is active, report after each successful replan: retained block count, retained token count, total represented frame span, block level histogram, per-layer K/V token count, and CUDA peak allocated/reserved bytes.
- Reset per-rollout counters and CUDA peak statistics at episode reset.
- Emit telemetry as a stable JSON-prefixed log line so later scripts can aggregate it without parsing free-form text.

## Verification

- Unit-test checkpoint validation with temporary valid and invalid payloads.
- Unit-test cache telemetry on a synthetic `NativeCacheState`.
- Compose the Hydra config and assert that it selects the native compressor while retaining the FullKV evaluation values.
- Syntax-check the launcher and run its preflight-only mode on actionmem-01. Do not start a RoboTwin rollout until a requested checkpoint exists and suitable GPUs are available.

