# Native-Cache Battery 25k Training Design

## Goal

Add a Volcano Cloud-ready Battery Try training entry to the isolated native-cache project, matching the formal PutBack experiment except for the RMBench task data.

## Experiment contract

- Use all 50 `battery_try` episodes from the existing RGB-v3 assets.
- Use `FullKVRobotVideoDataset` with the continuous episode stride-16 latent cache, dataset statistics, and text cache already produced for Battery Try.
- Use `fastwam_native_cache_consolidation` in end-to-end `native_cache_train_mode: full` mode.
- Start from the same official Wan2.2 video initialization and official ActionDiT initialization as PutBack, not a task-trained checkpoint.
- Train for 25,000 optimizer steps on eight visible GPUs with FSDP full shard, batch size 1 per rank, BF16, learning rate `2e-4`, seed 42.
- Save portable checkpoints at step 1,000 and every 5,000 steps; retain one resumable FSDP training-state checkpoint.
- Never overwrite an existing formal output directory.

## Files and boundaries

- `configs/data/rmbench_battery_fullattention.yaml` owns Battery dataset paths and preprocessing only.
- `configs/task/rmbench_battery_native_cache_25k.yaml` owns optimization, checkpointing, and native-cache training settings.
- `ops/train_battery_native_cache_25k.sh` owns Volcano Cloud environment setup, asset validation, the eight-GPU contract, preflight mode, and launch.
- `tests/test_battery_training_contract.py` executes the real launcher's preflight against controlled assets and checks observable behavior.

## Validation

The launcher validates the RGB build contract, exactly 50 episodes, the continuous-latent manifest schema and completeness, nonempty text cache, official ActionDiT initialization, eight unique visible GPU identifiers, and a non-existing output directory. `PREFLIGHT_ONLY=1` performs all checks and Hydra config composition without starting distributed workers. Any real smoke training may run only on actionmem-01.

