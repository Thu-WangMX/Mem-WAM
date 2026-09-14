# Native Cache Consolidation

## Formal 25k end-to-end mainline

The paper mainline is layerwise block memory, not the legacy block-0 compressor
described later in this file. Every completed four-frame block produces 32
learned memory tokens. Those tokens traverse every VideoDiT layer and retain
that layer's native K/V. Inference retains the first two raw anchor frames, all
completed block memories, and the four most recent raw frames. Memory and recent
raw frames may overlap temporarily, matching the MemoryWAM gist-plus-recent
contract. There is no K/V averaging, copied attention block, residual last-frame
shortcut, scalar gate, distillation, or recursive memory-of-memory in phase one.

Training uses a packed causal mask that reproduces the same reads, memory commits,
and raw-frame evictions as online inference. A numerical parity test compares all
retained per-layer K/V between training and inference for the same history.

The run starts from the official initialization, not a task-trained checkpoint:
VideoDiT loads Wan2.2-TI2V-5B and ActionDiT loads the official interpolated
initialization while its action encoder and head are initialized normally.
`resume` is null and `native_cache_train_mode` is `full`, so VideoDiT, ActionDiT,
proprio, and the layerwise memory slots train together from step zero.

The current PutBack run uses all 50 episodes, the V4 continuous cache, histories beginning
at one frame, logit-normal timestep sampling, global batch 8, AdamW at `2e-4`,
BF16, seed 42, and 25,000 optimizer steps. Portable checkpoints and resumable
training state are saved at steps 1,000, 5,000, 10,000, 15,000, 20,000, and
25,000.

```bash
cd /mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_native_cache_consolidation
PREFLIGHT_ONLY=1 ./ops/train_putback_native_cache_25k.sh
./ops/train_putback_native_cache_25k.sh
```

The launcher requires exactly eight visible GPUs, validates all 50 episodes and
cached artifacts, verifies the resolved training contract, and refuses to
overwrite an existing formal output directory. It persists the complete run to
`train.log`. To resume an interrupted run, keep the same `OUTPUT_DIR` and pass
its complete distributed state directory as `RESUME_STATE`.

An eight-GPU PutBack A/B probe compared the inherited gradient-checkpointed path
with checkpointing disabled. Disabling it fit only narrowly (observed peak about
70.2 GiB per 80 GiB GPU), changed later BF16 optimizer trajectories slightly,
and improved the three-step measured throughput only from 2.18 to 2.20 samples/s.
The formal launcher therefore retains gradient checkpointing: the measured gain
did not justify reduced memory headroom or numerical risk.

## 8-GPU end-to-end verification on 2026-08-11

The formal code path completed a real one-step BF16 FSDP smoke on machine 02,
including forward, backward, optimizer step, resumable state save, and portable
checkpoint save. The portable checkpoint validator reported step 1 with one
`layerwise_block_memory` tensor, 1,649 MoT tensors, and two proprio tensors.

```text
8 GPUs, FSDP full shard, official initialization, resume=null
checkpoint=/mnt/vepfs02/output/kevin_wang/memorywam/train/layerwise_block_memory_battery_smoke1_8gpu_20260811/checkpoints/weights/step_000001.pt
project tests after PutBack launcher closeout: 79 passed
```

The step-24k-based compressor-only run below is retained as an engineering smoke
and post-hoc compression ablation; it is not the formal paper mainline.

## Experiment contract

- Base code: `fastwam_memory_fullkv_continuousvae`
- Base weights: `memorywam_fullattention_putback_fsdp_unchunked_resume20k_to30k_save2k_seed42_cloud/checkpoints/weights/step_024000.pt`
- Observation cache: `fastwam_putback_continuous_episode_stride16_v4`
- Compressor input/output: four `[120,3072]` pre-DiT native blocks to one `[120,3072]` native block
- Query anchors: the newest input block
- Keys/values: all 480 input tokens with native 3D RoPE
- Initialization: copied VideoDiT block-0 attention and exact zero-gate Last-block behavior
- Phase scope: level-0 groups only; level-1 recursive carry remains disabled

Training compresses only completed four-block groups before the newest clean decision. The newest clean decision and noisy future tokens remain raw. `minimum_history_frames=5` guarantees every compressor-only optimizer step exercises at least one completed group.

Online inference commits the current raw cache for action prediction first. After the action call succeeds, a completed four-raw suffix is replaced with one native summary and rematerialized through every unchanged VideoDiT layer. Older retained K/V remains an immutable prefix.

## Verification on 2026-08-10

Project suite:

```text
PYTHONPATH=src pytest -q tests
33 passed in 6.10s
```

Real one-step smoke:

```text
8 GPUs, FSDP full shard, BF16, step-24000 base, v4 continuous cache
loss=0.2781
loss_action=0.2372
loss_video=0.0409
history_frames=14.5000
native_retained_history_units=5.1250
full_kv_video_tokens=615.0000
exit=0
```

Smoke output:

`/mnt/vepfs02/output/kevin_wang/memorywam/train/native_cache_consolidation_putback_smoke_20260810_r3`

Re-run command:

```bash
cd /mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_native_cache_consolidation
OUTPUT_DIR=/mnt/vepfs02/output/kevin_wang/memorywam/train/<new-output-name> \
  ./ops/train_putback_native_cache_smoke.sh
```

Running bare `pytest -q` also collects RoboTwin's vendored `code_gen/test_gen_code.py`, which requires an external `assets/objects/objaverse/list.json`. The maintained project suite is `pytest -q tests`; all 33 project tests pass.
# RoboTwin evaluation

Native-cache checkpoints must use `sim_robotwin_native_cache`; the legacy
`sim_robotwin_full_kv` config intentionally instantiates a model without the
compressor. The shared evaluation launcher is safe to invoke from
`actionmem-01`:

```bash
cd /mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_native_cache_consolidation
PREFLIGHT_ONLY=1 CHECKPOINT=/path/to/step_001000.pt \
  bash ops/eval_putback_native_cache.sh

CHECKPOINT=/path/to/step_001000.pt SEEDS=0 GPUS=2 EVAL_NUM_EPISODES=1 \
  bash ops/eval_putback_native_cache.sh
```

For a five-seed run, use matching seed and GPU lists, for example
`SEEDS=0,1,2,3,4 GPUS=0,1,2,3,4`. The launcher rejects malformed native-cache
checkpoints before model construction and writes a resolved run contract beside
the per-seed logs. Each successful replan emits
`FASTWAM_NATIVE_CACHE_METRICS` JSON containing cache structure, K/V token count,
and CUDA peak-memory counters.
