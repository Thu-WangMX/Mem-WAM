# FastWAM RMBench memory comparison

Working copy on ecs-a100:
`/mnt/vepfs01/output/spidy.wang/fastwam-memory-comparison`

Source: `/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_rearrange_memory_ablations_k8`.
The colleague has multiple task and experiment forks, including a separate nine-task training fork.
This copy starts from the validated Rearrange Blocks ablation fork. Source files, datasets and the
colleague's environment are never modified. `migration/source_manifest.json` records source hashes.
The copy excludes run outputs, caches, Git internals and symlinks; simulator assets are not bundled.

## Shared experiment contract

- Same LeRobot RGB data, 384×320 mosaic (wrists above, head below), 14-dimensional actions/proprioception.
- Same task-specific `dataset_stats.json`, z-score normalization, no per-step action normalization.
- Every episode-local decision at frames 0,16,32,… is retained; tails use padding and the original loss masks.
- RGB stride 4, action horizon/execution 16; five RGB frames give current + one future latent.
- Clean history comes from the colleague's continuous episode causal VAE cache. Online inference encodes
  the complete stride-4 observed prefix with the same VAE, then selects the history needed by the backend.
- As in the colleague's implementation, the supervised future latent comes from independently encoding
  the current five-frame training clip. This choice is common to both backends.
- The historical cache is generated directly from original HDF5 simulator images, while auxiliary
  future-video targets use the converted LeRobot clips. A history parity check must replay original
  HDF5 images through the deployment preprocessor; re-encoding the LeRobot MP4 is a different input.
- Same video/action backbones, initialization, proprioception conditioning, action RoPE basis, loss weights,
  logit-normal timestep sampling, video/action shifts 5/1, and history conditioning noise augmentation.

## Switches

`MEMORY_BACKEND=helios` selects bounded latent history `[long=16, middle=2, current=1]`.
Startup missing history is left-padded with zero latents, consistently in training and deployment.
Trainable convolutions use kernels/strides `(4,8,8)`, `(2,4,4)`, `(1,2,2)`; long/middle weights are
initialized from the pretrained patch convolution. RoPE uses local frame coordinates 0…18, with
spatial/temporal pooling matching the compressed token grid, including right padding at spatial edges.
At latent resolution 24×20 the three branches give 36 + 30 + 120 = 186 history tokens.
Future tokens cannot influence history or action. Historical Transformer features are recomputed each
decision; their KV is reused only across action denoising steps for that decision. There is no persistent
cross-decision Helios KV or learned key amplification in this version.

`MEMORY_BACKEND=memorywam` selects the colleague's existing fixed-length L4/K8 layerwise memory:
2 anchors, recent 4 frames and 8 tokens per closed segment. The original dataset/model implementations
are retained. This is the colleague's MemoryWAM-like variant, not a claim of an exact paper reproduction.

`MEMORY_BACKEND=fullkv` retains the original full-history KV baseline.

The original two backends require microbatch 1 because of variable history/segmentation. Their wrapper
defaults to GA=16; Helios defaults to microbatch 16, GA=1. With eight GPUs both have effective batch 128.
These settings match effective batch, not wall time or the colleague's original BS1/GA1 run.
For a strict microbatch-matched comparison set `BATCH_SIZE=1 GRAD_ACCUM=16` for every backend.
All launch with DeepSpeed ZeRO1; FSDP export is not supported for the new Helios checkpoint format.

## Environment and training

The wrapper uses `/mnt/vepfs01/output/spidy.wang/runtime/starwam-libero/venv` plus
`runtime/python` for additional dependencies. It does not install into either existing environment.
Caches, outputs and compiled extensions are placed in this project. Pretrained weights and data are read
from the colleague's existing shared paths with model/data downloads disabled.

```bash
cd /mnt/vepfs01/output/spidy.wang/fastwam-memory-comparison
MEMORY_BACKEND=helios RMBENCH_TASK=rearrange_blocks \
  NUM_GPUS=8 BATCH_SIZE=16 GRAD_ACCUM=1 NUM_EPOCHS=500 \
  bash scripts/train_comparison.sh --config

# Run this inside the allocated eight-GPU queue job; --config above starts no training.
MEMORY_BACKEND=helios RMBENCH_TASK=rearrange_blocks \
  NUM_GPUS=8 BATCH_SIZE=16 GRAD_ACCUM=1 NUM_EPOCHS=500 \
  bash scripts/train_comparison.sh

MEMORY_BACKEND=memorywam RMBENCH_TASK=rearrange_blocks \
  NUM_GPUS=8 BATCH_SIZE=1 GRAD_ACCUM=16 NUM_EPOCHS=500 \
  bash scripts/train_comparison.sh
```

Training defaults to `NUM_EPOCHS=500`. If `NUM_EPOCHS` and `MAX_STEPS` are both set, epoch scheduling
wins; unset `NUM_EPOCHS` to request an explicit `MAX_STEPS` run. Weight checkpoints are saved every
2,000 optimizer steps and once at the final step. `WANDB_ENABLED=true` enables W&B using credentials already exported by the caller.
`OUTPUT_DIR`, `COMPARISON_VENV`, `RMBENCH_TASK_ROOT` and the four data-path variables in
`scripts/comparison_env.sh` can override paths. Other tasks must have matching LeRobot/stats/text/continuous
cache assets; setting a task name does not synthesize or download missing data.

## Inference

The existing RoboTwin policy maintains the observed RGB prefix, normalizes proprioception, executes the
16-action chunk, denormalizes outputs and resets episode state. Helios plugs into its `infer_action`
interface and returns no persistent KV state. The evaluation config selects the same backend in the child
policy process. Backend/history/scheduler/RoPE contracts are checked when loading a Helios checkpoint.
Evaluation defaults to the normalization stats saved alongside that training run's checkpoint; only
set `EVAL_STATS_PATH` when explicitly overriding this snapshot.
Old Helios/RoboCasa checkpoints are not compatible with this new shared-pipeline experiment.

```bash
MEMORY_BACKEND=helios RMBENCH_TASK=rearrange_blocks \
  CHECKPOINT=/absolute/path/to/new/helios/checkpoint.pt \
  RMBENCH_EVAL_ROOT=/mnt/vepfs01/output/spidy.wang/your-own-RMBench-copy \
  bash scripts/eval_comparison.sh
```

Simulation needs a separate working RMBench installation and its dependencies. The wrapper refuses a
simulator directory outside your own output directory because the evaluation harness writes artifacts.
Successful numerical inference is not evidence of task success; closed-loop success requires a newly
trained checkpoint and simulator evaluation.

## Verification artifacts

- `tests/test_helios_contract.py`: real small MoT backward with batch 16, no future leakage,
  train/prefill equality, deterministic inference, checkpoint isolation, pretrained backbone transfer.
- `scripts/check_comparison_data.py`: samples spread across the real dataset, original preprocessing
  equality and default batch-16 collation; calls `_get` to disallow fallback to a random sample on errors.
- `scripts/smoke_helios.py`: one forward/backward with real pretrained weights, causal VAE prefix replay
  against the cached history, and a two-denoising-step action inference. No optimizer/full training run.
- `migration/*.log` and JSON files record executed checks and results.

Confirmed on ecs-a100: all six small-model tests passed; three train/deployment config combinations
passed; the real Rearrange Blocks dataset contains 1,281 decision windows, and a batch of 16 samples
spanning the dataset matched original images/actions/proprioception/masks exactly. Startup histories
and padded tails were included. Source hash verification found zero changes across 1,258 source files.

Real pretrained 5.00B video + 1.02B action model smoke test also passed on GPU 3: batch 16 forward/backward,
finite nonzero long/middle compressor gradients, 186 history tokens, and action inference output `[16,14]`.
Peak allocated memory was 31.31 GiB (reserved 35.01 GiB), excluding optimizer states and distributed
communication buffers; this is not an eight-GPU ZeRO1 training memory measurement. The VAE weight hash
matched the cache manifest. Replaying raw HDF5 frames 0,4,…,32 through the actual online image preprocessor
produced exactly the cached latents for decisions 0/16/32 (max and mean absolute error 0).

The earlier failed LeRobot-video replay check is retained as `migration/smoke_lerobot_replay_diagnostic.log`;
it used a different image source from the historical cache. The corrected successful run is recorded in
`migration/smoke_helios.json`. No full training run or closed-loop simulator success-rate evaluation was launched.
