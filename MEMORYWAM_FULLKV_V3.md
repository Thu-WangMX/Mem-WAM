# MemoryWAM-aligned FastWAM Full-KV v3

## Goal

This baseline starts from the official FastWAM architecture and matches the
published MemoryWAM Put Back Block training and deployment contract. The only
intended memory-method difference is that every clean observation K/V is kept:
there are no gist tokens, anchor/recent selection, compression, sliding
window, or eviction.

## MemoryWAM contract retained

- Wan2.2-TI2V-5B VideoDiT and a separate 1B ActionDiT.
- Official ActionDiT hidden-dimension interpolation initialization.
- Three-camera 384x320 mosaic: wrists on top, head camera on the bottom.
- 14D absolute dual-arm joint state and next-step joint targets.
- Sixteen actions per decision and four temporal subframes per action chunk.
- Causal five-image VAE encoding after the first decision.
- Shared VideoDiT 3D RoPE basis for video and action tokens.
- Video/action flow-matching shifts of 5.0/1.0.
- Clean-latent Gaussian mixing augmentation on the video side.
- Video/action loss weights of 1.0/1.0.
- AdamW, LR 2e-4, weight decay 0.01, betas (0.9, 0.95).
- BF16, gradient clipping 1.0, eight GPUs, per-GPU batch size 1.
- Fifty action denoising steps, no video generation during deployment.
- No action-history cache for Put Back Block.

## v3 correctness boundary

RMBench stores simulator RGB arrays by passing them directly to OpenCV JPEG
encoding. Decoding preserves the numeric simulator RGB channel order; applying
an additional BGR-to-RGB conversion swaps red and blue.

Version 3 therefore requires all of the following:

1. a corrected LeRobot dataset whose build manifest declares
   `color_contract=simulator_rgb_preserved`;
2. a passing raw-HDF5-versus-video RGB validation report;
3. a newly generated latent manifest with schema
   `fastwam_full_kv_temporal_observation_latents_v3`;
4. a latent manifest declaring the same RGB contract and
   `wrists_top_head_bottom_384x320` mosaic;
5. normalization statistics recomputed from the corrected dataset;
6. identical seen instruction type for the first training/evaluation
   reproduction.

Old `rmbench_lerobot_v21` data, v2 latent caches, and checkpoints trained from
them are incompatible and must not be resumed.

## Commands

Prepare and validate the corrected assets:

```bash
bash ops/build_putback_rgb_v3.sh
bash ops/prepare_putback_memorywam_fullkv_v3.sh
```

Train from the official initialization for 5000 steps:

```bash
bash ops/train_putback_memorywam_fullkv_5k_v3.sh
```

The training entrypoint saves a periodic weight checkpoint at step 3000 and a
final weight checkpoint at step 5000. It does not save optimizer state.

Evaluate five seeds by explicitly selecting either checkpoint:

```bash
CHECKPOINT=/path/to/step_003000.pt \
RUN_TAG=memorywam_fullkv_v3_step3000 \
bash ops/eval_putback_five_seed.sh
```

The same command applies to `step_005000.pt`. A run is not promoted unless its
configuration, dataset manifest, RGB report, latent manifest, initialization
path, and per-seed results are retained together.
