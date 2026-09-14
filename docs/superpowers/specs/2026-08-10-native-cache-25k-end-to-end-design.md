# Native Cache 25k End-to-End Training Design

## Goal

Train the native `4 -> 1` cache-consolidation architecture from the same initialization and with the same PutBack protocol as the verified FullKV baseline. The formal run ends at step 25,000 and emits portable checkpoints at steps 1,000, 5,000, 10,000, 15,000, 20,000, and 25,000.

## Common initialization and fairness contract

- Construct VideoDiT from `Wan-AI/Wan2.2-TI2V-5B` pretrained weights.
- Construct ActionDiT from `ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`.
- Do not load a PutBack-trained FullKV checkpoint (`resume: null`).
- Initialize the compressor from VideoDiT block 0 with a zero residual gate.
- Train the same VideoDiT, ActionDiT, and proprio parameters as FullKV, plus the compressor.
- Reuse the FullKV optimizer and data contract: AdamW, LR `2e-4`, betas `(0.9, 0.95)`, weight decay `0.01`, global batch 8, BF16, seed 42, logit-normal timestep sampling, and the complete 50-episode PutBack dataset.

## Checkpoint schedule

The trainer accepts `save_steps: [1000]` in addition to `save_every: 5000`. A single uninterrupted 25k process is required so the LR scheduler, optimizer, sampler, and RNG state are continuous. The resulting save set is exactly `1000, 5000, 10000, 15000, 20000, 25000`; final-save deduplication prevents a second save at step 25k.

## Runtime contract

The run uses eight GPUs under the existing FSDP full-shard configuration. It trains from the verified V4 continuous-episode cache and refuses to overwrite an existing output directory. Before launch it validates the 50-episode dataset, model initialization, caches, and all required paths.

## Verification

- Unit-test explicit and periodic checkpoint selection.
- Compose the Hydra task and assert `resume=null`, `max_steps=25000`, `save_steps=[1000]`, `save_every=5000`, `native_cache_train_mode=full`, and logit-normal sampling.
- Run the maintained test suite and a one-step real FSDP smoke from fresh initialization before starting the 25k process.
- After launch, verify the log says `Starting training with max_steps=25000`, all eight GPUs are active, and the first losses are finite.

