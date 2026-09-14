# Native Cache Consolidation Design

## Scope

Build the first paper-facing experiment on top of the verified FullKV continuous-VAE codebase. Four completed native VideoDiT decision blocks are consolidated into one native block before layer 0. The output remains one `tokens_per_frame x hidden_dim` block and is rematerialized through the unchanged VideoDiT to obtain valid K/V at every layer.

This phase implements only level-0 `4 -> 1` consolidation. It deliberately keeps the state format (`tokens`, `endpoint`, `span`, and `level`) compatible with later base-4 recursive carries, but it does not recursively merge level-1 summaries yet.

## Source and isolation

- Source of truth: `/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_memory_fullkv_continuousvae`
- Long-lived isolated destination: `/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_native_cache_consolidation`
- Verified starting checkpoint family: `memorywam_fullattention_putback_fsdp_unchunked_resume20k_to30k_save2k_seed42_cloud`, with step 24,000 used as the known 50/50 FullKV reference.
- The older `fastwam_full_history_stage1_v1` is not a source tree. Only its proven engineering patterns—deep-copying native attention weights and explicit checkpoint registration—may be reused.

## Compressor

For native pre-DiT blocks `X[0:4]` with shape `[B,4,N,D]`, the final block supplies `N` spatial anchor queries and all four blocks supply `4N` keys and values. Q/K/V/O and Q/K normalization are deep-copied from VideoDiT block 0. Compressor Q and K receive the original 3D RoPE coordinates, so global attention remains spatially and temporally identifiable.

Each source block receives an ordinal embedding, a level embedding, and a projected `log2(span)` scalar. The output is:

`M = X3 + alpha_max * tanh(raw_alpha) * O(Attention(Q(X3), K(X0..X3), V(X0..X3)))`

`raw_alpha` starts at zero. Therefore a newly initialized compressor is exactly the Last-block baseline, while its first optimizer update can open the residual path. The output inherits the fourth block's endpoint RoPE coordinate and has `span=sum(input_spans)` and `level=max(input_levels)+1`.

## Training path

The existing continuous-VAE data and `pre_dit` path remain unchanged. After `pre_dit`, completed groups in the clean-history prefix are compressed. The newest clean decision stays raw. For history length `H`, only `floor((H-1)/4)*4` oldest frames are grouped; remaining clean frames and all noisy future frames stay raw.

The MoT attention mask is rebuilt over retained clean units plus noisy future units and action tokens. Future video decoding slices only future tokens and uses a future-only post-DiT state, avoiding dependence on the original uncompressed clean grid length. The objective remains the existing video flow-matching loss plus action flow-matching loss. No teacher or distillation loss is introduced.

The first run freezes the FullKV backbone and trains only the compressor. Loading a FullKV checkpoint that lacks compressor weights first loads MoT weights and then initializes the compressor copy from the now-loaded VideoDiT block 0. Native-consolidation checkpoints save both MoT and compressor state.

## Online inference state

Online state contains the layer-wise K/V cache plus retained native blocks. Each block records pre-layer tokens, endpoint frame index, temporal span, and level. Current observations are appended normally and used for action prediction before any rewrite.

After the fourth raw level-0 block has been used, the four raw blocks are compressed. Their K/V suffix is removed, the older retained K/V prefix is preserved, and the summary is prefetched through every unchanged VideoDiT layer against that prefix. The next decision therefore sees one valid summary cache block instead of four raw cache blocks. Only level-0 groups are merged in this phase.

## Verification gates

1. The copied source baseline passes its current test suite.
2. Unit tests prove zero-gate identity, output shape, source-wide gradients, metadata conservation, correct grouping, and cache shrinkage from `4N` to `N`.
3. Checkpoint tests prove a FullKV checkpoint initializes compressor weights from the loaded block 0 and a native checkpoint round-trips compressor weights.
4. A tiny-model CPU/GPU integration test runs training forward/backward and online consolidation.
5. The real 8-GPU one-step PutBack smoke config loads the verified FullKV checkpoint, performs one optimizer step, and emits finite loss/gradient metrics.

