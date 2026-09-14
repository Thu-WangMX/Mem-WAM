# FastWAM FullKV with continuous episode VAE history

## Purpose

This variant changes only how clean historical observation latents are
created. It keeps the production FastWAM MoT experts, terminal-padding fix,
full-history Transformer attention, action alignment, losses, optimizer, and
RMBench evaluation contract unchanged.

## Difference from the decision-window baseline

The previous cache independently encoded `[t-16, t-12, t-8, t-4, t]` at every
decision and retained the second Wan latent. The VAE feature cache therefore
restarted at each decision boundary.

This variant samples the episode at simulator frames `0, 4, 8, ...` and sends
that complete sequence through one causal Wan VAE call. Wan's 4x temporal
compression yields one latent at decision frames `0, 16, 32, ...`. Those
latents are consumed by the unchanged FastWAM FullKV path.

Online evaluation retains the same stride-4 RGB prefix for the whole episode
and re-encodes the complete causal prefix at each decision. This is exact but
slower than a future persistent streaming-VAE implementation.

## First experiment contract

- task: `swap_blocks`
- eight GPUs, per-rank batch size 1
- fresh official FastWAM/ActionDiT initialization
- 5,000 optimization steps
- checkpoints: step 3,000 and step 5,000 only
- seed 42
- video/action timestep sampling: `logit_normal`
- Action RoPE: `memorywam`

The cache schema is
`fastwam_full_kv_continuous_episode_vae_latents_v4`, preventing accidental
reuse of independent-window latent files.
