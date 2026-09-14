"""Small real-data check for the causal two-anchor Helios contract."""
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


root = Path(__file__).resolve().parents[1]
with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
    cfg = compose(config_name="train", overrides=["task=rmbench_helios"])
dataset = instantiate(cfg.data.train)
samples = [dataset._get(i) for i in range(3)]
assert all(int(sample["episode_index"]) == int(samples[0]["episode_index"]) for sample in samples)
assert samples[0]["anchor_valid"].tolist() == [True, False]
assert samples[1]["anchor_valid"].tolist() == [True, True]
assert torch.count_nonzero(samples[0]["anchor_latents"][1]) == 0
torch.testing.assert_close(samples[0]["anchor_latents"][0], samples[1]["anchor_latents"][0])
torch.testing.assert_close(samples[1]["anchor_latents"], samples[2]["anchor_latents"])
torch.testing.assert_close(samples[0]["anchor_latents"][0], samples[0]["history_latents"][-1])
torch.testing.assert_close(samples[1]["anchor_latents"][1], samples[1]["history_latents"][-1])
print({
    "dataset_length": len(dataset),
    "anchor_shape": tuple(samples[2]["anchor_latents"].shape),
    "anchor_valid_start": samples[0]["anchor_valid"].tolist(),
    "anchor_valid_after_second_decision": samples[1]["anchor_valid"].tolist(),
    "first_two_anchors_stable": True,
})
