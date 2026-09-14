"""Compute normalization statistics through the official FastWAM data path."""

from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict

from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers


register_default_resolvers()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    output_dir = Path(cfg.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(output_dir)

    dataset_cfg = cfg.data.train
    with open_dict(dataset_cfg):
        dataset_cfg._target_ = (
            "fastwam.datasets.lerobot.robot_video_dataset.RobotVideoDataset"
        )
        dataset_cfg.pretrained_norm_stats = None
        # These arguments belong only to the full-KV wrapper.  Statistics must
        # come from the same official action/state processor, independently of
        # any latent cache.
        del dataset_cfg.full_kv_cache_path
        del dataset_cfg.replan_stride

    instantiate(dataset_cfg)
    stats_path = output_dir / "dataset_stats.json"
    if not stats_path.is_file():
        raise RuntimeError(f"Statistics were not written to {stats_path}")
    print(stats_path, flush=True)


if __name__ == "__main__":
    main()
