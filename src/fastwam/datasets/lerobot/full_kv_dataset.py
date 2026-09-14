from __future__ import annotations

import bisect
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from .robot_video_dataset import RobotVideoDataset


SCHEMA_VERSION = "fastwam_full_kv_continuous_episode_vae_latents_v4"


def _as_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.reshape(-1)[0].item())
    return int(value)


def _decision_source_indices(
    episode_start: int,
    episode_end: int,
    replan_stride: int,
    minimum_history_frames: int = 1,
) -> list[int]:
    """Return every episode-local policy decision, including its padded tail."""
    episode_start = int(episode_start)
    episode_end = int(episode_end)
    replan_stride = int(replan_stride)
    minimum_history_frames = int(minimum_history_frames)
    if episode_end <= episode_start:
        raise ValueError("Episode end must be greater than episode start")
    if replan_stride <= 0:
        raise ValueError("Replan stride must be positive")
    if minimum_history_frames <= 0:
        raise ValueError("Minimum history frames must be positive")
    decisions = list(range(episode_start, episode_end, replan_stride))
    return decisions[minimum_history_frames - 1 :]


class FullKVObservationStore:
    """Lazy store of causal temporal-VAE latents at policy decision points."""

    def __init__(
        self,
        root: str | Path,
        *,
        expected_replan_stride: int,
        lru_episodes: int = 2,
    ):
        self.root = Path(root).expanduser().resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing full-KV manifest: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        metadata = manifest.get("metadata", {})
        expected = {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "replan_stride": int(expected_replan_stride),
            "temporal_subframes": 4,
            "temporal_subframe_stride": int(expected_replan_stride) // 4,
            "latent_shape": [48, 1, 24, 20],
            "latent_dtype": "bfloat16",
            "mosaic": "wrists_top_head_bottom_384x320",
            "color_contract": "simulator_rgb_preserved",
            "encoding": "continuous_episode_stride4_causal_vae",
        }
        mismatches = {
            key: (value, metadata.get(key))
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Incompatible full-KV latent cache: {mismatches}")
        episodes = manifest.get("episodes")
        if not isinstance(episodes, dict) or not episodes:
            raise ValueError("Full-KV manifest contains no episodes")
        self.episodes = {str(key): str(value) for key, value in episodes.items()}
        self.replan_stride = int(expected_replan_stride)
        self._lru_limit = max(int(lru_episodes), 1)
        self._lru: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()

    @staticmethod
    def _candidate_keys(sample: dict[str, Any]) -> list[str]:
        episode = _as_int(sample["episode_index"])
        dataset_name = Path(str(sample["dataset_name"])).name
        return [f"{dataset_name}/{episode}", str(episode)]

    def _episode_key(self, sample: dict[str, Any]) -> str:
        for key in self._candidate_keys(sample):
            if key in self.episodes:
                return key
        raise KeyError(
            f"No cached episode for {self._candidate_keys(sample)}"
        )

    def _load_episode(self, key: str) -> dict[str, torch.Tensor]:
        cached = self._lru.pop(key, None)
        if cached is not None:
            self._lru[key] = cached
            return cached
        path = self.root / self.episodes[key]
        payload = torch.load(path, map_location="cpu", weights_only=True)
        frame_indices = torch.as_tensor(payload["frame_indices"], dtype=torch.int64)
        latents = torch.as_tensor(payload["latents"], dtype=torch.bfloat16)
        expected_shape = (frame_indices.numel(), 48, 1, 24, 20)
        if tuple(latents.shape) != expected_shape:
            raise ValueError(
                f"{path}: expected latents {expected_shape}, got {tuple(latents.shape)}"
            )
        if frame_indices.ndim != 1:
            raise ValueError(f"{path}: frame_indices must be one-dimensional")
        if frame_indices.numel() and (
            int(frame_indices[0]) != 0
            or not bool(
                (frame_indices % self.replan_stride == 0).all().item()
            )
            or not bool((frame_indices[1:] > frame_indices[:-1]).all().item())
        ):
            raise ValueError(f"{path}: invalid decision-frame sequence")
        result = {"frame_indices": frame_indices, "latents": latents}
        self._lru[key] = result
        while len(self._lru) > self._lru_limit:
            self._lru.popitem(last=False)
        return result

    def strict_history(self, sample: dict[str, Any]) -> torch.Tensor:
        episode = self._load_episode(self._episode_key(sample))
        current_frame = _as_int(sample["frame_index"])
        frames = episode["frame_indices"]
        position = int(
            torch.searchsorted(
                frames,
                torch.tensor(current_frame, dtype=frames.dtype),
                right=False,
            ).item()
        )
        if position >= frames.numel() or int(frames[position]) != current_frame:
            raise KeyError(
                f"Current frame {current_frame} is absent from the latent cache"
            )
        # The cached latent at `current_frame` summarizes only observations
        # through the current decision point, so it is valid causal context.
        history_frames = frames[: position + 1]
        if history_frames.numel() and not bool(
            (history_frames <= current_frame).all().item()
        ):
            raise RuntimeError("Full-KV cache leaked current/future observations")
        return episode["latents"][: position + 1]


class FullKVRobotVideoDataset(RobotVideoDataset):
    """Official FastWAM samples augmented with strict-causal observation history."""

    def __init__(
        self,
        *args,
        full_kv_cache_path: str,
        replan_stride: int = 16,
        minimum_history_frames: int = 1,
        episode_indices: list[int] | None = None,
        **kwargs,
    ):
        kwargs["skip_padding_as_possible"] = False
        super().__init__(*args, **kwargs)
        self.replan_stride = int(replan_stride)
        self.minimum_history_frames = int(minimum_history_frames)
        if self.replan_stride <= 0:
            raise ValueError("`replan_stride` must be positive")
        if int(self.num_frames) - 1 != self.replan_stride:
            raise ValueError(
                "This first full-KV baseline requires action horizon == "
                f"replan stride, got horizon={int(self.num_frames)-1}, "
                f"stride={self.replan_stride}"
            )
        self.store = FullKVObservationStore(
            full_kv_cache_path,
            expected_replan_stride=self.replan_stride,
        )

        self.sample_indices: list[int] = []
        episode_index = self.lerobot_dataset.episode_data_index
        episode_starts = episode_index["from"].tolist()
        episode_ends = episode_index["to"].tolist()
        selected_episodes = (
            None
            if episode_indices is None
            else {int(episode) for episode in episode_indices}
        )
        if selected_episodes is not None:
            invalid = selected_episodes.difference(range(len(episode_starts)))
            if invalid:
                raise ValueError(f"Invalid episode indices: {sorted(invalid)}")
        for episode, (start, end) in enumerate(
            zip(episode_starts, episode_ends)
        ):
            if selected_episodes is not None and episode not in selected_episodes:
                continue
            # Include the episode-start decision.  Online inference begins
            # with exactly one clean anchor latent in the KV cache, so
            # excluding this sample creates a severe train/eval mismatch in
            # the first (and often trajectory-defining) action chunk.
            # Keep the terminal decision even when fewer than `horizon`
            # actions remain. LeRobot replicates the final observation/action
            # and marks the synthetic suffix in `image_is_pad` and
            # `action_is_pad`; the losses mask those entries. Dropping this
            # decision removes the expert's final gripper-release supervision.
            for source_index in _decision_source_indices(
                int(start),
                int(end),
                self.replan_stride,
                minimum_history_frames=self.minimum_history_frames,
            ):
                self.sample_indices.append(source_index)
        if not self.sample_indices:
            raise ValueError("No valid full-KV decision states were found")

    def __len__(self) -> int:
        return len(self.sample_indices)

    def _get(self, idx):
        source_index = self.sample_indices[int(idx)]
        sample = super()._get(source_index)
        if int(sample["idx"]) != source_index:
            raise RuntimeError("Dataset substituted a sample and broke history alignment")
        required = {"episode_index", "frame_index", "dataset_index"}
        missing = required.difference(sample)
        if missing:
            raise RuntimeError(
                "Official processor dropped full-KV provenance fields: "
                f"{sorted(missing)}"
            )
        dataset_index = _as_int(sample["dataset_index"])
        dataset_names = self.lerobot_dataset.multi_dataset.ds_names
        if dataset_index < 0 or dataset_index >= len(dataset_names):
            raise RuntimeError(f"Invalid dataset index {dataset_index}")
        sample["dataset_name"] = dataset_names[dataset_index]
        sample["history_latents"] = self.store.strict_history(sample)
        return sample

    def __getitem__(self, idx):
        # History alignment is a correctness boundary: never silently replace
        # a failed sample with a random episode.
        return self._get(idx)
