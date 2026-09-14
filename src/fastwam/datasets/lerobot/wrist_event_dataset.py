"""PutBack full-KV dataset with one frozen wrist-event segment manifest."""

from __future__ import annotations

from .dynamic_surprise_dataset import attach_dynamic_surprise_groups
from .full_kv_dataset import FullKVRobotVideoDataset
from fastwam.memory.wrist_event import WristEventManifestStore


class WristEventRobotVideoDataset(FullKVRobotVideoDataset):
    def __init__(
        self,
        *args,
        wrist_event_manifest_path: str,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.wrist_event_manifest_store = WristEventManifestStore(
            wrist_event_manifest_path,
        )

    def _get(self, idx):
        sample = super()._get(idx)
        return attach_dynamic_surprise_groups(
            sample,
            self.wrist_event_manifest_store,
            replan_stride=self.replan_stride,
        )
