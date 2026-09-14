from __future__ import annotations

import torch

from scripts.freeze_putback_control_information_manifest import freeze_episode


def _statistics():
    return {
        "schema_version": "putback_control_information_statistics_v1",
        "episodes": list(range(30)),
        "phases": [0, 4, 8, 12],
        "depth_cap": 4,
        "median": torch.zeros(4, 4),
        "mad_scale": torch.ones(4, 4),
    }


def test_freezer_uses_shared_state_and_builds_complete_planning_partition():
    frames = torch.arange(16, 68, 4)
    trace = {
        "frame_indices": frames,
        "information": torch.tensor(
            [4.0 if int(frame) in {20, 48} else 0.0 for frame in frames]
        ),
    }
    payload = freeze_episode(
        episode=2,
        trace=trace,
        episode_length=68,
        statistics=_statistics(),
        selector_config={
            "threshold": 3.0,
            "drift": 0.5,
            "decay": 0.9,
            "detector_stride": 4,
            "min_units": 4,
            "max_units": 8,
            "initial_group_start": 0,
        },
    )

    assert payload["episode"] == 2
    assert payload["boundaries"][0] == 0
    assert payload["boundaries"][-1] == payload["decision_count"]
    assert payload["retroactive_boundary_count"] == 0
    assert set(payload["reasons"].values()) <= {
        "start",
        "counterfactual_control_information",
        "forced_maximum",
        "terminal_tail",
        "end",
    }
