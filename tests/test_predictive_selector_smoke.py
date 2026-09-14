from __future__ import annotations

import torch

from fastwam.memory.dynamic_selection_manifest import freeze_selection_manifest
from scripts.smoke_putback_predictive_dynamic_selection import run_four_episode_smoke
from tests.test_dynamic_selection_manifest import HASHES, _config, _trace


def test_four_episode_training_runtime_smoke_passes_all_contracts():
    traces = {
        episode: {
            "observations": [
                {"frame": 16 + 4 * index, "residuals": torch.tensor([0.0, value])}
                for index, value in enumerate([0, 4, 2, 0, 0, 4, 2 + episode * 0.1])
            ],
            "terminal_frame": 44,
        }
        for episode in range(4)
    }
    manifest = freeze_selection_manifest(
        traces=traces, selector_config=_config(), source_hashes=HASHES,
        heldout_gate={"pass": True}, allow_single_dynamic_episode=True,
    )
    report = run_four_episode_smoke(
        manifest=manifest, traces=traces, rollout_fn=lambda: True
    )
    assert report["pass"] is True
    assert report["episodes"] == 4
    assert report["dataset_runtime_exact"] is True
    assert report["resume_next_loss_exact"] is True
    assert report["rmbench_rollout_completed"] is True
    assert report["maximum_queue_depth"] == 1
