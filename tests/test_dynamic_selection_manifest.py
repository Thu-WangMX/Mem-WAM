from __future__ import annotations

import pytest
import torch

from fastwam.memory.dynamic_selection_manifest import freeze_selection_manifest


HASHES = {
    "initialization_fingerprint": "a" * 64,
    "code_commit": "b" * 40,
    "predictor_sha256": "c" * 64,
    "pca_sha256": "d" * 64,
    "stats_sha256": "e" * 64,
    "annotation_sha256": "f" * 64,
}


def _trace(values):
    return [
        {"frame": 4 * (index + 1), "residuals": torch.tensor([0.0, value])}
        for index, value in enumerate(values)
    ]


def _config():
    return {
        "stats": {"median": [0.0, 0.0], "mad_scale": [1.0, 1.0]},
        "weights": [0.0, 1.0],
        "high_threshold": 3.0,
        "low_threshold": 1.0,
        "detector_stride": 4,
        "min_units": 2,
        "max_units": 20,
        "nms_units": 2,
        "initial_group_start": 0,
    }


def test_freeze_requires_final_gate_hashes_and_true_dynamic_coverage():
    traces = {
        0: {"observations": _trace([0, 4, 2, 0, 0, 4, 2]), "terminal_frame": 32},
        1: {"observations": _trace([0, 0, 4, 2, 0, 0, 0, 4, 2]), "terminal_frame": 40},
    }
    manifest = freeze_selection_manifest(
        traces=traces,
        selector_config=_config(),
        source_hashes=HASHES,
        heldout_gate={"pass": True},
    )
    assert manifest["schema_version"] == "putback_dynamic_selection_manifest_v1"
    assert set(manifest["episodes"]) == {"0", "1"}
    lengths = set()
    for episode in manifest["episodes"].values():
        assert episode["groups"][0]["start_frame"] == 0
        assert episode["groups"][-1]["end_frame"] == episode["terminal_frame"]
        for group in episode["groups"]:
            assert set(group) == {
                "start_frame", "end_frame", "detector_units", "confirmation_frame",
                "peak_frame", "reason", "score", "selected_streams", "source_hashes",
            }
            lengths.add(group["detector_units"])
    assert len(lengths) >= 2

    with pytest.raises(ValueError, match="held-out reliability gate"):
        freeze_selection_manifest(
            traces=traces, selector_config=_config(), source_hashes=HASHES,
            heldout_gate={"pass": False},
        )
    with pytest.raises(ValueError, match="source hashes"):
        freeze_selection_manifest(
            traces=traces, selector_config=_config(), source_hashes={**HASHES, "extra": "x"},
            heldout_gate={"pass": True},
        )


def test_freeze_rejects_missing_duplicated_frames_and_fixed_length_degeneration():
    for bad in (
        [{"frame": 4, "residuals": torch.zeros(2)}, {"frame": 12, "residuals": torch.zeros(2)}],
        [{"frame": 4, "residuals": torch.zeros(2)}, {"frame": 4, "residuals": torch.zeros(2)}],
    ):
        with pytest.raises(ValueError, match="consecutive"):
            freeze_selection_manifest(
                traces={0: {"observations": bad, "terminal_frame": 16}},
                selector_config=_config(), source_hashes=HASHES,
                heldout_gate={"pass": True},
            )

    with pytest.raises(ValueError, match="fixed-length degeneration"):
        freeze_selection_manifest(
            traces={0: {"observations": _trace([0] * 7), "terminal_frame": 32}},
            selector_config=_config(), source_hashes=HASHES,
            heldout_gate={"pass": True},
        )
