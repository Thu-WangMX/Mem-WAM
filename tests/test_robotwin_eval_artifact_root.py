from pathlib import Path

from experiments.robotwin.eval_robotwin_single import (
    resolve_robotwin_artifact_root,
)


def test_artifact_root_defaults_to_legacy_repository_location(tmp_path):
    assert resolve_robotwin_artifact_root(None, tmp_path) == (
        tmp_path / "evaluate_results" / "robotwin"
    )


def test_artifact_root_preserves_absolute_vepfs01_mirror(tmp_path):
    mirror = Path(
        "/mnt/vepfs01/output/kevin_wang/memorywam/eval_runs/wrist_event_5k"
    )

    assert resolve_robotwin_artifact_root(str(mirror), tmp_path) == mirror
