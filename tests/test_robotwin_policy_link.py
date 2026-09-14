from pathlib import Path

from experiments.robotwin.eval_robotwin_single import _ensure_policy_symlink


def test_policy_link_uses_isolated_runtime_name(tmp_path: Path):
    robotwin_root = tmp_path / "RoboTwin"
    (robotwin_root / "policy").mkdir(parents=True)
    policy_source = tmp_path / "fastwam_policy"
    policy_source.mkdir()

    link = _ensure_policy_symlink(
        robotwin_root=robotwin_root,
        policy_source_dir=policy_source,
        policy_name="fastwam_native_cache_policy",
    )

    assert link.name == "fastwam_native_cache_policy"
    assert link.is_symlink()
    assert link.resolve() == policy_source.resolve()

