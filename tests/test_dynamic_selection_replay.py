from __future__ import annotations

from tests.test_dynamic_selection_manifest import HASHES, _config, _trace
from fastwam.memory.dynamic_selection_manifest import freeze_selection_manifest
from scripts.verify_putback_dynamic_selection_replay import verify_replay


def test_online_replay_is_exact_and_future_append_cannot_change_prior_groups():
    base = {0: {"observations": _trace([0, 4, 2, 0, 0, 4, 2]), "terminal_frame": 32}}
    manifest = freeze_selection_manifest(
        traces=base, selector_config=_config(), source_hashes=HASHES,
        heldout_gate={"pass": True}, allow_single_dynamic_episode=True,
    )
    report = verify_replay(manifest=manifest, traces=base)
    assert report == {"episodes": 1, "groups": len(manifest["episodes"]["0"]["groups"]), "exact": True}

    extended_observations = base[0]["observations"] + _trace([0, 4, 2])
    # Re-index appended observations so they continue, rather than replaying at frame 4.
    for offset, row in enumerate(extended_observations[len(base[0]["observations"]):], start=8):
        row["frame"] = offset * 4
    extended = {0: {"observations": extended_observations, "terminal_frame": 44}}
    future_manifest = freeze_selection_manifest(
        traces=extended, selector_config=_config(), source_hashes=HASHES,
        heldout_gate={"pass": True}, allow_single_dynamic_episode=True,
    )
    prior = manifest["episodes"]["0"]["groups"][:-1]
    assert future_manifest["episodes"]["0"]["groups"][: len(prior)] == prior

