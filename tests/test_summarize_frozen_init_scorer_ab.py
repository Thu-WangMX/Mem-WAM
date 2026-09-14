import json

import pytest

from scripts.summarize_frozen_init_scorer_ab import parse_run


def _write_log(root, scene, policy, *, closed, center, press):
    path = root / f"scene{scene}_policy{policy}.log"
    rows = [
        "FASTWAM_DYNAMIC_SURPRISE "
        + json.dumps(
            {
                "arrival": stop,
                "close_range": [start, stop],
                "reason": reason,
                "score": 0.5,
                "threshold": 0.4,
            }
        )
        for start, stop, reason in closed
    ]
    rows.append(
        "RMBENCH_EVAL_DIAGNOSTICS "
        + json.dumps(
            {
                "official_success": False,
                "scene_seed": scene,
                "task_state": {
                    "check_block_in_center": center,
                    "press_cnt": press,
                    "stage_id": 0,
                },
            }
        )
    )
    path.write_text("\n".join(rows) + "\n")


def test_parse_run_returns_hand_derived_segments_and_task_state(tmp_path):
    _write_log(
        tmp_path,
        100000,
        1000,
        closed=[(0, 4, "surprise"), (4, 12, "max_length")],
        center=True,
        press=0,
    )

    parsed = parse_run(tmp_path, {100000: 1000})

    assert parsed[100000] == {
        "policy_seed": 1000,
        "official_success": False,
        "stage_id": 0,
        "press_cnt": 0,
        "check_block_in_center": True,
        "segment_lengths": [4, 8],
        "trigger_reasons": ["surprise", "max_length"],
    }


def test_parse_run_rejects_missing_scene(tmp_path):
    with pytest.raises(ValueError, match="missing log"):
        parse_run(tmp_path, {100000: 1000})
