#!/usr/bin/env python3
"""Audit completed rate-comparison rollouts without rerunning RoboTwin."""

from __future__ import annotations

import json
from pathlib import Path
import sys


def audit_log(path: Path, rate_mode: str) -> str:
    selector = None
    events = []
    arrivals = []
    for raw in path.read_text(errors="replace").splitlines():
        for tag, target in (
            ("FASTWAM_CONTROL_INFORMATION_SELECTOR ", "selector"),
            ("FASTWAM_CONTROL_INFORMATION_DETECTOR ", "detector"),
            ("FASTWAM_CONTROL_INFORMATION_PLANNING_ARRIVAL ", "arrival"),
        ):
            if tag not in raw:
                continue
            payload = json.loads(raw.split(tag, 1)[1])
            if target == "selector":
                selector = payload
            elif target == "detector" and payload.get("event") is not None:
                events.append(payload["event"])
            elif target == "arrival":
                arrivals.append(payload)
    assert selector is not None, f"{path}: missing selector record"
    assert selector["runtime_version"] == "v3"
    assert selector["source"] == "frozen_initialization_wam"
    assert selector["strict_online"] is True
    assert selector["forward_alignment"] is True
    assert selector["uses_old_dynamic_surprise"] is False
    assert selector["uses_gripper_hard_trigger"] is False
    assert events, f"{path}: no closed events"

    previous_end = 0
    learned = forced = 0
    for event in events:
        start = int(event["group_start"])
        end = int(event["group_end"])
        confirmation = int(event["confirmation_frame"])
        reason = event["reason"]
        assert start == previous_end, (path, start, previous_end)
        assert end == confirmation, (path, end, confirmation)
        assert 32 <= end - start <= 96, (path, start, end)
        assert end % 4 == 0
        if reason == "forced_maximum":
            forced += 1
            assert end - start == 96
        else:
            learned += 1
            assert reason == "segment_relative_control_information"
        expected_arrival = ((confirmation + 15) // 16) * 16
        matching = [
            item
            for item in arrivals
            if int(item["frame"]) == expected_arrival
            and item.get("close_range") is not None
        ]
        assert matching, (path, confirmation, expected_arrival)
        arrival = matching[-1]
        close_start, close_stop = (int(value) for value in arrival["close_range"])
        assert close_stop > close_start
        assert close_start == (start + 15) // 16, (path, start, close_start)
        assert close_stop == (end + 15) // 16, (path, end, close_stop)
        expected_tokens = 8 * (close_stop - close_start)
        if rate_mode == "event" and reason == "forced_maximum":
            expected_tokens = 8 * ((close_stop - close_start + 1) // 2)
        assert int(arrival.get("close_token_count", -1)) == expected_tokens, (
            path,
            reason,
            start,
            end,
            arrival,
        )
        previous_end = end
    result = "".join(
        line for line in path.read_text(errors="replace").splitlines()
        if "Success rate:" in line
    )
    result = result.splitlines()[-1] if result else "missing_result"
    return f"{path.stem} {result} online_audit=ok events={len(events)} learned={learned} forced={forced}"


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    rate_mode = sys.argv[2]
    rows = [audit_log(path, rate_mode) for path in sorted(root.glob("scene*_policy*.log"))]
    if len(rows) != 4:
        raise SystemExit(f"expected 4 scene logs under {root}, found {len(rows)}")
    print(f"rate_mode={rate_mode} strict_online=true forward_alignment=true")
    print("\n".join(rows))
    print("online_audit_status=ok")


if __name__ == "__main__":
    main()
