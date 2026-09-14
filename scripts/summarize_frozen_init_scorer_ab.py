#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_run(root: Path, expected_pairs: dict[int, int]) -> dict[int, dict]:
    result = {}
    for scene, policy in expected_pairs.items():
        matches = list(root.glob(f"scene{scene}_policy{policy}.log"))
        if len(matches) != 1:
            raise ValueError(
                f"missing log or duplicate logs for scene {scene}, policy {policy}"
            )
        boundaries = []
        diagnostic = None
        for line in matches[0].read_text(errors="replace").splitlines():
            if line.startswith("FASTWAM_DYNAMIC_SURPRISE "):
                row = json.loads(line.split(" ", 1)[1])
                if row.get("close_range") is not None:
                    boundaries.append(row)
            elif line.startswith("RMBENCH_EVAL_DIAGNOSTICS "):
                if diagnostic is not None:
                    raise ValueError(f"duplicate diagnostics for scene {scene}")
                diagnostic = json.loads(line.split(" ", 1)[1])
        if diagnostic is None:
            raise ValueError(f"missing diagnostics for scene {scene}")
        if int(diagnostic["scene_seed"]) != scene:
            raise ValueError(f"diagnostic scene mismatch for scene {scene}")
        state = diagnostic["task_state"]
        result[scene] = {
            "policy_seed": policy,
            "official_success": bool(diagnostic["official_success"]),
            "stage_id": int(state["stage_id"]),
            "press_cnt": int(state["press_cnt"]),
            "check_block_in_center": bool(state["check_block_in_center"]),
            "segment_lengths": [
                int(row["close_range"][1]) - int(row["close_range"][0])
                for row in boundaries
            ],
            "trigger_reasons": [str(row["reason"]) for row in boundaries],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    pairs = {100000: 1000, 200000: 1002, 300000: 1004, 400000: 1006}
    payload = parse_run(args.run, pairs)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
