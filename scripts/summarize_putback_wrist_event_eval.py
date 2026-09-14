#!/usr/bin/env python3
"""Summarize an RMBench PutBack wrist-event evaluation run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from fastwam.evaluation.wrist_event_summary import summarize_wrist_event_run


DEFAULT_PAIRS = ("100000:1000", "200000:1002", "1300000:1008", "1400000:1010")


def _parse_pair(text: str) -> tuple[int, int]:
    try:
        scene, policy = text.split(":", maxsplit=1)
        return int(scene), int(policy)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"scene pair must be SCENE_SEED:POLICY_SEED, got {text!r}"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--scene-pair",
        action="append",
        type=_parse_pair,
        dest="scene_pairs",
        help="repeatable SCENE_SEED:POLICY_SEED; defaults to the fixed four",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    pairs = args.scene_pairs or tuple(_parse_pair(pair) for pair in DEFAULT_PAIRS)
    summary = summarize_wrist_event_run(args.run_root, pairs)
    payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    output = args.output or args.run_root / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, output)
    print(payload, end="")
    if not summary["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
