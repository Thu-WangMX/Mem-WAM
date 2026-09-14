#!/usr/bin/env python3
"""Validate a PutBack wrist-event policy checkpoint before RMBench rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir

from fastwam.evaluation.wrist_event_contract import (
    validate_wrist_event_eval_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictor", type=Path, required=True)
    parser.add_argument("--policy-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, default=5000)
    parser.add_argument("--eval-config-name", default="sim_robotwin_wrist_event")
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(repo / "configs"),
    ):
        evaluation_config = compose(config_name=args.eval_config_name)
    report = validate_wrist_event_eval_contract(
        training_config_path=args.training_config,
        evaluation_config=evaluation_config,
        manifest_path=args.manifest,
        predictor_checkpoint_path=args.predictor,
        policy_checkpoint_path=args.policy_checkpoint,
        expected_step=args.expected_step,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
