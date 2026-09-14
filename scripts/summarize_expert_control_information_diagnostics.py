from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fastwam.evaluation.embodied_information_online import load_predictor
from fastwam.memory.control_information_probe import (
    load_compact_control_probe,
    score_counterfactual_control_information,
)
from scripts.build_putback_feature_predictor_dataset import build_projected_episode
from scripts.prepare_putback_phase_annotation import load_episode_control


METRICS = (
    "information",
    "observed_feature_rms",
    "observed_feature_abs_max",
    "predicted_feature_rms",
    "feature_prediction_residual_rms",
    "prior_action_rms",
    "posterior_action_rms",
    "relative_control_information",
    "raw_proprio_rms",
    "executed_action_rms",
    "executed_action_delta_rms",
)


def _rms(value: torch.Tensor) -> float:
    value = torch.as_tensor(value, dtype=torch.float32)
    return float(torch.sqrt(torch.mean(value.square())))


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "min": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--pca", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--probe-dataset", type=Path, required=True)
    parser.add_argument("--feature-predictor", type=Path, required=True)
    parser.add_argument("--control-probe", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing output: {args.output}")

    device = torch.device("cpu")
    predictor = load_predictor(args.feature_predictor, device=device)
    probe, _ = load_compact_control_probe(args.control_probe, device=device)
    pca = torch.load(args.pca, map_location="cpu", weights_only=True)
    probe_payload = torch.load(args.probe_dataset, map_location="cpu", weights_only=False)
    trace_payload = torch.load(args.scores, map_location="cpu", weights_only=False)
    probe_rows = {
        (int(row["episode"]), int(row["frame"])): row
        for row in probe_payload["rows"]
    }

    rows = []
    maximum_trace_error = 0.0
    for episode in range(40):
        feature = torch.load(
            args.feature_root / "features" / f"episode_{episode:03d}.pt",
            map_location="cpu",
            weights_only=True,
        )
        maximum_frame = max(
            int(stream["frame_indices"].max()) for stream in feature["phases"].values()
        )
        actions, proprio = load_episode_control(
            args.dataset_root, episode, list(range(maximum_frame + 1))
        )
        dynamics = build_projected_episode(
            episode=episode,
            feature_payload=feature,
            pca_artifact=pca,
            actions=torch.from_numpy(actions),
            proprio=torch.from_numpy(proprio),
            max_history=8,
        )["visual_action"]
        trace = trace_payload["episodes"][episode]
        trace_by_frame = {
            int(frame): float(info)
            for frame, info in zip(trace["frame_indices"], trace["information"])
        }
        for row in dynamics:
            frame = int(row["target_frame"])
            probe_row = probe_rows[(episode, frame)]
            result = score_counterfactual_control_information(
                feature_predictor=predictor,
                control_probe=probe,
                history=torch.as_tensor(row["history"], dtype=torch.float32),
                dynamics_condition=torch.as_tensor(row["condition"], dtype=torch.float32),
                observed_feature=torch.as_tensor(row["target"], dtype=torch.float32),
                normalized_proprio=torch.as_tensor(probe_row["proprio"], dtype=torch.float32),
            )
            maximum_trace_error = max(
                maximum_trace_error, abs(float(result.information) - trace_by_frame[frame])
            )
            observed = torch.as_tensor(row["target"], dtype=torch.float32)
            predicted = torch.as_tensor(result.predicted_feature, dtype=torch.float32)
            action_window = torch.from_numpy(actions[frame - 16 : frame])
            prior_rms = _rms(result.prior_action)
            posterior_rms = _rms(result.posterior_action)
            rows.append(
                {
                    "episode": episode,
                    "frame": frame,
                    "information": float(result.information),
                    "observed_feature_rms": _rms(observed),
                    "observed_feature_abs_max": float(observed.abs().max()),
                    "predicted_feature_rms": _rms(predicted),
                    "feature_prediction_residual_rms": _rms(observed - predicted),
                    "prior_action_rms": prior_rms,
                    "posterior_action_rms": posterior_rms,
                    "relative_control_information": float(result.information)
                    / max(prior_rms, posterior_rms, 1e-8),
                    "raw_proprio_rms": _rms(torch.from_numpy(proprio[frame])),
                    "executed_action_rms": _rms(action_window),
                    "executed_action_delta_rms": _rms(action_window[-1] - action_window[0]),
                }
            )
        print(json.dumps({"episode": episode, "rows": len(dynamics)}), flush=True)

    def summarize(selected: list[dict]) -> dict:
        return {name: _quantiles([float(row[name]) for row in selected]) for name in METRICS}

    report = {
        "schema_version": "expert_control_information_diagnostics_v1",
        "maximum_locked_trace_error": maximum_trace_error,
        "all": summarize(rows),
        "statistics_episodes_0_29": summarize(
            [row for row in rows if row["episode"] < 30]
        ),
        "calibration_episodes_30_39": summarize(
            [row for row in rows if row["episode"] >= 30]
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"report": str(args.output), "rows": len(rows), "maximum_locked_trace_error": maximum_trace_error}, sort_keys=True))


if __name__ == "__main__":
    main()
