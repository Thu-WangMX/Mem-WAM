from __future__ import annotations

from scripts.evaluate_putback_control_information_selector import evaluate_gate


def _passing_report():
    return {
        "shortcut": {
            "wam_probe_mae": 0.04,
            "proprio_only_mae": 0.10,
            "schedule_baseline_mae": 0.25,
        },
        "segments": {
            "distinct_segment_patterns": 10,
            "learned_boundary_fraction": 0.82,
        },
        "replay": {
            "matching_episodes": 50,
            "retroactive_boundary_count": 0,
        },
        "semantic": {
            "control_information_f1": 0.72,
            "raw_residual_f1": 0.64,
            "gripper_selector_f1": 0.75,
            "reviewed_transition_count": 50,
        },
        "visual": {
            "development_reviewed": True,
            "heldout_rendered": 4,
        },
        "deployment": {
            "selector_observations": 8,
            "fallback_count": 0,
            "old_selector_count": 0,
        },
        "latency": {
            "selector_latency_p95_ms": 1200.0,
            "locked_latency_limit_ms": 1500.0,
            "peak_allocated_bytes": 70_000_000_000,
            "device_total_bytes": 80_000_000_000,
        },
    }


def test_gate_rejects_schedule_shortcut_and_forced_boundary_dominance():
    report = _passing_report()
    report["shortcut"]["wam_probe_mae"] = report["shortcut"][
        "proprio_only_mae"
    ]
    assert evaluate_gate(report)["pass"] is False

    report = _passing_report()
    report["segments"]["learned_boundary_fraction"] = 0.49
    assert evaluate_gate(report)["pass"] is False


def test_gate_requires_full_replay_parity_and_zero_retroactive_boundaries():
    report = _passing_report()
    report["replay"]["matching_episodes"] = 49
    assert evaluate_gate(report)["pass"] is False

    report = _passing_report()
    report["replay"]["retroactive_boundary_count"] = 1
    assert evaluate_gate(report)["pass"] is False


def test_gate_rejects_unreviewed_semantics_or_any_online_fallback():
    report = _passing_report()
    report["semantic"]["reviewed_transition_count"] = 0
    assert evaluate_gate(report)["pass"] is False

    report = _passing_report()
    report["deployment"]["fallback_count"] = 1
    assert evaluate_gate(report)["pass"] is False


def test_gate_requires_visual_review_and_latency_memory_headroom():
    report = _passing_report()
    report["visual"]["development_reviewed"] = False
    assert evaluate_gate(report)["pass"] is False

    report = _passing_report()
    report["latency"]["peak_allocated_bytes"] = 81_000_000_000
    assert evaluate_gate(report)["pass"] is False
