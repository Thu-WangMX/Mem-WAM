from __future__ import annotations

import pytest

from scripts.lock_putback_embodied_information_candidate import validate_lock_evidence


def _evidence():
    return {
        "dev": {
            "proxy_micro": {"precision": .91, "recall": 1.0, "f1": .95},
            "reason_counts": {"wam_information_budget": 5},
            "dynamic_length_count": 13,
            "retroactive_boundary_count": 0,
        },
        "latency": {"all_updates_before_deadline": True, "wall_max_seconds": .08},
    }


def test_lock_requires_quality_wam_activity_causality_and_latency():
    validate_lock_evidence(_evidence())
    for path, value in (
        (("dev", "proxy_micro", "precision"), .5),
        (("dev", "reason_counts", "wam_information_budget"), 0),
        (("dev", "retroactive_boundary_count"), 1),
        (("latency", "all_updates_before_deadline"), False),
    ):
        evidence = _evidence(); cursor = evidence
        for key in path[:-1]: cursor = cursor[key]
        cursor[path[-1]] = value
        with pytest.raises(ValueError, match="lock gate"):
            validate_lock_evidence(evidence)

