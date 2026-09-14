from __future__ import annotations

import numpy as np

from fastwam.memory.physical_settle_rate_debt import (
    FALLBACK_REASON,
    SCHEMA_VERSION,
    validate_segments,
)
from fastwam.memory.physical_settle_rate_debt_online import (
    OnlinePhysicalSettleRateSegmenter,
    physical_evidence,
)


def test_physical_evidence_is_prefix_invariant() -> None:
    generator = np.random.default_rng(42)
    prefix = generator.normal(size=(20, 14)).cumsum(axis=0)
    suffix = prefix[-1] + generator.normal(size=(8, 14)).cumsum(axis=0)
    extended = np.concatenate([prefix, suffix], axis=0)
    np.testing.assert_allclose(
        physical_evidence(prefix),
        physical_evidence(extended)[: len(prefix)],
        rtol=0,
        atol=1e-12,
    )


def test_static_episode_uses_causal_l8_fallbacks() -> None:
    runtime = OnlinePhysicalSettleRateSegmenter()
    closed = []
    for decision in range(30):
        segment = runtime.arrive_planning(
            decision=decision, state=np.zeros(14, dtype=np.float32)
        )
        if segment is not None:
            closed.append(segment)
    assert closed
    assert all(segment.length == 8 for segment in closed)
    assert all(segment.reason == FALLBACK_REASON for segment in closed)
    assert all(segment.confirmed_at == segment.start + 8 for segment in closed)


def test_manifest_validator_accepts_rate_debt_cap5() -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "decision_count": 30,
        "segments": [
            {
                "start": 2,
                "end": 7,
                "confirmed_at": 10,
                "reason": "selfcal_settle_or_gripper",
                "evidence": 0.9,
                "rate_debt": 2.0,
                "cumulative_mean_length": 5.0,
            },
            {
                "start": 7,
                "end": 11,
                "confirmed_at": 15,
                "reason": "selfcal_settle_or_gripper",
                "evidence": 0.85,
                "rate_debt": 5.0,
                "cumulative_mean_length": 4.5,
            },
            {
                "start": 11,
                "end": 19,
                "confirmed_at": 19,
                "reason": "max8_rate_repay",
                "evidence": None,
                "rate_debt": 4.0,
                "cumulative_mean_length": 17 / 3,
            },
        ],
    }
    rows = validate_segments(payload)
    assert [row.length for row in rows] == [5, 4, 8]
    assert rows[-1].rate_debt == 4.0
