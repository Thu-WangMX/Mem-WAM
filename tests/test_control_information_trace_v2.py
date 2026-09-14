from __future__ import annotations

import json

import pytest

from scripts.trace_putback_control_information_v2 import (
    validate_heldout_trace_access_v2,
)


def _locks(tmp_path, *, runtime_schema="putback_locked_control_information_runtime_v2"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps(
            {
                "schema_version": "putback_locked_control_information_candidate_v2",
                "hashes": {},
            },
            sort_keys=True,
        )
        + "\n"
    )
    import hashlib

    candidate_sha256 = hashlib.sha256(candidate.read_bytes()).hexdigest()
    runtime = tmp_path / "runtime.json"
    runtime.write_text(
        json.dumps(
            {
                "schema_version": runtime_schema,
                "candidate_sha256": candidate_sha256,
                "candidate": json.loads(candidate.read_text()),
            },
            sort_keys=True,
        )
        + "\n"
    )
    return runtime, candidate


def test_v2_heldout_trace_requires_exact_locked_split_and_v2_runtime(tmp_path):
    runtime, candidate = _locks(tmp_path)
    validated_runtime, validated_candidate = validate_heldout_trace_access_v2(
        list(range(40, 50)),
        runtime_lock_path=runtime,
        candidate_lock_path=candidate,
    )
    assert validated_runtime == runtime.resolve()
    assert validated_candidate == candidate.resolve()

    with pytest.raises(ValueError, match="exactly episodes 40-49"):
        validate_heldout_trace_access_v2(
            list(range(40, 49)),
            runtime_lock_path=runtime,
            candidate_lock_path=candidate,
        )

    old_runtime, old_candidate = _locks(
        tmp_path / "old", runtime_schema="putback_locked_control_information_runtime_v1"
    )
    with pytest.raises(ValueError, match="schema"):
        validate_heldout_trace_access_v2(
            list(range(40, 50)),
            runtime_lock_path=old_runtime,
            candidate_lock_path=old_candidate,
        )


def test_v2_heldout_trace_rejects_candidate_changed_after_runtime_lock(tmp_path):
    runtime, candidate = _locks(tmp_path)
    candidate.write_text(candidate.read_text() + "\n")
    with pytest.raises(ValueError, match="candidate lock sha256 mismatch"):
        validate_heldout_trace_access_v2(
            list(range(40, 50)),
            runtime_lock_path=runtime,
            candidate_lock_path=candidate,
        )
