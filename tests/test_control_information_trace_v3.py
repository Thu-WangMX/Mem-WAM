from __future__ import annotations

import hashlib
import json

import pytest

from scripts.trace_putback_control_information_v3 import (
    validate_heldout_trace_access_v3,
)


def _locks(tmp_path, runtime_schema, candidate_schema):
    tmp_path.mkdir(parents=True, exist_ok=True)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps({"schema_version": candidate_schema, "hashes": {}}, sort_keys=True)
        + "\n"
    )
    runtime = tmp_path / "runtime.json"
    runtime.write_text(
        json.dumps(
            {
                "schema_version": runtime_schema,
                "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                "candidate": json.loads(candidate.read_text()),
            },
            sort_keys=True,
        )
        + "\n"
    )
    return runtime, candidate


def test_v3_trace_accepts_only_v3_lock_after_exact_heldout_split(tmp_path):
    runtime, candidate = _locks(
        tmp_path,
        "putback_locked_control_information_runtime_v3",
        "putback_locked_control_information_candidate_v3",
    )
    assert validate_heldout_trace_access_v3(
        list(range(40, 50)),
        runtime_lock_path=runtime,
        candidate_lock_path=candidate,
    ) == (runtime.resolve(), candidate.resolve())
    with pytest.raises(ValueError, match="exactly episodes 40-49"):
        validate_heldout_trace_access_v3(
            list(range(40, 49)),
            runtime_lock_path=runtime,
            candidate_lock_path=candidate,
        )

    old_runtime, old_candidate = _locks(
        tmp_path / "old",
        "putback_locked_control_information_runtime_v2",
        "putback_locked_control_information_candidate_v2",
    )
    with pytest.raises(ValueError, match="schema"):
        validate_heldout_trace_access_v3(
            list(range(40, 50)),
            runtime_lock_path=old_runtime,
            candidate_lock_path=old_candidate,
        )
