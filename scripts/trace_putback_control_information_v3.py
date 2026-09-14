from __future__ import annotations

from pathlib import Path
from typing import Sequence

from fastwam.evaluation.control_information_online_v3 import (
    CANDIDATE_LOCK_SCHEMA_V3,
    RUNTIME_LOCK_SCHEMA_V3,
)
from scripts.trace_putback_control_information_v2 import (
    _validate_heldout_trace_access,
    run_heldout_trace_cli,
)


def validate_heldout_trace_access_v3(
    episodes: Sequence[int],
    *,
    runtime_lock_path: str | Path,
    candidate_lock_path: str | Path,
) -> tuple[Path, Path]:
    return _validate_heldout_trace_access(
        episodes,
        runtime_lock_path=runtime_lock_path,
        candidate_lock_path=candidate_lock_path,
        runtime_lock_schema=RUNTIME_LOCK_SCHEMA_V3,
        candidate_lock_schema=CANDIDATE_LOCK_SCHEMA_V3,
        version_label="v3",
    )


def main() -> None:
    run_heldout_trace_cli(
        runtime_lock_schema=RUNTIME_LOCK_SCHEMA_V3,
        candidate_lock_schema=CANDIDATE_LOCK_SCHEMA_V3,
        trace_authorization="locked_control_information_runtime_v3",
        version_label="v3",
    )


if __name__ == "__main__":
    main()
