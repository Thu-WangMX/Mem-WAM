from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "ops" / "analyze_putback_wam_semantic_alignment.sh"


def test_launcher_preflight_validates_inputs_without_gpu(tmp_path):
    analysis = tmp_path / "analysis"
    dataset = tmp_path / "dataset"
    (analysis / "episodes").mkdir(parents=True)
    (dataset / "data" / "chunk-000").mkdir(parents=True)
    (dataset / "videos" / "chunk-000").mkdir(parents=True)
    for episode in (40, 41):
        (analysis / "episodes" / f"episode_{episode:03d}.json").write_text("{}")
        (dataset / "data" / "chunk-000" / f"episode_{episode:06d}.parquet").write_bytes(b"x")
    env = {**os.environ, "REPO_ROOT": str(ROOT), "ANALYSIS_ROOT": str(analysis), "DATASET_ROOT": str(dataset), "RUNTIME_BIN": str(Path(os.sys.executable).parent), "PREFLIGHT_ONLY": "1"}

    result = subprocess.run(["bash", str(LAUNCHER)], cwd=ROOT, env=env, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
    assert "preflight_status=ok" in result.stdout
    assert "gpu" not in result.stdout.lower()
