from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fastwam.memory.dynamic_selection_manifest import freeze_selection_manifest, write_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True)
    parser.add_argument("--locked-selector", required=True)
    parser.add_argument("--heldout-report", required=True)
    parser.add_argument("--source-hashes", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    traces = torch.load(args.traces, map_location="cpu", weights_only=False)
    selector = json.loads(Path(args.locked_selector).read_text())
    gate = json.loads(Path(args.heldout_report).read_text())["gate"]
    hashes = json.loads(Path(args.source_hashes).read_text())
    manifest = freeze_selection_manifest(
        traces=traces, selector_config=selector["selector_config"],
        source_hashes=hashes, heldout_gate=gate,
    )
    write_manifest(args.output, manifest)


if __name__ == "__main__":
    main()

