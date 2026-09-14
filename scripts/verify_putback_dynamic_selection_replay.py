from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fastwam.memory.dynamic_selection_manifest import SCHEMA, replay_episode


def verify_replay(*, manifest, traces):
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError("dynamic selection manifest schema mismatch")
    total = 0
    for episode_key, frozen in manifest["episodes"].items():
        episode = int(episode_key)
        groups, _ = replay_episode(
            traces[episode], manifest["selector_config"], manifest["source_hashes"]
        )
        if groups != frozen["groups"]:
            raise RuntimeError(f"episode {episode} online replay differs from frozen groups")
        total += len(groups)
    return {"episodes": len(manifest["episodes"]), "groups": total, "exact": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--traces", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    traces = torch.load(args.traces, map_location="cpu", weights_only=False)
    print(json.dumps(verify_replay(manifest=manifest, traces=traces), sort_keys=True))


if __name__ == "__main__":
    main()

