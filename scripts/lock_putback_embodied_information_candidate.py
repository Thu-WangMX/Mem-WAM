from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha(path: str | Path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_lock_evidence(evidence):
    dev, latency = evidence["dev"], evidence["latency"]
    checks = {
        "precision": dev["proxy_micro"]["precision"] >= .90,
        "recall": dev["proxy_micro"]["recall"] >= .95,
        "f1": dev["proxy_micro"]["f1"] >= .90,
        "wam_active": dev["reason_counts"].get("wam_information_budget", 0) > 0,
        "dynamic": dev["dynamic_length_count"] >= 2,
        "causal": dev["retroactive_boundary_count"] == 0,
        "latency": latency["all_updates_before_deadline"] is True
        and latency["wall_max_seconds"] < latency.get("deadline_seconds", .4),
    }
    if not all(checks.values()):
        raise ValueError(f"candidate lock gate failed: {checks}")
    return checks


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dev-report",required=True)
    parser.add_argument("--latency-report",required=True)
    parser.add_argument("--predictor",required=True)
    parser.add_argument("--pca",required=True)
    parser.add_argument("--contextual-stats",required=True)
    parser.add_argument("--code-commit",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args(); output=Path(args.output).resolve()
    if output.exists(): raise FileExistsError(f"refusing locked candidate overwrite: {output}")
    dev=json.loads(Path(args.dev_report).read_text()); latency=json.loads(Path(args.latency_report).read_text())
    gates=validate_lock_evidence({"dev":dev,"latency":latency})
    payload={
        "schema_version":"putback_locked_embodied_information_candidate_v1",
        "selector_config":dev["selector_config"],"development_gate":gates,
        "label_scope":"control_transition_proxy_not_human_semantics",
        "code_commit":args.code_commit,
        "hashes":{
            "dev_report_sha256":_sha(args.dev_report),"latency_report_sha256":_sha(args.latency_report),
            "predictor_sha256":_sha(args.predictor),"pca_sha256":_sha(args.pca),
            "contextual_stats_sha256":_sha(args.contextual_stats),
        },
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    temp=output.with_suffix(output.suffix+".tmp");temp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");temp.replace(output)
    print(_sha(output))


if __name__=="__main__":main()

