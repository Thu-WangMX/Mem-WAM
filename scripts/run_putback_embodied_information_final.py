from __future__ import annotations

import argparse, hashlib, json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

from fastwam.memory.embodied_information_boundary import EmbodiedInformationBoundaryState, contextual_residual_z
from scripts.analyze_putback_feature_predictor_residuals import _load_model, predict_stream_residuals
from scripts.build_putback_feature_predictor_dataset import build_projected_episode
from scripts.evaluate_putback_embodied_information_candidate import _control, match_event_frames
from scripts.prepare_putback_phase_annotation import propose_gripper_events


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    for name in ("feature-bank","pca","dataset-root","predictor-resume","predictor-fpbin","contextual-stats","locked-candidate","marker","output"):
        p.add_argument("--"+name,required=True)
    args=p.parse_args(); out=Path(args.output).resolve(); marker=Path(args.marker).resolve()
    if out.exists() or marker.exists(): raise FileExistsError("final control-proxy evaluation is one-time only")
    lock_path=Path(args.locked_candidate).resolve(); lock=json.loads(lock_path.read_text())
    if lock.get("schema_version")!="putback_locked_embodied_information_candidate_v1": raise ValueError("locked Candidate 3 schema mismatch")
    actual={"predictor_sha256":sha(args.predictor_fpbin),"pca_sha256":sha(args.pca),"contextual_stats_sha256":sha(args.contextual_stats)}
    if any(lock["hashes"][k]!=v for k,v in actual.items()): raise ValueError("final dependency hash differs from lock")
    marker.parent.mkdir(parents=True,exist_ok=True)
    marker_payload={"schema_version":"putback_embodied_information_control_proxy_final_v1","locked_candidate_sha256":sha(lock_path),"opened_at":datetime.now(timezone.utc).isoformat(),"episodes":list(range(40,50))}
    temp=marker.with_suffix(marker.suffix+".tmp");temp.write_text(json.dumps(marker_payload,indent=2,sort_keys=True)+"\n");temp.replace(marker)
    pca=torch.load(args.pca,map_location="cpu",weights_only=True);stats=torch.load(args.contextual_stats,map_location="cpu",weights_only=False)
    model,model_config=_load_model(Path(args.predictor_resume),torch.device("cuda:0")); feature_root=Path(args.feature_bank); dataset_root=Path(args.dataset_root)
    episode_reports={};traces={};totals=Counter(); reasons=Counter();lengths=Counter()
    for ep in range(40,50):
        feature=torch.load(feature_root/"features"/f"episode_{ep:03d}.pt",map_location="cpu",weights_only=True)
        maximum=max(int(x["frame_indices"].max()) for x in feature["phases"].values());actions,proprio=_control(dataset_root,ep)
        built=build_projected_episode(episode=ep,feature_payload=feature,pca_artifact=pca,actions=actions,proprio=proprio,max_history=8)["visual_action"]
        trace=predict_stream_residuals(model,built,layers=5,regions=4,components=64,batch_size=model_config.batch_size);traces[ep]=trace;by={int(f):r for f,r in zip(trace["frame_indices"],trace["residuals"])}
        detector=list(range(0,len(proprio),4));target=[x["frame"] for x in propose_gripper_events(actions[detector].numpy(),detector)]
        state=EmbodiedInformationBoundaryState(**lock["selector_config"]);events=[]
        for frame in detector:
            z=None if frame<16 else contextual_residual_z(by[frame],frame=frame,stats=stats)
            event=state.update(frame=frame,proprio=proprio[frame],standardized_residual=z)
            if event:events.append(event)
        tail=state.finalize(frame=detector[-1]);
        if tail:events.append(tail)
        predicted=[e.confirmation_frame for e in events if e.reason in {"embodied_gripper_transition","wam_information_budget"}]
        metrics=match_event_frames(predicted,target,tolerance=4);totals.update(metrics["counts"])
        for e in events:reasons[e.reason]+=1;lengths[e.group_end-e.group_start]+=1
        episode_reports[str(ep)]={"proxy_metrics":metrics,"target_control_transitions":target,"events":[e.__dict__ for e in events]}
    tp,fp,fn=totals["true_positive"],totals["false_positive"],totals["false_negative"];precision=tp/(tp+fp);recall=tp/(tp+fn);f1=2*precision*recall/(precision+recall)
    report={"schema_version":"putback_embodied_information_control_proxy_final_v1","locked_candidate_sha256":sha(lock_path),"label_type":"control_transition_proxy_not_human_semantics","proxy_micro":{"precision":precision,"recall":recall,"f1":f1,"counts":dict(totals)},"reason_counts":dict(reasons),"group_length_histogram":dict(sorted(lengths.items())),"dynamic_length_count":len(lengths),"retroactive_boundary_count":0,"episodes":episode_reports}
    out.mkdir(parents=True);torch.save(traces,out/"residual_traces.pt");(out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps({k:report[k] for k in ("proxy_micro","reason_counts","group_length_histogram","dynamic_length_count")},indent=2,sort_keys=True))


if __name__=="__main__":main()

