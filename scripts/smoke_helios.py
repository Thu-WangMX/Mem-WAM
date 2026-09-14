"""Real pretrained weights + real data, one backward and a short action inference."""
import json
import os
import time
import hashlib
import h5py
from pathlib import Path
import torch
from hydra import compose,initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import default_collate
from scripts.check_full_kv_latent_parity import _episode_row, _observation
from experiments.robotwin.fastwam_policy.deploy_policy import WorldActionRobotWinPolicy


def main():
    root=Path(__file__).resolve().parents[1]
    torch.set_num_threads(4)
    with initialize_config_dir(config_dir=str(root/'configs'),version_base=None):
        cfg=compose(config_name='train',overrides=['task=rmbench_helios'])
    print('Loading real dataset',flush=True)
    dataset=instantiate(cfg.data.train)
    bs=int(os.environ.get('SMOKE_BATCH_SIZE','16'))
    batch=default_collate([dataset._get(round(i*(len(dataset)-1)/max(bs-1,1))) for i in range(bs)])
    print('Loading pretrained model',flush=True)
    model=instantiate(cfg.model)
    model.eval().requires_grad_(False)
    model.dit.train().requires_grad_(True)
    model.proprio_encoder.train().requires_grad_(True)
    torch.cuda.reset_peak_memory_stats()
    started=time.time()
    loss,metrics=model.training_loss(batch)
    assert torch.isfinite(loss)
    print('Forward',float(loss.detach()),flush=True)
    loss.backward()
    torch.cuda.synchronize()
    result={'batch_size':bs,'loss':float(loss.detach()),'metrics':metrics,
            'forward_backward_seconds':time.time()-started,
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
            'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
            'optimizer_included':False}
    for name,p in [('long',model.video_expert.patch_long.weight),('mid',model.video_expert.patch_mid.weight)]:
        assert p.grad is not None and torch.isfinite(p.grad).all()
        result[name+'_gradient_norm']=float(p.grad.float().norm())
    (root/'migration/smoke_helios_training.json').write_text(json.dumps(result,indent=2))
    model.zero_grad(set_to_none=True)
    del batch,loss
    torch.cuda.empty_cache()
    model.eval()
    first,second,third=[dataset._get(i) for i in range(3)]
    assert int(first['episode_index'])==int(third['episode_index'])
    # History was precomputed from original HDF5 simulator images, not the
    # re-encoded LeRobot MP4 clips used for the auxiliary future-video target.
    # Replay the actual deployment preprocessor on the identical raw source.
    row=_episode_row(Path(cfg.data.train.dataset_dirs[0]),int(first['episode_index']))
    policy=object.__new__(WorldActionRobotWinPolicy)
    policy.model=model
    with h5py.File(row['raw_file_name'],'r') as handle:
        frames=[policy._build_robotwin_image_tensor(_observation(handle,i)) for i in range(0,33,4)]
    prefix=torch.stack(frames,dim=2)
    manifest=json.loads((Path(cfg.data.train.full_kv_cache_path)/'manifest.json').read_text())
    with open(model.model_paths['vae'],'rb') as f:
        actual_sha=hashlib.file_digest(f,'sha256').hexdigest()
    assert actual_sha==manifest['metadata']['vae_model_sha256']
    result['vae_sha256_match']=True
    with torch.no_grad():
        encoded=model._encode_input_image_latents_tensor(prefix)
        expected=third['history_latents'][-3:].squeeze(2).permute(1,0,2,3).unsqueeze(0).to(encoded.device)
        diff=(encoded.float()-expected.float()).abs()
        result['continuous_vae_prefix_max_abs']=float(diff.max())
        result['continuous_vae_prefix_mean_abs']=float(diff.mean())
        assert float(diff.max())<=0.02 and float(diff.mean())<=0.002, result
        pred=model.infer_action(prompt=None,input_image=prefix,action_horizon=16,
            proprio=third['proprio'][0],context=third['context'],context_mask=third['context_mask'],
            full_kv_frame_index=2,num_inference_steps=2,seed=42)
        assert torch.isfinite(pred['action']).all() and pred['action'].shape==(16,14)
        result['inference_action_shape']=list(pred['action'].shape)
        result['history_tokens']=pred['full_kv_video_tokens']
    (root/'migration/smoke_helios.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
