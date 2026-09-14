"""Compose train and deployment configs without constructing a model."""
import json
from pathlib import Path
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf

root=Path(__file__).resolve().parents[1]
rows=[]
with initialize_config_dir(config_dir=str(root/'configs'),version_base=None):
    for backend,data,model,sim in [
        ('helios','rmbench_helios','fastwam_helios','sim_robotwin_helios'),
        ('memorywam','rmbench_rearrange_blocks_fixed_l4_k8','fastwam_segment_compress_k8','sim_robotwin_memorywam'),
        ('fullkv','rmbench_cover_blocks_fullattention','fastwam_full_kv','sim_robotwin_full_kv')]:
        train=compose(config_name='train',overrides=['task=rmbench_helios',f'data={data}',f'model={model}'])
        # Reproduce the eval launcher passing the selected task into the child policy process.
        evaluate=compose(config_name=sim,overrides=['task=robotwin_full_kv_eval'])
        assert train.model.get('memory_backend','original')==evaluate.model.get('memory_backend','original')
        assert bool(train.model.get('native_cache',{}).get('enabled',False))==bool(evaluate.model.get('native_cache',{}).get('enabled',False))
        for key in ('video_dit_config','action_dit_config'):
            for field in ('hidden_dim','num_layers','num_heads','attn_head_dim'):
                assert train.model[key][field]==evaluate.model[key][field]
        assert train.model.action_rope_spatial_mode==evaluate.model.action_rope_spatial_mode
        assert evaluate.EVALUATION.action_horizon==16 and evaluate.EVALUATION.replan_steps==16
        rows.append({'backend':backend,'train_dataset':train.data.train._target_,
                     'eval_model_backend':evaluate.model.get('memory_backend','original'),
                     'native_memory':bool(evaluate.model.get('native_cache',{}).get('enabled',False)),
                     'sim_config':evaluate.sim_cfg_name})
output = root / "migration/config_check.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(rows, indent=2))
print(json.dumps(rows,indent=2))
