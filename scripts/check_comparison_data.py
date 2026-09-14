"""Check real data, common preprocessing and heterogeneous-history batch collation."""
import json
from pathlib import Path
import torch
from torch.utils.data import default_collate
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from fastwam.memory.helios_history import episode_anchors, fixed_history


def main():
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root/'configs'),version_base=None):
        cfg=compose(config_name='train',overrides=['task=rmbench_helios'])
    torch.set_num_threads(2)
    dataset=instantiate(cfg.data.train)
    indices=[round(i*(len(dataset)-1)/15) for i in range(16)]
    # _get deliberately avoids the original __getitem__ fallback to a random sample.
    samples=[dataset._get(i) for i in indices]
    batch=default_collate(samples)
    original_cfg=OmegaConf.to_container(cfg.data.train,resolve=True)
    original_cfg.pop('history_sizes')
    original_cfg.pop('anchor_frames')
    original_cfg['_target_']='fastwam.datasets.lerobot.full_kv_dataset.FullKVRobotVideoDataset'
    original=instantiate(original_cfg)
    assert len(dataset)==len(original)
    for idx,sample in zip(indices,samples):
        reference=original._get(idx)
        for key in ('image','video','action','proprio','action_is_pad','image_is_pad','context','context_mask'):
            if key in sample:
                torch.testing.assert_close(sample[key],reference[key],rtol=0,atol=0)
        history,valid=fixed_history(reference['history_latents'])
        torch.testing.assert_close(sample['history_latents'],history,rtol=0,atol=0)
        assert torch.equal(sample['history_valid'],valid)
        anchors,anchor_valid=episode_anchors(reference['history_latents'])
        torch.testing.assert_close(sample['anchor_latents'],anchors,rtol=0,atol=0)
        assert torch.equal(sample['anchor_valid'],anchor_valid)
    result={'dataset_length':len(dataset),'indices':indices,
            'tensor_shapes':{k:list(v.shape) for k,v in batch.items() if isinstance(v,torch.Tensor)},
            'valid_history_counts':batch['history_valid'].sum(1).tolist(),
            'padded_actions':batch['action_is_pad'].sum(1).tolist(),
            'baseline_data_identical':True}
    (root/'migration/data_check.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
