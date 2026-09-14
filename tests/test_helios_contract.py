"""Small real MoT tests: masks, gradients, cache parity and checkpoint isolation."""
import types
import pytest
import torch
from fastwam.models.wan22.helios_video import MTMVideoDiT
from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.fastwam_helios import FastWAMHelios
from fastwam.memory.helios_history import episode_anchors, fixed_history


@pytest.fixture
def model():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    common = dict(hidden_dim=32, ffn_dim=64, text_dim=16, freq_dim=16,
                  eps=1e-6, num_heads=2, attn_head_dim=24, num_layers=2)
    video = MTMVideoDiT(**common, in_dim=4, out_dim=4, patch_size=(1,2,2),
                        has_image_input=False, seperated_timestep=True,
                        multi_term_memory=True, mtm_history_sizes=(16,2,1),
                        mtm_anchor_size=2, mtm_pred_size=1)
    action = ActionDiT(**common, action_dim=14)
    mot = MoT({'video': video, 'action': action}, mot_checkpoint_mixed_attn=False)
    m = FastWAMHelios(video, action, mot, torch.nn.Identity(), text_dim=16,
                      proprio_dim=14, action_train_shift=1., action_infer_shift=1.)
    m.helios_config = {'history_sizes': [16,2,1], 'anchor_frames': 2,
                       'patch_long': [4,8,8],
                       'patch_mid': [2,4,4], 'condition_augmentation': False}
    m.vae.temporal_downsample_factor=4
    return m


def make_inputs(b=1):
    context = torch.randn(b,3,16)
    mask = torch.ones(b,3,dtype=torch.bool)
    return dict(anchor_latents=torch.randn(b,2,4,1,8,8),
                anchor_valid=torch.ones(b,2,dtype=torch.bool),
                history_latents=torch.randn(b,19,4,1,8,8),
                history_valid=torch.ones(b,19,dtype=torch.bool),
                input_latents=torch.randn(b,4,2,8,8), action=torch.randn(b,16,14),
                video_context=context, video_context_mask=mask,
                context=context, context_mask=mask,
                image_is_pad=torch.zeros(b,5,dtype=torch.bool),
                action_is_pad=torch.zeros(b,16,dtype=torch.bool))


def joint(m, x, future=None):
    b = len(x['action'])
    vp = m._video_pre(x['anchor_latents'], x['history_latents'],
                       x['input_latents'][:,:,1:] if future is None else future,
                       torch.full((b,),500.), x['context'],x['context_mask'])
    ap = m.action_expert.pre_dit(x['action'],torch.full((b,),700.),x['context'],x['context_mask'])
    af = m._action_freqs(16,vp)
    out=m.mot(embeds_all={'video':vp['tokens'],'action':ap['tokens']},
        attention_mask=m._joint_mask(vp,16),freqs_all={'video':vp['freqs'],'action':af},
        context_all={'video':{'context':vp['context'],'mask':vp['context_mask']},
                     'action':{'context':ap['context'],'mask':ap['context_mask']}},
        t_mod_all={'video':vp['t_mod'],'action':ap['t_mod']})
    return m.action_expert.post_dit(out['action'],ap), vp


def test_history_window_startup_and_long_episode():
    h=torch.arange(45.).reshape(45,1,1,1,1)
    first,valid=fixed_history(h[:1])
    assert valid.sum()==1 and valid[-1]
    assert torch.equal(first[-1],h[0])
    end,valid=fixed_history(h)
    torch.testing.assert_close(end,h[26:])
    assert valid.all()


def test_episode_anchors_are_causal_and_fixed():
    h=torch.arange(45.).reshape(45,1,1,1,1)
    empty,empty_valid=episode_anchors(h,0)
    assert empty.shape[0]==0 and empty_valid.shape[0]==0
    first,valid=episode_anchors(h[:1])
    assert valid.tolist()==[True,False]
    torch.testing.assert_close(first[0],h[0])
    assert torch.count_nonzero(first[1])==0
    end,valid=episode_anchors(h)
    assert valid.all()
    torch.testing.assert_close(end,h[:2])


def test_batch16_gradients_and_padding(model):
    x=make_inputs(16)
    # Exercise the production training loss with the VAE replaced by known latents.
    model.build_inputs=types.MethodType(lambda self,sample,tiled=False: sample,model)
    x['action_is_pad'][-1]=True
    loss,metrics=model.training_loss(x)
    assert torch.isfinite(loss) and metrics['loss_action']>0
    loss.backward()
    for p in (model.video_expert.patch_long.weight,model.video_expert.patch_mid.weight,
              model.video_expert.patch_embedding.weight,
              model.action_expert.action_encoder.weight):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0


def test_future_cannot_leak_and_prefill_matches_training(model):
    model.eval()
    x=make_inputs(2)
    with torch.no_grad():
        reference,vp=joint(model,x)
        assert vp['meta']['mtm']['L_anchor']==32
        assert vp['meta']['mtm']['L_hist']==56
        changed,_=joint(model,x,torch.randn_like(x['input_latents'][:,:,1:])*100)
        torch.testing.assert_close(reference,changed,atol=1e-6,rtol=1e-5)
        h=vp['meta']['mtm']['L_hist']
        kv=model.mot.prefill_video_cache(video_tokens=vp['tokens'][:,:h],
            video_freqs=vp['freqs'][:h],video_t_mod=vp['t_mod'][:,:h],
            video_context_payload={'context':vp['context'],'mask':vp['context_mask'][:,:h]},
            video_attention_mask=torch.ones(h,h,dtype=torch.bool))
        pred=model._predict_action_noise_with_cache(x['action'],torch.full((2,),700.),
            x['context'],x['context_mask'],model._action_freqs(16,vp),kv,
            torch.ones(16,h+16,dtype=torch.bool),h)
        torch.testing.assert_close(reference,pred,atol=2e-6,rtol=2e-5)


def test_joint_mask_exact_history_future_action_visibility(model):
    x=make_inputs(1)
    _,vp=joint(model,x)
    meta=vp['meta']['mtm']
    h,p,a=meta['L_hist'],meta['L_pred'],16
    mask=model._joint_mask(vp,a)
    assert mask.shape==(h+p+a,h+p+a)
    assert mask[:h,:h].all()
    assert not mask[:h,h:].any()
    assert mask[h:h+p,:h+p].all()
    assert not mask[h:h+p,h+p:].any()
    assert mask[h+p:,:h].all()
    assert not mask[h+p:,h:h+p].any()
    assert mask[h+p:,h+p:].all()


def test_checkpoint_roundtrip_and_backend_guard(model,tmp_path):
    path=tmp_path/'helios.pt'
    model.save_checkpoint(path,step=31)
    expected=model.video_expert.patch_long.weight.detach().clone()
    with torch.no_grad(): model.video_expert.patch_long.weight.zero_()
    assert model.load_checkpoint(path)==31
    torch.testing.assert_close(model.video_expert.patch_long.weight,expected)
    torch.save({'mot':model.mot.state_dict()},path)
    with pytest.raises(ValueError,match='backend'): model.load_checkpoint(path)


def test_online_prefix_and_history_selection(model):
    model.eval()
    x=make_inputs(1)
    prefix=torch.randn(1,4,27,8,8)
    model._encode_input_image_latents_tensor=types.MethodType(lambda self,image,tiled=False: prefix,model)
    captured={}
    original_video_pre=model._video_pre
    def capture_video_pre(self,anchors,history,*args,**kwargs):
        captured['anchors']=anchors.detach().clone()
        captured['history']=history.detach().clone()
        return original_video_pre(anchors,history,*args,**kwargs)
    model._video_pre=types.MethodType(capture_video_pre,model)
    kwargs=dict(prompt=None,input_image=torch.zeros(1),action_horizon=16,
                proprio=torch.zeros(14),context=x['context'],context_mask=x['context_mask'],
                full_kv_frame_index=26,num_inference_steps=2,seed=5)
    result=model.infer_action(**kwargs)
    assert result['action'].shape==(16,14) and result['full_kv_cache'] is None
    chronological=prefix[0].permute(1,0,2,3).unsqueeze(2)
    torch.testing.assert_close(captured['anchors'][0],chronological[:2])
    torch.testing.assert_close(captured['history'][0],chronological[-19:])
    torch.testing.assert_close(result['action'],model.infer_action(**kwargs)['action'])
    kwargs['full_kv_frame_index']=25
    with pytest.raises(ValueError,match='prefix'): model.infer_action(**kwargs)


def test_pretrained_factory_preserves_backbone_without_meta_parameters(monkeypatch):
    from fastwam.models.wan22 import fastwam as base
    from fastwam.models.wan22.wan_video_dit import WanVideoDiT
    common=dict(hidden_dim=32,ffn_dim=64,text_dim=16,freq_dim=16,eps=1e-6,
                num_heads=2,attn_head_dim=24,num_layers=1)
    vcfg=dict(**common,in_dim=4,out_dim=4,patch_size=(1,2,2),has_image_input=False,
              seperated_timestep=True,fuse_vae_embedding_in_latents=True)
    original=WanVideoDiT(**vcfg)
    components=types.SimpleNamespace(dit=original,vae=torch.nn.Identity(),text_encoder=None,
        tokenizer=None,dit_path='test',vae_path='test',text_encoder_path=None,tokenizer_path=None)
    monkeypatch.setattr(base,'load_wan22_ti2v_5b_components',lambda **kw:components)
    m=FastWAMHelios.from_wan22_pretrained(video_dit_config=vcfg,
        action_dit_config=dict(**common,action_dim=14),device='cpu',torch_dtype=torch.float32,
        load_text_encoder=False,mot_checkpoint_mixed_attn=False)
    assert not any(p.is_meta for p in m.parameters())
    assert m.video_expert.blocks[0].self_attn.q.weight.data_ptr()==original.blocks[0].self_attn.q.weight.data_ptr()
    assert torch.isfinite(m.video_expert.patch_long.weight).all()
    torch.testing.assert_close(m.video_expert.patch_long.bias,original.patch_embedding.bias)
