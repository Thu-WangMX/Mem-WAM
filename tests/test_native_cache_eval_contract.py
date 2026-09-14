from pathlib import Path

from hydra import compose, initialize_config_dir


def test_native_cache_eval_inherits_fullkv_contract():
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="sim_robotwin_native_cache")

    assert cfg.sim_cfg_name == "sim_robotwin_native_cache.yaml"
    assert cfg.EVALUATION.policy_name == "fastwam_native_cache_policy"
    assert cfg.model.native_cache.enabled is True
    assert cfg.model.native_cache.group_size == 4
    assert cfg.EVALUATION.instruction_type == "seen"
    assert cfg.EVALUATION.action_horizon == 16
    assert cfg.EVALUATION.replan_steps == 16
    assert cfg.EVALUATION.num_inference_steps == 50
    assert cfg.EVALUATION.skip_get_obs_within_replan is False
    assert cfg.EVALUATION.action_delta_fraction is None
