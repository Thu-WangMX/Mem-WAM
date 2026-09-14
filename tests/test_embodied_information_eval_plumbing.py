from experiments.robotwin import eval_robotwin_single


def test_robotwin_forwards_embodied_information_selector_arguments():
    required = {
        "embodied_information_online",
        "embodied_information_lock",
        "embodied_information_pca",
        "embodied_information_predictor",
        "embodied_information_statistics",
        "embodied_information_init_action_dit_path",
    }
    assert required <= set(eval_robotwin_single.MEMORY_POLICY_FORWARD_KEYS)

