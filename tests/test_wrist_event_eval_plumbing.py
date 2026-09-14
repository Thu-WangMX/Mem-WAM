from experiments.robotwin import eval_robotwin_single


def test_robotwin_forwards_all_online_wrist_event_arguments():
    required = {
        "wrist_event_online",
        "wrist_event_predictor_repo",
        "wrist_event_checkpoint",
        "wrist_event_threshold",
        "wrist_event_history",
        "wrist_event_min_segment",
        "wrist_event_max_segment",
    }

    assert required <= set(eval_robotwin_single.MEMORY_POLICY_FORWARD_KEYS)
