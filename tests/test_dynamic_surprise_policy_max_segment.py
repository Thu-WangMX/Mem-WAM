from collections import deque
from types import SimpleNamespace

from fastwam.memory.dynamic_surprise import OnlineSurpriseSegmenter
from experiments.robotwin.fastwam_policy.deploy_policy import WorldActionRobotWinPolicy


def test_policy_reset_preserves_diagnostic_max_segment() -> None:
    """A new episode must not silently restore the production max of eight."""

    policy = WorldActionRobotWinPolicy.__new__(WorldActionRobotWinPolicy)
    policy.pending_actions = deque()
    policy._temporal_frames = []
    policy._surprise_segmenter = OnlineSurpriseSegmenter(
        min_segment=2,
        max_segment=4,
        gamma=1.5,
        window=5,
    )
    policy._surprise_latents = []
    policy._surprise_previous_action = None
    policy._surprise_previous_conditioning = None
    policy.episode_count = 0
    policy._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}
    policy.model = SimpleNamespace(device=SimpleNamespace(type="cpu"))

    policy.reset()

    decisions = [policy._surprise_segmenter.preview(0.0)]
    policy._surprise_segmenter.commit(decisions[-1])
    for _ in range(2):
        decisions.append(policy._surprise_segmenter.preview(0.0))
        policy._surprise_segmenter.commit(decisions[-1])
    decisions.append(policy._surprise_segmenter.preview(0.0))

    assert decisions[-1].close_range == (0, 4)
    assert decisions[-1].reason == "max_length"
