import pytest

from fastwam.memory.dynamic_surprise_scorer import transition_noise_seed


def test_per_episode_noise_is_constant_within_episode() -> None:
    assert transition_noise_seed(3, 0, mode="per_episode") == 300000
    assert transition_noise_seed(3, 17, mode="per_episode") == 300000


def test_per_episode_noise_changes_between_episodes() -> None:
    assert transition_noise_seed(2, 9, mode="per_episode") == 200000
    assert transition_noise_seed(3, 9, mode="per_episode") == 300000


def test_per_transition_preserves_existing_seed_contract() -> None:
    assert transition_noise_seed(3, 17, mode="per_transition") == 300017


def test_unknown_noise_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="noise mode"):
        transition_noise_seed(0, 0, mode="random")
