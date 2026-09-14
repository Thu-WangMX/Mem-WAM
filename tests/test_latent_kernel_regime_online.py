import torch

from fastwam.memory.latent_kernel_regime_online import OnlineLatentKernelRegimeSegmenter


def _latent(channel: int) -> torch.Tensor:
    value = torch.zeros(48, 1, 24, 20)
    value[channel] = 1.0
    return value


def test_stable_sequence_uses_l8():
    runtime = OnlineLatentKernelRegimeSegmenter()
    result = None
    for decision in range(11):
        runtime.observe(decision=decision, latent=_latent(0))
        result = runtime.arrive_planning(decision=decision)
    assert result == (2, 10)


def test_persistent_regime_change_selects_l4():
    runtime = OnlineLatentKernelRegimeSegmenter()
    result = None
    for decision in range(11):
        runtime.observe(decision=decision, latent=_latent(0 if decision < 6 else 1))
        result = runtime.arrive_planning(decision=decision)
    assert result == (2, 6)


def test_reset_restores_causal_sequence():
    runtime = OnlineLatentKernelRegimeSegmenter()
    runtime.observe(decision=0, latent=_latent(0))
    assert runtime.arrive_planning(decision=0) is None
    runtime.reset()
    runtime.observe(decision=0, latent=_latent(1))
    assert runtime.arrive_planning(decision=0) is None
