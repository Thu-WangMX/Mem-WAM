from omegaconf import OmegaConf
import pytest

from experiments.robotwin.fastwam_policy.deploy_policy import (
    _frozen_init_scorer_model_cfg,
    _normalize_surprise_scorer_source,
    _surprise_scorer_memory_groups,
)


def test_frozen_init_scorer_uses_official_initialization_without_text_encoder() -> None:
    source = OmegaConf.create(
        {
            "load_text_encoder": True,
            "skip_dit_load_from_pretrain": True,
            "action_dit_pretrained_path": None,
        }
    )

    configured = _frozen_init_scorer_model_cfg(
        source,
        official_action_dit_path="/models/official_action_dit.pt",
    )

    assert configured.load_text_encoder is False
    assert configured.skip_dit_load_from_pretrain is False
    assert configured.action_dit_pretrained_path == "/models/official_action_dit.pt"
    assert source.load_text_encoder is True
    assert source.skip_dit_load_from_pretrain is True
    assert source.action_dit_pretrained_path is None


@pytest.mark.parametrize("value", ["initialization", " INITIALIZATION "])
def test_initialization_scorer_uses_full_history_not_policy_memory_groups(value) -> None:
    source = _normalize_surprise_scorer_source(value)

    assert source == "initialization"
    assert _surprise_scorer_memory_groups(source, ((0, 3), (4, 7))) is None


def test_policy_scorer_preserves_current_memory_groups() -> None:
    groups = ((0, 3), (4, 7))

    assert _surprise_scorer_memory_groups("policy", groups) == groups


def test_unknown_scorer_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="dynamic surprise scorer source"):
        _normalize_surprise_scorer_source("latest")
