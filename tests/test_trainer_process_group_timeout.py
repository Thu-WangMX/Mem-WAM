from __future__ import annotations

from datetime import timedelta

import pytest

from fastwam.trainer import build_accelerator_kwargs_handlers


def test_build_accelerator_handler_uses_two_hour_timeout():
    handlers = build_accelerator_kwargs_handlers(7200)

    assert len(handlers) == 1
    assert handlers[0].timeout == timedelta(seconds=7200)


@pytest.mark.parametrize("invalid", [0, -1])
def test_build_accelerator_handler_rejects_non_positive_timeout(invalid):
    with pytest.raises(ValueError, match="process_group_timeout_seconds.*positive"):
        build_accelerator_kwargs_handlers(invalid)
