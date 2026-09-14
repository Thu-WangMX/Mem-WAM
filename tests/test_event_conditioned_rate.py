import ast
import inspect
import textwrap

import pytest

from fastwam.memory.event_conditioned_rate import memory_tokens_for_segment
from fastwam.models.wan22.fastwam import FastWAM


@pytest.mark.parametrize("span", [2, 3, 4, 5, 6])
def test_information_event_uses_full_rate(span):
    assert memory_tokens_for_segment(
        span, "segment_relative_control_information"
    ) == 8 * span


@pytest.mark.parametrize(("span", "expected"), [(2, 8), (3, 16), (6, 24)])
def test_forced_close_uses_half_frame_rate(span, expected):
    assert memory_tokens_for_segment(span, "forced_maximum") == expected


def test_unknown_reason_is_rejected():
    with pytest.raises(ValueError, match="unsupported"):
        memory_tokens_for_segment(4, "end")


def test_training_loss_passes_manifest_plan_into_layerwise_transformer():
    source = textwrap.dedent(inspect.getsource(FastWAM._training_loss_full_kv))
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_run_layerwise_memory_training_transformer"
    ]
    assert len(calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    assert ast.unparse(keywords["memory_groups"]) == "dynamic_memory_groups"
    assert (
        ast.unparse(keywords["memory_token_counts"])
        == "dynamic_memory_token_counts"
    )
    assert "layerwise training ignored the manifest memory-token plan" in source
