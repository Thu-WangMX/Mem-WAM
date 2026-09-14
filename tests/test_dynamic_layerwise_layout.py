import torch

from fastwam.memory.native_cache import build_dynamic_layerwise_training_layout


def _layout():
    return build_dynamic_layerwise_training_layout(
        clean_frames=7,
        noisy_frames=1,
        tokens_per_frame=2,
        action_tokens=3,
        memory_groups=((2, 3, 4, 5),),
        memory_tokens=8,
        anchor_frames=2,
        recent_frames=4,
        device=torch.device("cpu"),
    )


def test_dynamic_layout_packs_variable_segments_and_raw_current_tail():
    layout = _layout()

    assert [(segment.kind, segment.frame_indices) for segment in layout.segments] == [
        ("anchor", (0,)),
        ("anchor", (1,)),
        ("source", (2, 3, 4, 5)),
        ("memory", (2, 3, 4, 5)),
        ("recent", (6,)),
    ]
    assert [stop - start for start, stop in layout.retained_ranges] == [2, 2, 8, 2]
    assert layout.retained_clean_tokens == 14


def test_action_and_future_see_anchors_memories_and_complete_open_tail_only():
    layout = _layout()
    memory_ranges = [
        (segment.start, segment.stop)
        for segment in layout.segments
        if segment.kind == "memory"
    ]
    source = next(segment for segment in layout.segments if segment.kind == "source")
    frame_ranges = {
        segment.frame_indices[0]: (segment.start, segment.stop)
        for segment in layout.segments
        if segment.kind in {"anchor", "recent"}
    }
    frame_ranges.update(
        **{
            str(frame): (
                source.start + (frame - 2) * 2,
                source.start + (frame - 1) * 2,
            )
            for frame in range(2, 6)
        }
    )
    frame_ranges = {int(frame): value for frame, value in frame_ranges.items()}

    for rows in (layout.noisy_range, layout.action_range):
        visible_frames = (0, 1, 6)
        assert all(
            layout.attention_mask[rows[0] : rows[1], start:stop].all()
            for start, stop in memory_ranges
        )
        assert all(
            layout.attention_mask[
                rows[0] : rows[1], frame_ranges[frame][0] : frame_ranges[frame][1]
            ].all()
            for frame in visible_frames
        )
        for frame in range(2, 6):
            assert not layout.attention_mask[
                rows[0] : rows[1], frame_ranges[frame][0] : frame_ranges[frame][1]
            ].any()


def test_dynamic_reader_retains_an_open_tail_longer_than_four_frames():
    layout = build_dynamic_layerwise_training_layout(
        clean_frames=11,
        noisy_frames=1,
        tokens_per_frame=2,
        action_tokens=3,
        memory_groups=((2, 3, 4, 5),),
        memory_tokens=8,
        anchor_frames=2,
        recent_frames=4,
        device=torch.device("cpu"),
    )
    raw_tail = [
        segment
        for segment in layout.segments
        if segment.kind == "recent"
    ]
    assert [segment.frame_indices[0] for segment in raw_tail] == [6, 7, 8, 9, 10]
    for segment in raw_tail:
        assert layout.attention_mask[
            layout.action_range[0] : layout.action_range[1],
            segment.start : segment.stop,
        ].all()


def test_open_tail_encoding_is_causal_and_does_not_depend_on_memory_confirmation():
    layout = build_dynamic_layerwise_training_layout(
        clean_frames=11,
        noisy_frames=1,
        tokens_per_frame=2,
        action_tokens=3,
        memory_groups=((2, 3, 4, 5),),
        memory_tokens=8,
        anchor_frames=2,
        recent_frames=4,
        device=torch.device("cpu"),
    )
    memory = next(segment for segment in layout.segments if segment.kind == "memory")
    tail = [segment for segment in layout.segments if segment.kind == "recent"]
    last = tail[-1]
    rows = layout.attention_mask[last.start : last.stop]
    assert not rows[:, memory.start : memory.stop].any()
    assert all(rows[:, segment.start : segment.stop].all() for segment in tail)


def test_dynamic_layout_rejects_noncontiguous_or_current_frame_groups():
    common = dict(
        clean_frames=5,
        noisy_frames=1,
        tokens_per_frame=2,
        action_tokens=3,
        memory_tokens=8,
        anchor_frames=2,
        recent_frames=4,
        device=torch.device("cpu"),
    )
    for groups in (((2, 3), (5, 6)), ((2, 3), (4, 5, 6))):
        try:
            build_dynamic_layerwise_training_layout(memory_groups=groups, **common)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid memory groups accepted: {groups}")
