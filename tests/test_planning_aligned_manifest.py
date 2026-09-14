from fastwam.memory.planning_aligned_manifest import (
    OnlinePlanningBoundaryAligner,
    align_detector_events,
)


def _event(frame, reason="embodied_gripper_transition"):
    return {"confirmation_frame": frame, "reason": reason}


def test_aligns_confirmation_forward_without_backdating():
    result = align_detector_events(
        [_event(56), _event(124), _event(180), _event(268), _event(336), _event(340)],
        episode_frames=348,
        replan_stride=16,
    )

    assert result["decision_count"] == 22
    assert result["boundaries"] == [0, 4, 8, 12, 17, 21, 22]
    assert result["confirmation_to_decision"][:4] == [
        {"confirmation_frame": 56, "aligned_frame": 64, "decision_index": 4,
         "reason": "embodied_gripper_transition", "status": "kept"},
        {"confirmation_frame": 124, "aligned_frame": 128, "decision_index": 8,
         "reason": "embodied_gripper_transition", "status": "kept"},
        {"confirmation_frame": 180, "aligned_frame": 192, "decision_index": 12,
         "reason": "embodied_gripper_transition", "status": "kept"},
        {"confirmation_frame": 268, "aligned_frame": 272, "decision_index": 17,
         "reason": "embodied_gripper_transition", "status": "kept"},
    ]
    assert result["confirmation_to_decision"][-1]["status"] == "after_last_decision"
    assert result["retroactive_boundary_count"] == 0


def test_coalesces_events_that_cannot_form_two_decision_observations():
    result = align_detector_events(
        [_event(20), _event(36), _event(52)], episode_frames=80, replan_stride=16
    )

    assert result["boundaries"] == [0, 2, 4, 5]
    assert [row["status"] for row in result["confirmation_to_decision"]] == [
        "kept", "coalesced_short_group", "kept"
    ]


def test_partition_covers_every_planning_observation_exactly_once():
    result = align_detector_events(
        [_event(44), _event(100, "wam_information_budget")],
        episode_frames=143,
        replan_stride=16,
    )

    covered = [index for left, right in zip(result["boundaries"], result["boundaries"][1:])
               for index in range(left, right)]
    assert covered == list(range(result["decision_count"]))
    assert all(right - left <= 8 for left, right in zip(
        result["boundaries"], result["boundaries"][1:]
    ))


def test_terminal_tail_is_not_used_as_an_online_boundary():
    result = align_detector_events(
        [
            _event(48, "segment_relative_control_information"),
            _event(144, "forced_maximum"),
            _event(224, "terminal_tail"),
        ],
        episode_frames=240,
        replan_stride=16,
    )

    assert result["decision_count"] == 15
    assert result["boundaries"] == [0, 3, 9, 15]
    assert result["reasons"] == {
        "0": "start",
        "3": "segment_relative_control_information",
        "9": "forced_maximum",
        "15": "end",
    }
    assert result["confirmation_to_decision"][-1]["status"] == (
        "noncausal_terminal_tail"
    )


def test_online_arrivals_exactly_match_frozen_alignment():
    events = [_event(20), _event(36), _event(52), _event(100, "wam_information_budget")]
    frozen = align_detector_events(events, episode_frames=143, replan_stride=16)
    online = OnlinePlanningBoundaryAligner(replan_stride=16)
    emitted = [0]
    for decision_index in range(frozen["decision_count"]):
        decision_frame = decision_index * 16
        for event in events:
            if (event["confirmation_frame"] + 15) // 16 == decision_index:
                online.observe_confirmation(**event)
        boundary = online.arrive_decision(decision_frame)
        if boundary is not None:
            emitted.append(boundary[1])
    emitted.append(frozen["decision_count"])

    assert emitted == frozen["boundaries"]
    assert online.retroactive_boundary_count == 0
