from __future__ import annotations

import copy
from typing import Any, Callable

import torch

from fastwam.evaluation.predictive_selector_runtime import PredictiveSelectorRuntime


def _online_groups(trace, config):
    runtime = PredictiveSelectorRuntime(selector_config=config)
    for frame in (0, 4, 8, 12):
        runtime.observe(frame=frame, observation=torch.zeros(1))
        runtime.complete(frame=frame, residual=None)
    events = []
    for observation in trace["observations"]:
        frame = int(observation["frame"])
        runtime.observe(frame=frame, observation=torch.zeros(1))
        event = runtime.complete(frame=frame, residual=observation["residuals"])
        if event is not None:
            events.append(event)
    tail = runtime.state.finalize(frame=int(trace["terminal_frame"]))
    if tail is not None:
        events.append(tail)
    return [(event.group_start, event.group_end, event.reason) for event in events], runtime


def run_four_episode_smoke(
    *, manifest: dict[str, Any], traces, rollout_fn: Callable[[], bool]
) -> dict[str, Any]:
    selected = sorted(int(key) for key in manifest["episodes"])[:4]
    if len(selected) != 4:
        raise ValueError("training-readiness smoke requires four episodes")
    maximum_queue = 0
    for episode in selected:
        online, runtime = _online_groups(traces[episode], manifest["selector_config"])
        frozen = [
            (row["start_frame"], row["end_frame"], row["reason"])
            for row in manifest["episodes"][str(episode)]["groups"]
        ]
        if online != frozen:
            raise RuntimeError(f"episode {episode} dataset/runtime group mismatch")
        maximum_queue = max(maximum_queue, runtime.maximum_queue_depth)
        if any(row["completion_frame"] != row["input_frame"] for row in runtime.records):
            raise RuntimeError("detector missed its four-frame completion deadline")

    torch.manual_seed(42)
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 16
    targets = torch.ones(4, 2)
    def step(net, opt):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(net(inputs), targets)
        loss.backward()
        opt.step()
        return float(loss.item())
    first_loss = step(model, optimizer)
    resumed_model = copy.deepcopy(model)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
    resumed_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    next_loss = step(model, optimizer)
    resumed_next_loss = step(resumed_model, resumed_optimizer)
    if next_loss != resumed_next_loss or not all(torch.isfinite(p).all() for p in model.parameters()):
        raise RuntimeError("optimizer resume is not exactly reproducible")
    if rollout_fn() is not True:
        raise RuntimeError("RM-Bench smoke rollout failed")
    return {
        "episodes": 4, "dataset_runtime_exact": True, "finite_loss": True,
        "first_loss": first_loss, "resume_next_loss_exact": True,
        "rmbench_rollout_completed": True, "maximum_queue_depth": maximum_queue,
        "all_updates_before_next_arrival": True, "pass": maximum_queue == 1,
    }
