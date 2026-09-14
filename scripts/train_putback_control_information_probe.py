from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Sequence

import torch

from fastwam.memory.control_information_probe import (
    FutureActionProbe,
    masked_action_huber_loss,
    save_compact_control_probe,
)
from scripts.build_putback_control_information_dataset import (
    DATASET_SCHEMA,
    SPLIT_ROLES,
)


MODES = ("wam_proprio", "proprio_only")
DETERMINISTIC_CUBLAS_CONFIGS = (":4096:8", ":16:8")


@dataclass(frozen=True)
class ControlProbeTrainingConfig:
    feature_dim: int = 1280
    proprio_dim: int = 14
    hidden_dim: int = 512
    proprio_hidden_dim: int = 128
    horizon: int = 16
    action_dim: int = 14
    batch_size: int = 256
    epochs: int = 200
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 15
    seed: int = 42

    def model_config(self) -> dict[str, int]:
        return {
            key: int(getattr(self, key))
            for key in (
                "feature_dim",
                "proprio_dim",
                "hidden_dim",
                "proprio_hidden_dim",
                "horizon",
                "action_dim",
            )
        }


class ProbeBatch(NamedTuple):
    feature: torch.Tensor
    proprio: torch.Tensor
    target: torch.Tensor
    mask: torch.Tensor


@dataclass
class ControlProbeTrainingResult:
    model: FutureActionProbe
    report: dict[str, Any]


def ensure_deterministic_cuda_environment(device: torch.device) -> None:
    """Fail before CUDA initialization when deterministic cuBLAS is not configured."""

    if torch.device(device).type != "cuda":
        return
    configured = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if configured not in DETERMINISTIC_CUBLAS_CONFIGS:
        raise RuntimeError(
            "deterministic CUDA training requires CUBLAS_WORKSPACE_CONFIG="
            f"{DETERMINISTIC_CUBLAS_CONFIGS[0]} or {DETERMINISTIC_CUBLAS_CONFIGS[1]}; "
            f"received {configured!r}"
        )


def split_rows(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = [row for row in rows if int(row["episode"]) in SPLIT_ROLES["train"]]
    validation = [
        row for row in rows if int(row["episode"]) in SPLIT_ROLES["validation"]
    ]
    if not train or not validation:
        raise ValueError("control-probe train and validation splits must be nonempty")
    return train, validation


def collate_rows(
    rows: Sequence[dict[str, Any]],
    *,
    mode: str,
    device: str | torch.device,
) -> ProbeBatch:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if not rows:
        raise ValueError("cannot collate an empty control-probe batch")
    feature = torch.stack(
        [torch.as_tensor(row["feature"], dtype=torch.float32) for row in rows]
    )
    if mode == "proprio_only":
        feature.zero_()
    proprio = torch.stack(
        [torch.as_tensor(row["proprio"], dtype=torch.float32) for row in rows]
    )
    target = torch.stack(
        [torch.as_tensor(row["target_actions"], dtype=torch.float32) for row in rows]
    )
    mask = torch.stack(
        [torch.as_tensor(row["target_mask"], dtype=torch.bool) for row in rows]
    )
    target = target.clone()
    target[~mask] = 0
    resolved = torch.device(device)
    return ProbeBatch(
        feature.to(resolved),
        proprio.to(resolved),
        target.to(resolved),
        mask.to(resolved),
    )


def _validate_rows(
    rows: Sequence[dict[str, Any]],
    *,
    config: ControlProbeTrainingConfig,
    allowed_episodes: set[int],
) -> None:
    if not rows:
        raise ValueError("control-probe rows are empty")
    for row in rows:
        if int(row["episode"]) not in allowed_episodes:
            raise ValueError("control-probe split contains an unauthorized episode")
        if torch.as_tensor(row["feature"]).shape != (config.feature_dim,):
            raise ValueError("control-probe feature dimension mismatch")
        if torch.as_tensor(row["proprio"]).shape != (config.proprio_dim,):
            raise ValueError("control-probe proprio dimension mismatch")
        if torch.as_tensor(row["target_actions"]).shape != (
            config.horizon,
            config.action_dim,
        ):
            raise ValueError("control-probe target action shape mismatch")
        mask = torch.as_tensor(row["target_mask"], dtype=torch.bool)
        if mask.shape != (config.horizon,) or not bool(mask.any()):
            raise ValueError("control-probe target mask is invalid")


def _batches(
    rows: Sequence[dict[str, Any]],
    *,
    batch_size: int,
    seed: int,
    epoch: int | None,
) -> Iterable[list[dict[str, Any]]]:
    if epoch is None:
        order = torch.arange(len(rows))
    else:
        generator = torch.Generator().manual_seed(int(seed) + int(epoch))
        order = torch.randperm(len(rows), generator=generator)
    for start in range(0, len(order), int(batch_size)):
        yield [rows[int(index)] for index in order[start : start + int(batch_size)]]


@torch.no_grad()
def evaluate_control_probe(
    model: FutureActionProbe,
    rows: Sequence[dict[str, Any]],
    *,
    config: ControlProbeTrainingConfig,
    mode: str,
) -> dict[str, float]:
    if not rows:
        raise ValueError("control-probe evaluation rows are empty")
    device = next(model.parameters()).device
    model.eval()
    loss_sum = 0.0
    absolute_sum = 0.0
    square_sum = 0.0
    element_count = 0
    for selected in _batches(
        rows, batch_size=config.batch_size, seed=config.seed, epoch=None
    ):
        batch = collate_rows(selected, mode=mode, device=device)
        prediction = model(batch.feature, batch.proprio)
        expanded = batch.mask.unsqueeze(-1).expand_as(prediction)
        difference = prediction[expanded] - batch.target[expanded]
        count = int(difference.numel())
        loss_sum += float(
            torch.nn.functional.huber_loss(
                prediction[expanded], batch.target[expanded], delta=1.0, reduction="sum"
            ).item()
        )
        absolute_sum += float(difference.abs().sum().item())
        square_sum += float(difference.square().sum().item())
        element_count += count
    if element_count <= 0:
        raise RuntimeError("control-probe evaluation has no valid targets")
    return {
        "loss": loss_sum / element_count,
        "mae": absolute_sum / element_count,
        "rmse": (square_sum / element_count) ** 0.5,
    }


def train_control_probe(
    train_rows: Sequence[dict[str, Any]],
    validation_rows: Sequence[dict[str, Any]],
    *,
    config: ControlProbeTrainingConfig,
    mode: str,
    device: str | torch.device,
) -> ControlProbeTrainingResult:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    _validate_rows(
        train_rows, config=config, allowed_episodes=set(SPLIT_ROLES["train"])
    )
    _validate_rows(
        validation_rows,
        config=config,
        allowed_episodes=set(SPLIT_ROLES["validation"]),
    )
    resolved_device = torch.device(device)
    ensure_deterministic_cuda_environment(resolved_device)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.seed)
    model = FutureActionProbe(**config.model_config()).to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    initial = evaluate_control_probe(
        model, validation_rows, config=config, mode=mode
    )
    best_loss = float(initial["loss"])
    best_model = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(config.epochs):
        if epochs_without_improvement >= config.patience:
            break
        model.train()
        for selected in _batches(
            train_rows,
            batch_size=config.batch_size,
            seed=config.seed,
            epoch=epoch,
        ):
            batch = collate_rows(selected, mode=mode, device=resolved_device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch.feature, batch.proprio)
            loss = masked_action_huber_loss(prediction, batch.target, batch.mask)
            loss.backward()
            optimizer.step()
        metrics = evaluate_control_probe(
            model, validation_rows, config=config, mode=mode
        )
        history.append(
            {
                "epoch": epoch + 1,
                "validation_loss": metrics["loss"],
                "validation_mae": metrics["mae"],
                "validation_rmse": metrics["rmse"],
            }
        )
        if metrics["loss"] < best_loss:
            best_loss = float(metrics["loss"])
            best_model = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
    model.load_state_dict(best_model, strict=True)
    final = evaluate_control_probe(model, validation_rows, config=config, mode=mode)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    report = {
        "schema_version": "putback_control_information_probe_training_v1",
        "mode": mode,
        "config": asdict(config),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "initial_validation_loss": initial["loss"],
        "best_validation_loss": best_loss,
        "validation_loss": final["loss"],
        "validation_mae": final["mae"],
        "validation_rmse": final["rmse"],
        "completed_epochs": len(history),
        "history": history,
    }
    return ControlProbeTrainingResult(model=model, report=report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    output = Path(args.output_root).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite control-probe output: {output}")
    dataset_path = Path(args.dataset).expanduser().resolve()
    payload = torch.load(dataset_path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != DATASET_SCHEMA or payload.get("complete") is not True:
        raise ValueError("control-probe dataset is incomplete or incompatible")
    if payload.get("split_roles") != SPLIT_ROLES:
        raise ValueError("control-probe dataset split roles changed")
    config = ControlProbeTrainingConfig(
        feature_dim=int(payload["feature_dim"]),
        proprio_dim=int(payload["proprio_dim"]),
        horizon=int(payload["horizon"]),
        action_dim=int(payload["action_dim"]),
    )
    train_rows, validation_rows = split_rows(payload["rows"])
    result = train_control_probe(
        train_rows,
        validation_rows,
        config=config,
        mode=args.mode,
        device=args.device,
    )
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary control-probe output: {temporary}")
    temporary.mkdir(parents=True)
    (temporary / "report.json").write_text(
        json.dumps(result.report, indent=2, sort_keys=True) + "\n"
    )
    (temporary / "split_roles.json").write_text(
        json.dumps(SPLIT_ROLES, indent=2, sort_keys=True) + "\n"
    )
    save_compact_control_probe(
        temporary / f"{args.mode}.cipbin",
        result.model,
        config=config.model_config(),
        report=result.report,
    )
    temporary.replace(output)


if __name__ == "__main__":
    main()
