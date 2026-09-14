from __future__ import annotations

import argparse
import copy
import json
import os
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from fastwam.memory.predictive_feature_model import FeaturePredictor, huber_prediction_loss


RESUME_SCHEMA = "putback_feature_predictor_resume_v1"
SPLIT_SCHEMA = "putback_predictor_split_v1"


@dataclass(frozen=True)
class PredictorTrainingConfig:
    feature_dim: int = 1280
    condition_dim: int = 238
    hidden_dim: int = 512
    condition_embedding_dim: int = 128
    batch_size: int = 256
    epochs: int = 200
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 15
    seed: int = 42


@dataclass
class PredictorTrainingResult:
    model: FeaturePredictor
    report: dict[str, Any]
    resume_state: dict[str, Any]


def make_split_manifest(
    *, train_episodes: Iterable[int], val_episodes: Iterable[int]
) -> dict[str, Any]:
    train = [int(value) for value in train_episodes]
    validation = [int(value) for value in val_episodes]
    if train != list(range(26)) or validation != list(range(26, 30)):
        raise ValueError("predictor split is locked to train 0-25 and validation 26-29")
    return {
        "schema_version": SPLIT_SCHEMA,
        "train_episodes": train,
        "val_episodes": validation,
    }


def serialize_split_manifest(manifest: dict[str, Any]) -> bytes:
    return (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _clone_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(state)


def _collate(rows: Sequence[dict[str, Any]], device: torch.device):
    lengths = torch.tensor([len(row["history"]) for row in rows], dtype=torch.int64)
    maximum = int(lengths.max().item())
    feature_dim = int(torch.as_tensor(rows[0]["history"]).shape[-1])
    history = torch.zeros((len(rows), maximum, feature_dim), dtype=torch.float32)
    for index, row in enumerate(rows):
        values = torch.as_tensor(row["history"], dtype=torch.float32)
        history[index, : len(values)] = values
    condition = torch.stack(
        [torch.as_tensor(row["condition"], dtype=torch.float32) for row in rows]
    )
    target = torch.stack(
        [torch.as_tensor(row["target"], dtype=torch.float32) for row in rows]
    )
    return history.to(device), condition.to(device), target.to(device), lengths


def _batches(
    examples: Sequence[dict[str, Any]],
    *,
    batch_size: int,
    epoch: int | None,
    seed: int,
) -> Iterable[list[dict[str, Any]]]:
    if epoch is None:
        order = torch.arange(len(examples))
    else:
        generator = torch.Generator().manual_seed(int(seed) + int(epoch))
        order = torch.randperm(len(examples), generator=generator)
    for start in range(0, len(order), int(batch_size)):
        yield [examples[int(index)] for index in order[start : start + int(batch_size)]]


@torch.no_grad()
def evaluate_loss(
    model: FeaturePredictor,
    examples: Sequence[dict[str, Any]],
    *,
    batch_size: int,
) -> float:
    if not examples:
        raise ValueError("validation examples are empty")
    device = next(model.parameters()).device
    model.eval()
    total = 0.0
    count = 0
    for rows in _batches(examples, batch_size=batch_size, epoch=None, seed=0):
        history, condition, target, lengths = _collate(rows, device)
        prediction = model(history, condition, lengths=lengths)
        loss = torch.nn.functional.huber_loss(
            prediction, target, delta=1.0, reduction="sum"
        )
        total += float(loss.item())
        count += int(target.numel())
    return total / count


def _validate_examples(
    examples: Sequence[dict[str, Any]], config: PredictorTrainingConfig
) -> None:
    if not examples:
        raise ValueError("predictor examples are empty")
    for row in examples:
        if torch.as_tensor(row["history"]).ndim != 2:
            raise ValueError("history must be a sequence")
        if torch.as_tensor(row["history"]).shape[-1] != config.feature_dim:
            raise ValueError("feature dimension mismatch")
        if torch.as_tensor(row["target"]).shape != (config.feature_dim,):
            raise ValueError("target dimension mismatch")
        if torch.as_tensor(row["condition"]).shape != (config.condition_dim,):
            raise ValueError("condition dimension mismatch")
        if int(row.get("episode", -1)) >= 30:
            raise ValueError("held-out episode leaked into predictor training")


def train_predictor(
    train_examples: Sequence[dict[str, Any]],
    validation_examples: Sequence[dict[str, Any]],
    *,
    config: PredictorTrainingConfig,
    resume_state: dict[str, Any] | None = None,
    device: str | torch.device = "cpu",
) -> PredictorTrainingResult:
    _validate_examples(train_examples, config)
    _validate_examples(validation_examples, config)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.seed)
    device = torch.device(device)
    model = FeaturePredictor(
        feature_dim=config.feature_dim,
        condition_dim=config.condition_dim,
        hidden_dim=config.hidden_dim,
        condition_embedding_dim=config.condition_embedding_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    if resume_state is None:
        start_epoch = 0
        initial_validation_loss = evaluate_loss(
            model, validation_examples, batch_size=config.batch_size
        )
        best_validation_loss = initial_validation_loss
        best_model = _clone_state_dict(model.state_dict())
        epochs_without_improvement = 0
        history: list[dict[str, float | int]] = []
    else:
        if resume_state.get("schema_version") != RESUME_SCHEMA:
            raise ValueError("resume schema mismatch")
        saved_config = dict(resume_state["config"])
        requested = asdict(config)
        for key, value in saved_config.items():
            if key != "epochs" and requested[key] != value:
                raise ValueError(f"resume config changed: {key}")
        if config.epochs < int(resume_state["epoch"]):
            raise ValueError("resume target epoch precedes saved epoch")
        model.load_state_dict(resume_state["model"])
        optimizer.load_state_dict(resume_state["optimizer"])
        start_epoch = int(resume_state["epoch"])
        best_model = _clone_state_dict(resume_state["best_model"])
        best_validation_loss = float(resume_state["best_validation_loss"])
        epochs_without_improvement = int(resume_state["epochs_without_improvement"])
        initial_validation_loss = float(
            resume_state["report"]["initial_validation_loss"]
        )
        history = copy.deepcopy(resume_state["report"]["history"])

    completed_epoch = start_epoch
    for epoch in range(start_epoch, config.epochs):
        if epochs_without_improvement >= config.patience:
            break
        model.train()
        train_total = 0.0
        train_elements = 0
        for rows in _batches(
            train_examples,
            batch_size=config.batch_size,
            epoch=epoch,
            seed=config.seed,
        ):
            history_batch, condition, target, lengths = _collate(rows, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(history_batch, condition, lengths=lengths)
            loss = huber_prediction_loss(prediction, target)
            loss.backward()
            optimizer.step()
            train_total += float(loss.item()) * int(target.numel())
            train_elements += int(target.numel())
        validation_loss = evaluate_loss(
            model, validation_examples, batch_size=config.batch_size
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_total / train_elements,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_model = _clone_state_dict(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        completed_epoch = epoch + 1

    report = {
        "initial_validation_loss": initial_validation_loss,
        "best_validation_loss": best_validation_loss,
        "completed_epochs": completed_epoch,
        "history": history,
    }
    resume = {
        "schema_version": RESUME_SCHEMA,
        "epoch": completed_epoch,
        "model": _clone_state_dict(model.state_dict()),
        "optimizer": _clone_state_dict(optimizer.state_dict()),
        "best_model": _clone_state_dict(best_model),
        "best_validation_loss": best_validation_loss,
        "epochs_without_improvement": epochs_without_improvement,
        "report": copy.deepcopy(report),
        "config": asdict(config),
    }
    model.load_state_dict(best_model)
    return PredictorTrainingResult(model=model, report=report, resume_state=resume)


def save_compact_model(
    path: str | Path,
    model: FeaturePredictor,
    *,
    config: PredictorTrainingConfig,
    report: dict[str, Any],
) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite model: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = [
        (key, value.detach().cpu().contiguous())
        for key, value in sorted(model.state_dict().items())
    ]
    metadata = {
        "schema_version": "putback_feature_predictor_model_v1",
        "config": asdict(config),
        "report": report,
        "tensors": [
            {
                "name": name,
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "shape": list(tensor.shape),
                "nbytes": tensor.numel() * tensor.element_size(),
            }
            for name, tensor in tensors
        ],
    }
    header = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(b"FASTWAM_PREDICTOR_V1\n")
        handle.write(struct.pack("<Q", len(header)))
        handle.write(header)
        for _, tensor in tensors:
            handle.write(tensor.numpy().tobytes(order="C"))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mode", required=True, choices=("visual_only", "visual_action"))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = Path(args.output_root).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite predictor root: {output}")
    payload = torch.load(args.dataset, map_location="cpu", weights_only=False)
    split = make_split_manifest(train_episodes=range(26), val_episodes=range(26, 30))
    if payload.get("split_manifest") != split:
        raise ValueError("predictor dataset split mismatch")
    rows = payload[args.mode]
    train_rows = [row for row in rows if int(row["episode"]) in split["train_episodes"]]
    val_rows = [row for row in rows if int(row["episode"]) in split["val_episodes"]]
    config = PredictorTrainingConfig()
    resume = (
        torch.load(args.resume, map_location="cpu", weights_only=False)
        if args.resume
        else None
    )
    result = train_predictor(
        train_rows,
        val_rows,
        config=config,
        resume_state=resume,
        device=args.device,
    )
    output.mkdir(parents=True)
    (output / "split_manifest.json").write_bytes(serialize_split_manifest(split))
    (output / "report.json").write_text(
        json.dumps(result.report, indent=2, sort_keys=True) + "\n"
    )
    save_compact_model(output / f"{args.mode}.fpbin", result.model, config=config, report=result.report)
    torch.save(result.resume_state, output / "resume.pt")


if __name__ == "__main__":
    main()
