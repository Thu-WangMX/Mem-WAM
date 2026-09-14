from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastwam.models.wan22.wan_video_dit import rope_apply


@dataclass(frozen=True)
class HistoryPartition:
    """Frame-index partition for anchor, block-memory, and recent cache units."""

    anchors: tuple[int, ...]
    memory_groups: tuple[tuple[int, ...], ...]
    raw_tail: tuple[int, ...]
    recent: tuple[int, ...]


def partition_layerwise_history(
    frame_count: int,
    *,
    anchor_frames: int = 2,
    recent_frames: int = 4,
    group_size: int = 4,
) -> HistoryPartition:
    """Partition a trajectory without compressing anchors or recent frames."""
    frame_count = int(frame_count)
    anchor_frames = int(anchor_frames)
    recent_frames = int(recent_frames)
    group_size = int(group_size)
    if frame_count < 1:
        raise ValueError(f"frame_count must be positive, got {frame_count}")
    if anchor_frames < 0 or recent_frames < 0:
        raise ValueError("anchor_frames and recent_frames must be non-negative")
    if group_size < 1:
        raise ValueError(f"group_size must be positive, got {group_size}")

    anchor_stop = min(anchor_frames, frame_count)
    non_anchor_count = frame_count - anchor_stop
    grouped_count = (non_anchor_count // group_size) * group_size
    grouped_stop = anchor_stop + grouped_count
    memory_groups = tuple(
        tuple(range(start, start + group_size))
        for start in range(anchor_stop, grouped_stop, group_size)
    )
    recent_start = max(anchor_stop, frame_count - recent_frames)
    return HistoryPartition(
        anchors=tuple(range(anchor_stop)),
        memory_groups=memory_groups,
        raw_tail=tuple(range(grouped_stop, frame_count)),
        recent=tuple(range(recent_start, frame_count)),
    )


@dataclass(frozen=True)
class CacheUnit:
    """One raw frame or one recursively consolidated layerwise cache unit."""

    kind: str
    endpoint: int
    span: int
    token_count: int
    level: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"anchor", "raw", "recent", "memory"}:
            raise ValueError(f"Unsupported cache unit kind: {self.kind!r}")
        inferred_level = 1 if self.kind == "memory" else 0
        if self.level is None:
            object.__setattr__(self, "level", inferred_level)
        level = int(self.level)
        if int(self.endpoint) < 0:
            raise ValueError(f"endpoint must be non-negative, got {self.endpoint}")
        if int(self.span) < 1:
            raise ValueError(f"span must be positive, got {self.span}")
        if int(self.token_count) < 1:
            raise ValueError(
                f"token_count must be positive, got {self.token_count}"
            )
        if self.kind == "memory":
            if level < 1:
                raise ValueError("Memory cache units must use level one or higher")
            # Level-one units also back the dynamic-surprise path, whose
            # phase-frozen segments have variable spans in [2, 8].  The
            # legacy fixed/recursive state tightens this again below.
            allowed_spans = set(range(1, 9)) if level == 1 else {4 ** level}
            if int(self.span) not in allowed_spans:
                raise ValueError(
                    f"A level-{level} memory must have span in "
                    f"{sorted(allowed_spans)}, "
                    f"got {self.span}"
                )
        else:
            if level != 0:
                raise ValueError("Raw cache units must use level zero")
            if int(self.span) != 1:
                raise ValueError("Raw cache units must represent exactly one frame")

    @property
    def start(self) -> int:
        return int(self.endpoint) - int(self.span) + 1


@dataclass(frozen=True)
class MemoryCarry:
    """One four-child base-4 consolidation event."""

    children: tuple[CacheUnit, ...]
    parent: CacheUnit

    def __post_init__(self) -> None:
        if len(self.children) != 4:
            raise ValueError(
                f"A recursive memory carry requires four children, got {len(self.children)}"
            )
        first = self.children[0]
        if any(int(child.level) != int(first.level) for child in self.children):
            raise ValueError("Carry children must have equal levels")
        if any(int(child.span) != int(first.span) for child in self.children):
            raise ValueError("Carry children must have equal spans")
        for previous, current in zip(self.children, self.children[1:]):
            if current.start != int(previous.endpoint) + 1:
                raise ValueError("Carry children must be temporally contiguous")
        if self.parent.kind != "memory":
            raise ValueError("A carry parent must be a memory unit")
        if int(self.parent.level) != int(first.level) + 1:
            raise ValueError("A carry parent must be exactly one level above its children")
        if int(self.parent.span) != sum(int(child.span) for child in self.children):
            raise ValueError("A carry parent span must equal the sum of its children")
        if int(self.parent.endpoint) != int(self.children[-1].endpoint):
            raise ValueError("A carry parent endpoint must match its final child")


@dataclass(frozen=True)
class RecursiveCarryPlan:
    """Topological carry events and canonical retained hierarchy."""

    carries: tuple[MemoryCarry, ...]
    retained_memories: tuple[CacheUnit, ...]
    raw_tail: tuple[CacheUnit, ...]


def _memory_parent(
    children: Sequence[CacheUnit],
    *,
    memory_tokens: int,
    max_levels: int,
) -> CacheUnit:
    if len(children) != 4:
        raise ValueError(f"Expected four memory children, got {len(children)}")
    output_level = int(children[0].level) + 1
    if output_level > int(max_levels):
        raise ValueError(
            f"Recursive memory level {output_level} exceeds max_levels={max_levels}"
        )
    return CacheUnit(
        kind="memory",
        endpoint=int(children[-1].endpoint),
        span=sum(int(child.span) for child in children),
        token_count=int(memory_tokens),
        level=output_level,
    )


def plan_recursive_carries(
    frame_count: int,
    *,
    anchor_frames: int = 2,
    group_size: int = 4,
    tokens_per_frame: int = 120,
    memory_tokens: int = 32,
    max_levels: int = 8,
    recursive: bool = True,
) -> RecursiveCarryPlan:
    """Plan deterministic base-4 memory creation without touching tensors."""
    frame_count = int(frame_count)
    anchor_frames = int(anchor_frames)
    if frame_count < 1:
        raise ValueError(f"frame_count must be positive, got {frame_count}")
    if anchor_frames < 0:
        raise ValueError("anchor_frames must be non-negative")
    if int(group_size) != 4:
        raise ValueError("Recursive layerwise memory requires group_size=4")
    if min(int(tokens_per_frame), int(memory_tokens), int(max_levels)) < 1:
        raise ValueError("token counts and max_levels must be positive")

    carries: list[MemoryCarry] = []
    active: list[CacheUnit] = []
    raw_tail: list[CacheUnit] = []
    for endpoint in range(min(anchor_frames, frame_count), frame_count):
        raw_tail.append(
            CacheUnit(
                kind="raw",
                endpoint=endpoint,
                span=1,
                token_count=int(tokens_per_frame),
                level=0,
            )
        )
        if len(raw_tail) < int(group_size):
            continue
        children = tuple(raw_tail)
        raw_tail.clear()
        parent = _memory_parent(
            children,
            memory_tokens=int(memory_tokens),
            max_levels=int(max_levels),
        )
        carries.append(MemoryCarry(children=children, parent=parent))
        active.append(parent)
        if not bool(recursive):
            continue
        while len(active) >= int(group_size):
            candidate = tuple(active[-int(group_size) :])
            if len({int(child.level) for child in candidate}) != 1:
                break
            parent = _memory_parent(
                candidate,
                memory_tokens=int(memory_tokens),
                max_levels=int(max_levels),
            )
            carries.append(MemoryCarry(children=candidate, parent=parent))
            active[-int(group_size) :] = [parent]
    return RecursiveCarryPlan(
        carries=tuple(carries),
        retained_memories=tuple(active),
        raw_tail=tuple(raw_tail),
    )


@dataclass(frozen=True)
class LayerwiseMemoryState:
    """Immutable logical cache units paired with their per-layer K/V tensors."""

    units: tuple[CacheUnit, ...]
    kv_cache: tuple[dict[str, torch.Tensor], ...]
    carry_levels: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.units:
            raise ValueError("LayerwiseMemoryState requires at least one cache unit")
        if not self.kv_cache:
            raise ValueError("LayerwiseMemoryState requires at least one cache layer")
        if any(int(level) < 1 for level in self.carry_levels):
            raise ValueError("carry_levels must contain positive memory levels")
        for unit in self.units:
            if unit.kind == "memory" and int(unit.level) == 1 and int(unit.span) not in {1, 4}:
                raise ValueError(
                    "Legacy layerwise level-one memory span must be 1 or 4, "
                    f"got {unit.span}"
                )
        expected_tokens = self.retained_tokens
        for layer_index, layer in enumerate(self.kv_cache):
            if "k" not in layer or "v" not in layer:
                raise ValueError(
                    f"kv_cache[{layer_index}] must contain both k and v"
                )
            if int(layer["k"].shape[1]) != expected_tokens:
                raise ValueError(
                    f"kv_cache[{layer_index}] has {layer['k'].shape[1]} tokens, "
                    f"expected {expected_tokens} from cache units"
                )
            if int(layer["v"].shape[1]) != expected_tokens:
                raise ValueError(
                    f"kv_cache[{layer_index}] value length must be {expected_tokens}"
                )

    @property
    def retained_tokens(self) -> int:
        return sum(int(unit.token_count) for unit in self.units)

    @property
    def represented_frames(self) -> int:
        return max(int(unit.endpoint) for unit in self.units) + 1


@dataclass(frozen=True)
class DynamicLayerwiseMemoryState:
    """Online cache for variable memories, retained raw context, and an open tail."""

    units: tuple[CacheUnit, ...]
    kv_cache: tuple[dict[str, torch.Tensor], ...]
    open_start: int

    def __post_init__(self) -> None:
        if not self.units:
            raise ValueError("DynamicLayerwiseMemoryState requires cache units")
        if not self.kv_cache:
            raise ValueError("DynamicLayerwiseMemoryState requires cache layers")
        represented_frames = max(int(unit.endpoint) for unit in self.units) + 1
        raw_endpoints: set[int] = set()
        covered_frames: set[int] = set()
        for unit in self.units:
            covered_frames.update(range(int(unit.start), int(unit.endpoint) + 1))
            if unit.kind == "memory":
                if int(unit.level) != 1 or not 2 <= int(unit.span) <= 8:
                    raise ValueError("dynamic memory span must be 2..8 at level one")
            else:
                endpoint = int(unit.endpoint)
                if endpoint in raw_endpoints:
                    raise ValueError(
                        f"dynamic cache has duplicate raw endpoint {endpoint}"
                    )
                raw_endpoints.add(endpoint)
        expected_coverage = set(range(represented_frames))
        if covered_frames != expected_coverage:
            raise ValueError(
                "dynamic cache units must cover frames contiguously from zero; "
                f"expected {sorted(expected_coverage)}, got {sorted(covered_frames)}"
            )
        open_start = int(self.open_start)
        if not 0 <= open_start < represented_frames:
            raise ValueError(
                f"open_start must be inside represented history, got {open_start}"
            )
        expected_open_tail = list(range(open_start, represented_frames))
        actual_open_tail = sorted(
            endpoint for endpoint in raw_endpoints if endpoint >= open_start
        )
        if actual_open_tail != expected_open_tail:
            raise ValueError(
                "dynamic cache must retain the contiguous open raw tail; "
                f"expected {expected_open_tail}, got {actual_open_tail}"
            )
        expected_tokens = self.retained_tokens
        for layer_index, layer in enumerate(self.kv_cache):
            if "k" not in layer or "v" not in layer:
                raise ValueError(f"kv_cache[{layer_index}] must contain both k and v")
            if int(layer["k"].shape[1]) != expected_tokens or int(layer["v"].shape[1]) != expected_tokens:
                raise ValueError(
                    f"kv_cache[{layer_index}] length must be {expected_tokens} tokens"
                )

    @property
    def retained_tokens(self) -> int:
        return sum(int(unit.token_count) for unit in self.units)

    @property
    def represented_frames(self) -> int:
        return max(int(unit.endpoint) for unit in self.units) + 1


class LayerwiseBlockMemory(nn.Module):
    """Learned block-gist slots advanced by the native VideoDiT layers."""

    def __init__(
        self,
        video_expert: nn.Module,
        *,
        memory_tokens: int = 32,
        dynamic_tokens_per_frame: int | None = None,
        allocation_mode: str = "span_full",
        group_size: int = 4,
        anchor_frames: int = 2,
        recent_frames: int = 4,
        reader_use_anchor: bool = True,
        reader_use_memory: bool = True,
        reader_use_recent: bool = True,
        recursive: bool = False,
        max_levels: int = 8,
    ) -> None:
        super().__init__()
        memory_tokens = int(memory_tokens)
        if memory_tokens < 1:
            raise ValueError(f"memory_tokens must be positive, got {memory_tokens}")
        if dynamic_tokens_per_frame is not None:
            dynamic_tokens_per_frame = int(dynamic_tokens_per_frame)
            if dynamic_tokens_per_frame < 1:
                raise ValueError(
                    "dynamic_tokens_per_frame must be positive when enabled, got "
                    f"{dynamic_tokens_per_frame}"
                )
        group_size = int(group_size)
        if group_size < 1:
            raise ValueError(f"group_size must be positive, got {group_size}")
        if int(anchor_frames) != 2:
            raise ValueError("Layerwise block memory requires anchor_frames=2")
        if int(recent_frames) != 4:
            raise ValueError("Layerwise block memory requires recent_frames=4")
        if int(max_levels) < 1:
            raise ValueError(f"max_levels must be positive, got {max_levels}")
        if bool(recursive) and group_size != 4:
            raise ValueError("Recursive layerwise memory requires group_size=4")
        self.memory_tokens = memory_tokens
        self.dynamic_tokens_per_frame = dynamic_tokens_per_frame
        self.allocation_mode = str(allocation_mode)
        if self.allocation_mode not in {"span_full", "event_full_forced_half"}:
            raise ValueError(f"unsupported layerwise allocation_mode: {self.allocation_mode!r}")
        self.group_size = group_size
        self.anchor_frames = int(anchor_frames)
        self.recent_frames = int(recent_frames)
        self.reader_use_anchor = bool(reader_use_anchor)
        self.reader_use_memory = bool(reader_use_memory)
        self.reader_use_recent = bool(reader_use_recent)
        self.recursive = bool(recursive)
        self.max_levels = int(max_levels)
        self.hidden_dim = int(video_expert.hidden_dim)
        native_dtype = next(video_expert.parameters()).dtype
        self.slots = nn.Parameter(
            torch.empty(1, memory_tokens, self.hidden_dim, dtype=native_dtype)
        )
        nn.init.normal_(self.slots, mean=0.0, std=0.02)
        if self.recursive:
            self.level_embedding = nn.Embedding(
                self.max_levels + 1,
                self.hidden_dim,
                dtype=native_dtype,
            )
            nn.init.zeros_(self.level_embedding.weight)
        else:
            self.level_embedding = None
        temporal, vertical, horizontal = video_expert.freqs
        self.register_buffer("temporal_freqs", temporal, persistent=False)
        self.register_buffer("vertical_freqs", vertical, persistent=False)
        self.register_buffer("horizontal_freqs", horizontal, persistent=False)

    def initial_tokens(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        level: int = 1,
        token_count: int | None = None,
    ) -> torch.Tensor:
        batch_size = int(batch_size)
        level = int(level)
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if level < 1 or level > self.max_levels:
            raise ValueError(
                f"memory level must lie in [1, {self.max_levels}], got {level}"
            )
        if not self.recursive and level != 1:
            raise ValueError("Flat layerwise memory only supports level 1")
        token_count = self.memory_tokens if token_count is None else int(token_count)
        if token_count < 1 or token_count > self.memory_tokens:
            raise ValueError(
                f"token_count must lie in [1, {self.memory_tokens}], got {token_count}"
            )
        tokens = self.slots[:, :token_count].to(device=device, dtype=dtype)
        if self.level_embedding is not None:
            tokens = tokens + self.level_embedding.weight[level].to(
                device=device,
                dtype=dtype,
            ).view(1, 1, self.hidden_dim)
        return tokens.expand(batch_size, -1, -1)

    def token_count_for_span(self, span: int) -> int:
        """Return the causal memory capacity assigned to one closed segment."""

        span = int(span)
        if span < 1:
            raise ValueError(f"memory span must be positive, got {span}")
        if self.dynamic_tokens_per_frame is None:
            return self.memory_tokens
        token_count = span * self.dynamic_tokens_per_frame
        if token_count > self.memory_tokens:
            raise ValueError(
                "dynamic segment exceeds configured memory capacity: "
                f"span={span}, rate={self.dynamic_tokens_per_frame}, "
                f"required={token_count}, maximum={self.memory_tokens}"
            )
        return token_count

    def build_freqs(
        self,
        *,
        endpoint: int,
        device: torch.device,
        token_count: int | None = None,
    ) -> torch.Tensor:
        endpoint = int(endpoint)
        if endpoint < 0 or endpoint >= int(self.temporal_freqs.shape[0]):
            raise ValueError(
                f"endpoint {endpoint} exceeds temporal RoPE capacity "
                f"{self.temporal_freqs.shape[0]}"
            )
        neutral = torch.cat(
            [
                self.temporal_freqs[endpoint],
                torch.ones_like(self.vertical_freqs[0]),
                torch.ones_like(self.horizontal_freqs[0]),
            ],
            dim=-1,
        )
        token_count = self.memory_tokens if token_count is None else int(token_count)
        if token_count < 1 or token_count > self.memory_tokens:
            raise ValueError(
                f"token_count must lie in [1, {self.memory_tokens}], got {token_count}"
            )
        return neutral.view(1, 1, -1).expand(token_count, -1, -1).to(
            device=device
        )

    def build_t_mod(
        self,
        endpoint_t_mod: torch.Tensor,
        *,
        token_count: int | None = None,
    ) -> torch.Tensor:
        if endpoint_t_mod.ndim == 3:
            endpoint_t_mod = endpoint_t_mod.unsqueeze(1)
        if endpoint_t_mod.ndim != 4 or endpoint_t_mod.shape[1] != 1:
            raise ValueError(
                "endpoint_t_mod must be [B,1,6,D] or [B,6,D], got "
                f"{tuple(endpoint_t_mod.shape)}"
            )
        token_count = self.memory_tokens if token_count is None else int(token_count)
        if token_count < 1 or token_count > self.memory_tokens:
            raise ValueError(
                f"token_count must lie in [1, {self.memory_tokens}], got {token_count}"
            )
        return endpoint_t_mod.expand(-1, token_count, -1, -1)


@dataclass(frozen=True)
class PackedSegment:
    kind: str
    frame_indices: tuple[int, ...]
    start: int
    stop: int
    level: int = 0
    span: int = 1


@dataclass(frozen=True)
class PackedMemoryTrainingLayout:
    segments: tuple[PackedSegment, ...]
    noisy_range: tuple[int, int]
    action_range: tuple[int, int]
    retained_ranges: tuple[tuple[int, int], ...]
    retained_clean_tokens: int
    attention_mask: torch.Tensor


def build_layerwise_training_layout(
    *,
    clean_frames: int,
    noisy_frames: int,
    tokens_per_frame: int,
    action_tokens: int,
    memory_tokens: int = 32,
    anchor_frames: int = 2,
    recent_frames: int = 4,
    group_size: int = 4,
    device: torch.device,
) -> PackedMemoryTrainingLayout:
    """Build packed hybrid-memory ranges and their exact attention visibility."""
    clean_frames = int(clean_frames)
    noisy_frames = int(noisy_frames)
    tokens_per_frame = int(tokens_per_frame)
    action_tokens = int(action_tokens)
    memory_tokens = int(memory_tokens)
    if min(clean_frames, noisy_frames, tokens_per_frame, action_tokens, memory_tokens) < 1:
        raise ValueError("training layout dimensions must all be positive")
    partition = partition_layerwise_history(
        clean_frames,
        anchor_frames=anchor_frames,
        recent_frames=recent_frames,
        group_size=group_size,
    )

    segments: list[PackedSegment] = []
    cursor = 0

    def append_segment(
        kind: str,
        frames: tuple[int, ...],
        length: int,
        *,
        level: int = 0,
        span: int = 1,
    ) -> None:
        nonlocal cursor
        segments.append(
            PackedSegment(
                kind=kind,
                frame_indices=frames,
                start=cursor,
                stop=cursor + int(length),
                level=int(level),
                span=int(span),
            )
        )
        cursor += int(length)

    for frame in partition.anchors:
        append_segment("anchor", (frame,), tokens_per_frame)
    for group in partition.memory_groups:
        append_segment("source", group, group_size * tokens_per_frame)
        append_segment(
            "memory",
            group,
            memory_tokens,
            level=1,
            span=group_size,
        )
    for frame in partition.raw_tail:
        append_segment("recent", (frame,), tokens_per_frame)

    noisy_range = (cursor, cursor + noisy_frames * tokens_per_frame)
    cursor = noisy_range[1]
    action_range = (cursor, cursor + action_tokens)
    total_tokens = action_range[1]
    mask = torch.zeros((total_tokens, total_tokens), dtype=torch.bool, device=device)
    def expose(rows: tuple[int, int], ranges: Sequence[tuple[int, int]]) -> None:
        for start, stop in ranges:
            mask[rows[0] : rows[1], start:stop] = True

    frame_ranges: dict[int, tuple[int, int]] = {}
    memory_ranges: list[tuple[tuple[int, ...], tuple[int, int]]] = []
    for segment in segments:
        if segment.kind == "source":
            for offset, frame in enumerate(segment.frame_indices):
                frame_ranges[frame] = (
                    segment.start + offset * tokens_per_frame,
                    segment.start + (offset + 1) * tokens_per_frame,
                )
        elif segment.kind == "memory":
            memory_ranges.append(
                (segment.frame_indices, (segment.start, segment.stop))
            )
        else:
            frame_ranges[segment.frame_indices[0]] = (segment.start, segment.stop)

    anchor_set = set(partition.anchors)
    for frame in range(clean_frames):
        rows = frame_ranges[frame]
        visible_frames = [
            anchor
            for anchor in partition.anchors
            if anchor <= frame
        ]
        recent_start = max(len(partition.anchors), frame - recent_frames + 1)
        visible_frames.extend(range(recent_start, frame + 1))
        visible_memories = [
            memory_range
            for group, memory_range in memory_ranges
            if group[-1] < frame
        ]
        expose(
            rows,
            [*(frame_ranges[index] for index in visible_frames), *visible_memories],
        )

    for group, memory_range in memory_ranges:
        previous_memories = [
            earlier_range
            for earlier_group, earlier_range in memory_ranges
            if earlier_group[-1] < group[0]
        ]
        expose(
            memory_range,
            [
                *(frame_ranges[index] for index in partition.anchors),
                *previous_memories,
                *(frame_ranges[index] for index in group),
                memory_range,
            ],
        )

    final_visible_ranges = [
        *(frame_ranges[index] for index in partition.anchors),
        *(memory_range for _, memory_range in memory_ranges),
        *(frame_ranges[index] for index in partition.recent),
    ]
    retained_ranges = tuple(sorted(set(final_visible_ranges)))
    expose(noisy_range, [*retained_ranges, noisy_range])
    expose(action_range, [*retained_ranges, action_range])
    return PackedMemoryTrainingLayout(
        segments=tuple(segments),
        noisy_range=noisy_range,
        action_range=action_range,
        retained_ranges=retained_ranges,
        retained_clean_tokens=sum(stop - start for start, stop in retained_ranges),
        attention_mask=mask,
    )


def build_dynamic_layerwise_training_layout(
    *,
    clean_frames: int,
    noisy_frames: int,
    tokens_per_frame: int,
    action_tokens: int,
    memory_groups: Sequence[Sequence[int]],
    memory_tokens: int = 8,
    memory_token_counts: Sequence[int] | None = None,
    anchor_frames: int = 2,
    recent_frames: int = 4,
    reader_use_anchor: bool = True,
    reader_use_memory: bool = True,
    reader_use_recent: bool = True,
    device: torch.device,
) -> PackedMemoryTrainingLayout:
    """Pack variable memories while retaining raw anchors and recent frames."""
    clean_frames = int(clean_frames)
    noisy_frames = int(noisy_frames)
    tokens_per_frame = int(tokens_per_frame)
    action_tokens = int(action_tokens)
    memory_tokens = int(memory_tokens)
    anchor_frames = int(anchor_frames)
    recent_frames = int(recent_frames)
    if min(clean_frames, noisy_frames, tokens_per_frame, action_tokens, memory_tokens) < 1:
        raise ValueError("training layout dimensions must all be positive")
    if anchor_frames < 0 or recent_frames < 0:
        raise ValueError("anchor_frames and recent_frames must be non-negative")

    groups = tuple(tuple(int(frame) for frame in group) for group in memory_groups)
    anchor_stop = min(anchor_frames, clean_frames)
    next_frame = anchor_stop
    if memory_token_counts is None:
        group_token_counts = (memory_tokens,) * len(groups)
    else:
        group_token_counts = tuple(int(value) for value in memory_token_counts)
        if len(group_token_counts) != len(groups):
            raise ValueError(
                "memory_token_counts must have one entry per dynamic memory group"
            )
        if any(value < 1 for value in group_token_counts):
            raise ValueError("dynamic memory token counts must all be positive")
        if any(value > memory_tokens for value in group_token_counts):
            raise ValueError(
                "dynamic memory token count exceeds the configured maximum "
                f"of {memory_tokens}"
            )

    for group in groups:
        if not 2 <= len(group) <= 8:
            raise ValueError(f"dynamic memory group length must be 2..8, got {len(group)}")
        expected = tuple(range(next_frame, next_frame + len(group)))
        if group != expected:
            raise ValueError(
                "dynamic memory groups must be contiguous after the raw anchors; "
                f"expected {expected}, got {group}"
            )
        next_frame = group[-1] + 1
    if groups and next_frame >= clean_frames:
        raise ValueError("dynamic memory groups must leave the current clean frame uncompressed")
    raw_tail = tuple(range(next_frame, clean_frames))

    segments: list[PackedSegment] = []
    cursor = 0

    def append_segment(
        kind: str,
        frames: tuple[int, ...],
        length: int,
        *,
        level: int = 0,
        span: int = 1,
    ) -> None:
        nonlocal cursor
        segments.append(
            PackedSegment(
                kind,
                frames,
                cursor,
                cursor + int(length),
                level=int(level),
                span=int(span),
            )
        )
        cursor += int(length)

    for frame in range(anchor_stop):
        append_segment("anchor", (frame,), tokens_per_frame)
    for group, token_count in zip(groups, group_token_counts):
        append_segment("source", group, len(group) * tokens_per_frame)
        append_segment("memory", group, token_count, level=1, span=len(group))
    for frame in raw_tail:
        append_segment("recent", (frame,), tokens_per_frame)

    noisy_range = (cursor, cursor + noisy_frames * tokens_per_frame)
    cursor = noisy_range[1]
    action_range = (cursor, cursor + action_tokens)
    mask = torch.zeros((action_range[1], action_range[1]), dtype=torch.bool, device=device)

    def expose(rows: tuple[int, int], ranges: Sequence[tuple[int, int]]) -> None:
        for start, stop in ranges:
            mask[rows[0] : rows[1], start:stop] = True

    frame_ranges: dict[int, tuple[int, int]] = {}
    memory_ranges: list[tuple[tuple[int, ...], tuple[int, int]]] = []
    for segment in segments:
        if segment.kind == "source":
            for offset, frame in enumerate(segment.frame_indices):
                frame_ranges[frame] = (
                    segment.start + offset * tokens_per_frame,
                    segment.start + (offset + 1) * tokens_per_frame,
                )
        elif segment.kind == "memory":
            memory_ranges.append((segment.frame_indices, (segment.start, segment.stop)))
        else:
            frame_ranges[segment.frame_indices[0]] = (segment.start, segment.stop)

    anchors = tuple(range(anchor_stop))
    group_start_for_frame = {
        frame: group[0]
        for group in groups
        for frame in group
    }
    for frame in range(clean_frames):
        if frame < anchor_stop:
            # Without persistent anchors, the arriving frame still has to see
            # itself.  It is an ephemeral current/recent observation online,
            # then disappears once the anchor warm-up advances past it.
            visible_frames = (
                [anchor for anchor in anchors if anchor <= frame]
                if reader_use_anchor
                else [frame]
            )
        else:
            visible_frames = (
                list(anchors) if reader_use_anchor else []
            )
            # Raw source tokens are encoded causally inside their own open
            # segment.  They deliberately do not read earlier memories:
            # with delayed boundary confirmation those memories may not have
            # existed when the raw token arrived online.  The memory slots
            # themselves read both the completed source and earlier memories.
            open_start = group_start_for_frame.get(frame, next_frame)
            visible_frames.extend(range(open_start, frame + 1))
        expose(
            frame_ranges[frame],
            [frame_ranges[index] for index in visible_frames],
        )

    for group, memory_range in memory_ranges:
        previous = [
            earlier_range
            for earlier_group, earlier_range in memory_ranges
            if earlier_group[-1] < group[0]
        ]
        expose(
            memory_range,
            [
                *(
                    [frame_ranges[index] for index in anchors]
                    if reader_use_anchor
                    else []
                ),
                *previous,
                *(frame_ranges[index] for index in group),
                memory_range,
            ],
        )

    memories = [memory_range for _, memory_range in memory_ranges]
    # The Reader consumes the complete still-open raw suffix.  Its length is
    # determined by the causal segmenter (and bounded by max segment length),
    # rather than clipped to a fixed recent-N window.
    retained_ranges = tuple(
        sorted(
            set(
                [
                    *(frame_ranges[frame] for frame in anchors),
                    *memories,
                    *(frame_ranges[frame] for frame in raw_tail),
                ]
            )
        )
    )
    reader_recent_ranges = [frame_ranges[frame] for frame in raw_tail]
    if (
        reader_use_recent
        and not reader_use_anchor
        and clean_frames <= anchor_frames
    ):
        # Match the online warm-up state: at decisions 0/1 only the newest
        # frame is a temporary recent observation; neither frame persists as
        # an anchor into decision 2.
        reader_recent_ranges = [frame_ranges[clean_frames - 1]]
    reader_ranges = [
        *([frame_ranges[frame] for frame in anchors] if reader_use_anchor else []),
        *(memories if reader_use_memory else []),
        *(reader_recent_ranges if reader_use_recent else []),
    ]
    expose(noisy_range, [*reader_ranges, noisy_range])
    expose(action_range, [*reader_ranges, action_range])
    return PackedMemoryTrainingLayout(
        segments=tuple(segments),
        noisy_range=noisy_range,
        action_range=action_range,
        retained_ranges=retained_ranges,
        retained_clean_tokens=sum(stop - start for start, stop in retained_ranges),
        attention_mask=mask,
    )


def build_recursive_layerwise_training_layout(
    *,
    clean_frames: int,
    noisy_frames: int,
    tokens_per_frame: int,
    action_tokens: int,
    memory_tokens: int = 32,
    anchor_frames: int = 2,
    recent_frames: int = 4,
    group_size: int = 4,
    max_levels: int = 8,
    recursive: bool = True,
    device: torch.device,
) -> PackedMemoryTrainingLayout:
    """Build a topological base-4 hierarchy with exact online visibility."""
    clean_frames = int(clean_frames)
    noisy_frames = int(noisy_frames)
    tokens_per_frame = int(tokens_per_frame)
    action_tokens = int(action_tokens)
    memory_tokens = int(memory_tokens)
    anchor_frames = int(anchor_frames)
    recent_frames = int(recent_frames)
    group_size = int(group_size)
    max_levels = int(max_levels)
    if min(
        clean_frames,
        noisy_frames,
        tokens_per_frame,
        action_tokens,
        memory_tokens,
        recent_frames,
        group_size,
        max_levels,
    ) < 1:
        raise ValueError("recursive training layout dimensions must all be positive")
    if anchor_frames < 0:
        raise ValueError("anchor_frames must be non-negative")
    if group_size != 4:
        raise ValueError("Recursive layerwise memory requires group_size=4")

    segments: list[PackedSegment] = []
    visibility: list[tuple[tuple[int, int], tuple[tuple[int, int], ...]]] = []
    frame_ranges: dict[int, tuple[int, int]] = {}
    anchor_ranges: list[tuple[int, int]] = []
    active: list[tuple[CacheUnit, tuple[int, int]]] = []
    raw_group: list[tuple[CacheUnit, tuple[int, int]]] = []
    cursor = 0

    def append_segment(
        kind: str,
        frames: tuple[int, ...],
        length: int,
        *,
        level: int = 0,
        span: int = 1,
    ) -> tuple[int, int]:
        nonlocal cursor
        token_range = (cursor, cursor + int(length))
        segments.append(
            PackedSegment(
                kind=kind,
                frame_indices=frames,
                start=token_range[0],
                stop=token_range[1],
                level=int(level),
                span=int(span),
            )
        )
        cursor = token_range[1]
        return token_range

    anchor_stop = min(anchor_frames, clean_frames)
    for frame in range(clean_frames):
        if frame < anchor_stop:
            token_range = append_segment("anchor", (frame,), tokens_per_frame)
            frame_ranges[frame] = token_range
            anchor_ranges.append(token_range)
            visibility.append((token_range, tuple(anchor_ranges)))
            continue

        raw_unit = CacheUnit(
            kind="raw",
            endpoint=frame,
            span=1,
            token_count=tokens_per_frame,
            level=0,
        )
        token_range = append_segment("source", (frame,), tokens_per_frame)
        frame_ranges[frame] = token_range
        recent_start = max(anchor_stop, frame - recent_frames + 1)
        raw_visible = tuple(
            frame_ranges[index] for index in range(recent_start, frame + 1)
        )
        visibility.append(
            (
                token_range,
                tuple(anchor_ranges)
                + tuple(memory_range for _, memory_range in active)
                + raw_visible,
            )
        )
        raw_group.append((raw_unit, token_range))
        if len(raw_group) < group_size:
            continue

        raw_children = tuple(unit for unit, _ in raw_group)
        child_ranges = tuple(child_range for _, child_range in raw_group)
        raw_group.clear()
        parent = _memory_parent(
            raw_children,
            memory_tokens=memory_tokens,
            max_levels=max_levels,
        )
        parent_range = append_segment(
            "memory",
            tuple(range(parent.start, parent.endpoint + 1)),
            memory_tokens,
            level=int(parent.level),
            span=int(parent.span),
        )
        visibility.append(
            (
                parent_range,
                tuple(anchor_ranges)
                + tuple(memory_range for _, memory_range in active)
                + child_ranges
                + (parent_range,),
            )
        )
        active.append((parent, parent_range))

        if not bool(recursive):
            continue
        while len(active) >= group_size:
            candidate = active[-group_size:]
            if len({int(unit.level) for unit, _ in candidate}) != 1:
                break
            children = tuple(unit for unit, _ in candidate)
            child_ranges = tuple(child_range for _, child_range in candidate)
            prefix = active[:-group_size]
            parent = _memory_parent(
                children,
                memory_tokens=memory_tokens,
                max_levels=max_levels,
            )
            parent_range = append_segment(
                "memory",
                tuple(range(parent.start, parent.endpoint + 1)),
                memory_tokens,
                level=int(parent.level),
                span=int(parent.span),
            )
            visibility.append(
                (
                    parent_range,
                    tuple(anchor_ranges)
                    + tuple(memory_range for _, memory_range in prefix)
                    + child_ranges
                    + (parent_range,),
                )
            )
            active = [*prefix, (parent, parent_range)]

    noisy_range = (cursor, cursor + noisy_frames * tokens_per_frame)
    cursor = noisy_range[1]
    action_range = (cursor, cursor + action_tokens)
    total_tokens = action_range[1]
    mask = torch.zeros((total_tokens, total_tokens), dtype=torch.bool, device=device)

    def expose(
        rows: tuple[int, int],
        ranges: Sequence[tuple[int, int]],
    ) -> None:
        for start, stop in ranges:
            mask[rows[0] : rows[1], start:stop] = True

    for rows, ranges in visibility:
        expose(rows, ranges)
    recent_start = max(anchor_stop, clean_frames - recent_frames)
    final_visible_ranges = [
        *anchor_ranges,
        *(memory_range for _, memory_range in active),
        *(frame_ranges[index] for index in range(recent_start, clean_frames)),
    ]
    retained_ranges = tuple(sorted(set(final_visible_ranges)))
    expose(noisy_range, [*retained_ranges, noisy_range])
    expose(action_range, [*retained_ranges, action_range])
    return PackedMemoryTrainingLayout(
        segments=tuple(segments),
        noisy_range=noisy_range,
        action_range=action_range,
        retained_ranges=retained_ranges,
        retained_clean_tokens=sum(stop - start for start, stop in retained_ranges),
        attention_mask=mask,
    )


@dataclass(frozen=True)
class NativeBlock:
    """One recursively compressible native pre-DiT memory unit."""

    tokens: torch.Tensor
    endpoint: int
    span: int = 1
    level: int = 0

    def __post_init__(self) -> None:
        if self.tokens.ndim != 3:
            raise ValueError(
                f"NativeBlock.tokens must be [B,N,D], got {tuple(self.tokens.shape)}"
            )
        if int(self.endpoint) < 0:
            raise ValueError(f"endpoint must be non-negative, got {self.endpoint}")
        if int(self.span) <= 0:
            raise ValueError(f"span must be positive, got {self.span}")
        if int(self.level) < 0:
            raise ValueError(f"level must be non-negative, got {self.level}")


@dataclass(frozen=True)
class NativeCacheState:
    """Immutable retained native blocks paired with their all-layer K/V."""

    blocks: tuple[NativeBlock, ...]
    kv_cache: tuple[dict[str, torch.Tensor], ...]

    def __post_init__(self) -> None:
        if not self.blocks:
            raise ValueError("NativeCacheState requires at least one retained block")
        if not self.kv_cache:
            raise ValueError("NativeCacheState requires a non-empty layer cache")
        expected_tokens = sum(block.tokens.shape[1] for block in self.blocks)
        for layer_index, layer in enumerate(self.kv_cache):
            if "k" not in layer or "v" not in layer:
                raise ValueError(
                    f"kv_cache[{layer_index}] must contain both k and v"
                )
            if layer["k"].shape[1] != expected_tokens:
                raise ValueError(
                    f"kv_cache[{layer_index}] has {layer['k'].shape[1]} tokens, "
                    f"expected {expected_tokens} from retained blocks"
                )
            if layer["v"].shape[1] != expected_tokens:
                raise ValueError(
                    f"kv_cache[{layer_index}] value length does not match blocks"
                )


def summarize_native_cache_state(
    state: NativeCacheState | LayerwiseMemoryState | DynamicLayerwiseMemoryState,
) -> dict[str, object]:
    """Return JSON-stable structural telemetry for one committed cache state."""
    if isinstance(state, (LayerwiseMemoryState, DynamicLayerwiseMemoryState)):
        unit_kind_histogram: dict[str, int] = {}
        level_histogram: dict[str, int] = {}
        for unit in state.units:
            unit_kind_histogram[unit.kind] = unit_kind_histogram.get(unit.kind, 0) + 1
            if unit.kind == "memory":
                key = str(int(unit.level))
                level_histogram[key] = level_histogram.get(key, 0) + 1
        summary = {
            "retained_units": len(state.units),
            "retained_tokens": state.retained_tokens,
            "represented_frames": state.represented_frames,
            "latest_endpoint": int(state.units[-1].endpoint),
            "unit_kind_histogram": dict(sorted(unit_kind_histogram.items())),
            "level_histogram": dict(
                sorted(level_histogram.items(), key=lambda item: int(item[0]))
            ),
            "max_level": max(
                (int(unit.level) for unit in state.units if unit.kind == "memory"),
                default=0,
            ),
            "last_carry_levels": [
                int(level) for level in getattr(state, "carry_levels", ())
            ],
            "kv_layers": len(state.kv_cache),
            "kv_tokens_per_layer": int(state.kv_cache[0]["k"].shape[1]),
        }
        if isinstance(state, DynamicLayerwiseMemoryState):
            summary["open_start"] = int(state.open_start)
        return summary
    level_histogram: dict[str, int] = {}
    for block in state.blocks:
        key = str(int(block.level))
        level_histogram[key] = level_histogram.get(key, 0) + 1
    retained_tokens = sum(int(block.tokens.shape[1]) for block in state.blocks)
    return {
        "retained_blocks": len(state.blocks),
        "retained_tokens": retained_tokens,
        "represented_frames": sum(int(block.span) for block in state.blocks),
        "latest_endpoint": int(state.blocks[-1].endpoint),
        "level_histogram": dict(sorted(level_histogram.items(), key=lambda item: int(item[0]))),
        "kv_layers": len(state.kv_cache),
        "kv_tokens_per_layer": int(state.kv_cache[0]["k"].shape[1]),
    }


class NativeBlockCompressor(nn.Module):
    """Consolidate four pre-DiT native blocks into the final block's anchors."""

    def __init__(
        self,
        video_block: nn.Module,
        *,
        group_size: int = 4,
        alpha_max: float = 1.0,
        max_levels: int = 8,
    ) -> None:
        super().__init__()
        if int(group_size) != 4:
            raise ValueError(f"Native consolidation requires group_size=4, got {group_size}")
        if float(alpha_max) <= 0.0:
            raise ValueError(f"alpha_max must be positive, got {alpha_max}")
        if int(max_levels) < 2:
            raise ValueError(f"max_levels must be at least 2, got {max_levels}")

        self.group_size = int(group_size)
        self.alpha_max = float(alpha_max)
        self.max_levels = int(max_levels)
        self.hidden_dim = int(video_block.hidden_dim)
        self.num_heads = int(video_block.num_heads)
        self.head_dim = int(video_block.attn_head_dim)
        if self.hidden_dim != self.num_heads * self.head_dim:
            raise ValueError(
                "Video attention width must equal hidden width, got "
                f"{self.hidden_dim} vs {self.num_heads}x{self.head_dim}"
            )

        self.input_norm = copy.deepcopy(video_block.norm1)
        self.attention = copy.deepcopy(video_block.self_attn)
        self.ordinal_embedding = nn.Embedding(self.group_size, self.hidden_dim)
        self.level_embedding = nn.Embedding(self.max_levels, self.hidden_dim)
        self.log_span_projection = nn.Linear(1, self.hidden_dim, bias=False)
        nn.init.zeros_(self.ordinal_embedding.weight)
        nn.init.zeros_(self.level_embedding.weight)
        nn.init.zeros_(self.log_span_projection.weight)
        self.raw_alpha = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        native_dtype = next(video_block.parameters()).dtype
        self.to(dtype=native_dtype)

    def alpha(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return (
            self.alpha_max * torch.tanh(self.raw_alpha.float())
        ).to(device=device, dtype=dtype)

    def initialize_from_video_block(self, video_block: nn.Module) -> None:
        """Refresh native projections after loading a legacy FullKV checkpoint."""
        self.input_norm.load_state_dict(video_block.norm1.state_dict(), strict=True)
        self.attention.load_state_dict(video_block.self_attn.state_dict(), strict=True)

    def consolidate_blocks(
        self,
        blocks: Sequence[NativeBlock],
        *,
        block_freqs: torch.Tensor,
    ) -> NativeBlock:
        if len(blocks) != self.group_size:
            raise ValueError(
                f"Expected exactly {self.group_size} blocks, got {len(blocks)}"
            )
        reference = blocks[0]
        if any(block.tokens.shape != reference.tokens.shape for block in blocks[1:]):
            raise ValueError("All native blocks must have the same token shape")
        if any(block.level != reference.level for block in blocks[1:]):
            raise ValueError("All native blocks in a carry must have the same level")
        if any(block.span != reference.span for block in blocks[1:]):
            raise ValueError("All native blocks in a carry must have the same span")
        expected_endpoints = [
            reference.endpoint + index * reference.span
            for index in range(self.group_size)
        ]
        actual_endpoints = [block.endpoint for block in blocks]
        if actual_endpoints != expected_endpoints:
            raise ValueError(
                "Native block endpoints must be contiguous for their span: "
                f"expected {expected_endpoints}, got {actual_endpoints}"
            )

        tokens = torch.stack([block.tokens for block in blocks], dim=1)
        batch = tokens.shape[0]
        levels = torch.full(
            (batch, self.group_size),
            reference.level,
            dtype=torch.long,
            device=tokens.device,
        )
        spans = torch.full(
            (batch, self.group_size),
            reference.span,
            dtype=torch.long,
            device=tokens.device,
        )
        return NativeBlock(
            tokens=self(
                block_tokens=tokens,
                block_freqs=block_freqs,
                levels=levels,
                spans=spans,
            ),
            endpoint=blocks[-1].endpoint,
            span=sum(block.span for block in blocks),
            level=reference.level + 1,
        )

    def forward(
        self,
        *,
        block_tokens: torch.Tensor,
        block_freqs: torch.Tensor,
        levels: torch.Tensor,
        spans: torch.Tensor,
    ) -> torch.Tensor:
        if block_tokens.ndim != 4:
            raise ValueError(
                f"block_tokens must be [B,4,N,D], got {tuple(block_tokens.shape)}"
            )
        batch, groups, tokens_per_block, hidden_dim = block_tokens.shape
        if groups != self.group_size or hidden_dim != self.hidden_dim:
            raise ValueError(
                "block_tokens shape mismatch: expected "
                f"[B,{self.group_size},N,{self.hidden_dim}], got {tuple(block_tokens.shape)}"
            )
        if levels.shape != (batch, self.group_size):
            raise ValueError(
                f"levels must be {(batch, self.group_size)}, got {tuple(levels.shape)}"
            )
        if spans.shape != (batch, self.group_size):
            raise ValueError(
                f"spans must be {(batch, self.group_size)}, got {tuple(spans.shape)}"
            )
        levels = levels.to(device=block_tokens.device, dtype=torch.long)
        spans = spans.to(device=block_tokens.device, dtype=torch.long)
        if bool((levels < 0).any().item()) or bool((levels >= self.max_levels).any().item()):
            raise ValueError(f"levels must lie in [0, {self.max_levels})")
        if bool((spans <= 0).any().item()):
            raise ValueError("spans must all be positive")

        sequence_len = self.group_size * tokens_per_block
        if block_freqs.ndim == 3:
            if block_freqs.shape[0] != sequence_len:
                raise ValueError(
                    f"block_freqs sequence must be {sequence_len}, got {block_freqs.shape[0]}"
                )
            block_freqs = block_freqs.unsqueeze(0).expand(batch, -1, -1, -1)
        elif block_freqs.ndim == 4:
            if block_freqs.shape[0] != batch or block_freqs.shape[1] != sequence_len:
                raise ValueError(
                    "batched block_freqs must be [B,4N,1,R], got "
                    f"{tuple(block_freqs.shape)}"
                )
        else:
            raise ValueError(
                f"block_freqs must be [4N,1,R] or [B,4N,1,R], got {tuple(block_freqs.shape)}"
            )

        ordinal = self.ordinal_embedding.weight.to(
            device=block_tokens.device, dtype=block_tokens.dtype
        ).view(1, self.group_size, 1, self.hidden_dim)
        level = self.level_embedding(levels).to(dtype=block_tokens.dtype).unsqueeze(2)
        log_span = torch.log2(spans.to(dtype=torch.float32)).to(
            dtype=block_tokens.dtype
        ).unsqueeze(-1)
        span = self.log_span_projection(log_span).unsqueeze(2)
        enriched = block_tokens + ordinal + level + span

        source = enriched.reshape(batch, sequence_len, self.hidden_dim)
        anchor = enriched[:, -1]
        q = self.attention.norm_q(self.attention.q(self.input_norm(anchor)))
        k = self.attention.norm_k(self.attention.k(self.input_norm(source)))
        v = self.attention.v(self.input_norm(source))
        q = rope_apply(q, block_freqs[:, -tokens_per_block:], self.num_heads)
        k = rope_apply(k, block_freqs, self.num_heads)

        q = q.view(batch, tokens_per_block, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, sequence_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, sequence_len, self.num_heads, self.head_dim).transpose(1, 2)
        attended = F.scaled_dot_product_attention(q, k, v)
        attended = attended.transpose(1, 2).reshape(
            batch, tokens_per_block, self.hidden_dim
        )
        branch = self.attention.o(attended)
        return block_tokens[:, -1] + self.alpha(
            dtype=block_tokens.dtype, device=block_tokens.device
        ) * branch
