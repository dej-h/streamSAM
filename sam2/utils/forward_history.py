"""Forward-only retention derived from SAM's spatial and pointer selectors."""

from dataclasses import dataclass
from typing import MutableMapping, TypeVar

T = TypeVar("T")


def discard_before(values: MutableMapping[int, T], cutoff: int, *, pin_zero: bool = False) -> int:
    """Drop expired entries from one bounded index without retaining its values."""
    expired = tuple(index for index in values if index < cutoff and not (pin_zero and index == 0))
    for index in expired:
        del values[index]
    return len(expired)


@dataclass(frozen=True)
class ForwardHistoryPolicy:
    num_maskmem: int
    temporal_stride: int
    use_object_pointers: bool
    max_object_pointers: int
    max_objects: int = 1

    def __post_init__(self) -> None:
        if self.num_maskmem < 0 or min(self.temporal_stride, self.max_object_pointers, self.max_objects) <= 0:
            raise ValueError("invalid forward memory selector or object capacity")

    @property
    def non_conditioning_capacity(self) -> int:
        """A suffix preserving every possible future spatial/pointer lookup.

        Spatial selection includes f-1 and stride-aligned frames as far back as
        floor((f-2)/stride)*stride-(num_maskmem-3)*stride. The worst stride phase
        requires 1+(num_maskmem-2)*stride recent outputs after each commit.
        Pointer selection independently reaches max_object_pointers-1 frames.
        """
        if self.num_maskmem == 0:
            return 0
        spatial = 1 + (self.num_maskmem - 2) * self.temporal_stride if self.num_maskmem >= 2 else 0
        pointers = self.max_object_pointers - 1 if self.use_object_pointers else 0
        return max(spatial, pointers)

    def cutoff_after(self, frame_idx: int) -> int:
        return max(0, frame_idx - self.non_conditioning_capacity + 1)


@dataclass(frozen=True)
class ForwardHistoryStats:
    non_conditioning_capacity: int
    max_objects: int
    committed_frames: int
    evicted_frames: int
    retained_conditioning_frames: int
    retained_non_conditioning_frames: int
    retained_tracking_frames: int
    maximum_non_conditioning_frames: int


@dataclass
class ForwardHistoryRetention:
    policy: ForwardHistoryPolicy
    last_committed_frame: int = -1
    evicted_frames: int = 0
    maximum_non_conditioning_frames: int = 0

    def require_next(self, frame_idx: int, reverse: bool = False) -> None:
        if reverse or frame_idx != self.last_committed_frame + 1:
            raise ValueError("bounded history requires consecutive forward frames starting at zero")

    def reset(self) -> None:
        self.last_committed_frame = -1
        self.evicted_frames = 0
        self.maximum_non_conditioning_frames = 0
