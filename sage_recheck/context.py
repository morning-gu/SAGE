"""Detection context and per-person data structures."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sage_recheck.features import FeatureReport


@dataclass
class DetectedObject:
    """Image-level object detection result."""
    box: list[float]
    class_id: int
    class_name: str


@dataclass
class PersonData:
    """Single person's complete detection data."""
    box: list[float]
    keypoints: dict[str, list[float]] | None
    kp_valid: dict[str, list[float]]
    head_height: float | None
    depth_info: dict[str, float]
    features: "FeatureReport | None" = None
    face_data: dict[str, Any] | None = None


@dataclass
class DetectionContext:
    """Layer-1 output, Layer-2 input."""
    image_base64: str
    image_width: int = 0
    image_height: int = 0
    persons: list[PersonData] = field(default_factory=list)
    objects: list[DetectedObject] = field(default_factory=list)
    raw_data: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def has_person(self) -> bool:
        return len(self.persons) > 0
