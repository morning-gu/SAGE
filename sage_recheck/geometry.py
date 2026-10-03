"""Geometric utility functions (ported from StudyBuddyAgent)."""
from __future__ import annotations

import numpy as np


def calculate_angle(v1, v2) -> float:
    """Angle between two vectors in degrees (0-180)."""
    v1, v2 = np.array(v1), np.array(v2)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def point_line_ratio(line_begin, line_end, point) -> float:
    """Projection ratio of *point* onto segment [line_begin, line_end] in [0, 1]."""
    lb = np.array(line_begin, dtype=np.float32)
    le = np.array(line_end, dtype=np.float32)
    pt = np.array(point, dtype=np.float32)
    seg = le - lb
    return float(np.dot(pt - lb, seg) / (np.dot(seg, seg) + 1e-8))


def point_distance(p1, p2) -> float:
    """Euclidean distance between two [x, y] points."""
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def dist_to_box(x: float, y: float, box) -> float:
    """Distance from point (x, y) to the nearest edge of box [x1, y1, x2, y2].

    Returns 0 if the point is inside the box.
    """
    x1, y1, x2, y2 = box[:4]
    dx = max(x1 - x, 0, x - x2)
    dy = max(y1 - y, 0, y - y2)
    return float(np.hypot(dx, dy))
