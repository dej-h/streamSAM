"""Select spatially distributed prompt points inside segmented regions."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class _InteriorRegion:
    area: int
    coordinates: NDArray[np.float32]
    clearance_squared: NDArray[np.float32]
    center: tuple[float, float]


def sample_mask_interior_points(
    mask: NDArray[np.bool_], *, max_points: int = 64
) -> NDArray[np.float32]:
    """Return up to ``max_points`` foreground pixel coordinates in (x, y) order.

    Each connected component has its own interior: pixels at least half its
    maximum distance from the background. Component centers are selected from
    foreground pixels, so holes and concavities cannot place a point outside
    the mask. Thin fragments retain their own interior even beside thick ones.

    Components receive one point each, then share the remaining budget in
    proportion to their area. If components outnumber the budget, the largest
    components are represented. Within each interior, farthest-point sampling
    spreads points while favoring greater boundary clearance. An empty mask
    returns an empty (0, 2) array.
    """
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("mask must be a two-dimensional boolean array")
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    if not mask.any():
        return np.empty((0, 2), dtype=np.float32)

    binary = mask.astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    distances = cv2.distanceTransform(
        np.pad(binary, 1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )[1:-1, 1:-1]
    # Squared distances between pixel centers are integers. Quantization removes
    # floating-point ambiguity when symmetric regions have equal clearance.
    distances = np.rint(distances**2)
    component_ids = sorted(
        range(1, count), key=lambda index: (-int(stats[index, cv2.CC_STAT_AREA]), index)
    )[:max_points]
    regions: list[_InteriorRegion] = []
    for component_id in component_ids:
        x, y, width, height, area = (int(value) for value in stats[component_id])
        local_labels = labels[y : y + height, x : x + width]
        local_distances = distances[y : y + height, x : x + width]
        membership = local_labels == component_id
        peak = float(local_distances[membership].max())
        rows, columns = np.nonzero(membership & (local_distances >= peak * 0.25))
        coordinates = np.column_stack((columns + x, rows + y)).astype(np.float32)
        regions.append(
            _InteriorRegion(
                area=area,
                coordinates=coordinates,
                clearance_squared=local_distances[rows, columns] / peak,
                center=(
                    float(centers[component_id, 0]),
                    float(centers[component_id, 1]),
                ),
            )
        )

    budgets = [1] * len(regions)
    remaining = min(max_points, sum(len(region.coordinates) for region in regions))
    for _ in range(remaining - len(regions)):
        candidates = (
            index
            for index, region in enumerate(regions)
            if budgets[index] < len(region.coordinates)
        )
        selected = max(
            candidates, key=lambda index: regions[index].area / budgets[index]
        )
        budgets[selected] += 1
    return np.concatenate(
        [_sample_region(region, budget) for region, budget in zip(regions, budgets)],
        axis=0,
    )


def _sample_region(region: _InteriorRegion, count: int) -> NDArray[np.float32]:
    coordinates = region.coordinates
    deepest = np.flatnonzero(region.clearance_squared == 1)
    center = np.asarray(region.center, dtype=np.float32)
    center_distances = np.sum((coordinates[deepest] - center) ** 2, axis=1)
    selected = int(deepest[np.argmin(center_distances)])
    nearest = np.full(len(coordinates), np.inf, dtype=np.float32)
    indices = np.empty(count, dtype=np.intp)
    indices[0] = selected
    weights = region.clearance_squared
    for index in range(1, count):
        squared_distance = np.sum((coordinates - coordinates[selected]) ** 2, axis=1)
        np.minimum(nearest, squared_distance, out=nearest)
        selected = int(np.argmax(nearest * weights))
        indices[index] = selected
    return coordinates[indices]
