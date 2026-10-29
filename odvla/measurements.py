"""Deterministic measurements; no mass, volume, or contact inferred from masks."""
from __future__ import annotations
import numpy as np

METRIC_NAMES = ["centroid_u", "centroid_v", "bbox_width", "bbox_height", "visible_area"]
METRIC_UNITS = ["image_fraction"] * 5


def mask_metrics(masks):
    masks = np.asarray(masks)
    if masks.ndim != 4 or masks.dtype != np.bool_:
        raise ValueError("Expected bool masks [T,O,H,W]")
    t, o, h, w = masks.shape
    features = np.zeros((t, o, 5), dtype=np.float32)
    valid = np.zeros_like(features, dtype=bool)
    for ti in range(t):
        for oi in range(o):
            ys, xs = np.nonzero(masks[ti, oi])
            if len(xs):
                features[ti, oi] = [(xs.mean() + 0.5) / w, (ys.mean() + 0.5) / h,
                                    (xs.max() - xs.min() + 1) / w,
                                    (ys.max() - ys.min() + 1) / h,
                                    len(xs) / (h * w)]
                valid[ti, oi] = True
    return features, valid


def backproject_visible_point(mask, depth_z, intrinsics, base_from_camera):
    """Median observed surface point. Depth is optical-axis z in METERS, not ray distance.

    This point is not an object's center of mass or a grasp pose. Reject empty depth.
    """
    mask, depth_z = np.asarray(mask, bool), np.asarray(depth_z)
    k, transform = np.asarray(intrinsics), np.asarray(base_from_camera)
    if mask.shape != depth_z.shape or k.shape != (3, 3) or transform.shape != (4, 4):
        raise ValueError("Expected matching mask/depth, K[3,3], T_base_camera[4,4]")
    good = mask & np.isfinite(depth_z) & (depth_z > 0)
    ys, xs = np.nonzero(good)
    if not len(xs):
        return np.zeros(3, np.float32), False
    pixels = np.stack([xs, ys, np.ones(len(xs))], axis=0)
    points = np.linalg.solve(k, pixels) * depth_z[good][None]
    point = np.median(points, axis=1)
    result = transform @ np.r_[point, 1.0]
    return result[:3].astype(np.float32), True
