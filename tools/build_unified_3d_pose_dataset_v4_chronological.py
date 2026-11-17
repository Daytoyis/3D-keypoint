#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a unified YOLO11-Pose 3D-cuboid dataset from KITTI, UrbanOmniView/UE5 and DAIR-V2X-I.

Canonical corner identity follows the official AIR-THU DAIR-V2X LiDAR cuboid order:
  0: (+L,+W,bottom)   1: (+L,-W,bottom)
  2: (-L,-W,bottom)   3: (-L,+W,bottom)
  4..7: corresponding top corners.

YOLO label format (21 columns):
  cls cx cy w h x1 y1 ... x8 y8

Important:
- The keypoints are 2D image projections of semantically ordered 3D cuboid corners.
- Source datasets are never modified.
- Existing output directories are never overwritten.
- Images are symlinked by default; --copy-images makes physical copies.
- KITTI testing/ is excluded because it has no public ground-truth labels.
- DAIR annotations are in the Virtual-LiDAR coordinate system and are projected through
  the Virtual-LiDAR-to-camera extrinsic and camera intrinsic matrices.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

# -----------------------------------------------------------------------------
# Defaults: match the directory layout previously supplied by the user.
# -----------------------------------------------------------------------------
DEFAULT_KITTI = Path('/mnt/sdb/lys/MonoCoP/data/KITTIDataset')
DEFAULT_UE5 = Path('/mnt/sdb/lys/yolo11/datasets/UrbanOmniView')
DEFAULT_DAIR = Path('/mnt/sdb/lys/yolo11/datasets/DAIR-V2X-I')
DEFAULT_OUT = Path('/mnt/sdb/lys/yolo11/datasets/YOLO11_3D_Pose_3Source')

CLASS_NAMES = ['car', 'person', 'bike']
CLASS_ID = {'car': 0, 'person': 1, 'bike': 2}

# Fixed taxonomy aligned with the UrbanOmniView three-class definition:
#   0 Car    : passenger cars, trucks, vans, buses
#   1 Person : pedestrians
#   2 Bike   : bicycles, motorcycles, scooters and their rider/vehicle annotations
# Tricyclist/three-wheeler, traffic cone, barrowlist and all unknown classes are excluded.
UE5_CLASS_MAP = {
    'car': 0, 'truck': 0, 'van': 0, 'bus': 0,
    'person': 1, 'pedestrian': 1,
    'bike': 2, 'bicycle': 2, 'cyclist': 2,
    'motorcycle': 2, 'motorcyclist': 2, 'scooter': 2,
}
KITTI_CLASS_MAP = {
    'Car': 0, 'Van': 0, 'Truck': 0,
    'Pedestrian': 1,
    'Cyclist': 2,
}
DAIR_CLASS_MAP = {
    'Car': 0, 'Van': 0, 'Truck': 0, 'Bus': 0,
    'Pedestrian': 1,
    'Cyclist': 2, 'Motorcyclist': 2,
    'Bicycle': 2, 'Bike': 2, 'Motorcycle': 2, 'Scooter': 2,
}

# DAIR official local cuboid corner identity, expressed as (longitudinal, lateral, vertical) signs.
# Bottom first; 0->1 is one lateral edge, 1->2 is the opposite longitudinal edge.
CANONICAL_PATTERNS = [
    ( 1,  1, -1), ( 1, -1, -1), (-1, -1, -1), (-1,  1, -1),
    ( 1,  1,  1), ( 1, -1,  1), (-1, -1,  1), (-1,  1,  1),
]
CANONICAL_PATTERN_SET = set(CANONICAL_PATTERNS)
# Horizontal image flip mirrors the object's lateral side: front-left <-> front-right,
# rear-left <-> rear-right, with the corresponding top corners.
FLIP_IDX = [1, 0, 3, 2, 5, 4, 7, 6]
BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]
COLORS = [(40, 220, 70), (40, 170, 255), (255, 165, 35)]


# -----------------------------------------------------------------------------
# Basic geometry / IO helpers
# -----------------------------------------------------------------------------
def read_image_size(path: Path) -> tuple[int, int]:
    if cv2 is not None:
        im = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if im is None:
            raise ValueError(f'Cannot read image: {path}')
        h, w = im.shape[:2]
        return int(w), int(h)
    from PIL import Image
    with Image.open(path) as im:
        return int(im.width), int(im.height)


def _as_float_array(value: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if shape is not None and arr.shape != shape:
        raise ValueError(f'Expected array shape {shape}, got {arr.shape}')
    if not np.isfinite(arr).all():
        raise ValueError('Array contains NaN/Inf')
    return arr


def q_xyzw_to_matrix(q: Any) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n < 1e-12:
        raise ValueError('Zero quaternion')
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float64)


def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _rotation_candidates(rotation: Any) -> list[tuple[str, float]]:
    """Return plausible yaw candidates from common UE5 rotation representations.

    For a three-value UE5 Rotator/Euler array, the yaw component is not assumed to
    occupy a fixed index. The caller selects the component that agrees with the
    cuboid's measured long-axis direction. Both radians and degrees are tested.
    """
    out: list[tuple[str, float]] = []
    if isinstance(rotation, dict):
        for key in ('yaw', 'Yaw', 'z', 'Z', 'heading'):
            if key in rotation:
                v = float(rotation[key])
                out.append((f'{key}:rad', v))
                out.append((f'{key}:deg', math.radians(v)))
    elif isinstance(rotation, (list, tuple, np.ndarray)):
        vals = list(rotation)
        for i, raw in enumerate(vals):
            v = float(raw)
            out.append((f'rotation[{i}]:rad', v))
            out.append((f'rotation[{i}]:deg', math.radians(v)))
    elif isinstance(rotation, (int, float)):
        v = float(rotation)
        out.extend([('rotation:rad', v), ('rotation:deg', math.radians(v))])
    return out


def _object_field(obj: dict[str, Any], key: str, default=None):
    if key in obj:
        return obj[key]
    geometry = obj.get('geometry')
    if isinstance(geometry, dict) and key in geometry:
        return geometry[key]
    return default


def iter_ue5_objects(data: dict[str, Any]):
    """Normalize list, dict, and single-object variants of the UE5 JSON schema."""
    objects = data.get('objects', [])
    if isinstance(objects, list):
        if objects:
            return [x for x in objects if isinstance(x, dict)]
        if 'corners_world' in data or (isinstance(data.get('geometry'), dict) and 'corners_world' in data['geometry']):
            return [data]
        return []
    if isinstance(objects, dict):
        if 'corners_world' in objects or (isinstance(objects.get('geometry'), dict) and 'corners_world' in objects['geometry']):
            return [objects]
        return [x for x in objects.values() if isinstance(x, dict)]
    # Some exports store a single object record directly at the JSON root.
    if 'corners_world' in data or (isinstance(data.get('geometry'), dict) and 'corners_world' in data['geometry']):
        return [data]
    return []


def ue5_local_coordinates(obj: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return (local corner coordinates, world corners) for a UE5 object.

    Preferred route: quaternion local-to-world transform.
    Fallback: infer the yaw component from the actual cuboid edge direction and
    the supplied extent/rotation fields. This supports the user's JSON form such
    as rotation=[0.168, -132.482, 1.324] without requiring quat_xyzw.
    """
    corners = _as_float_array(_object_field(obj, 'corners_world'), (8, 3))
    center = _as_float_array(_object_field(obj, 'center'), (3,))

    quat = None
    for key in ('quat_xyzw', 'quaternion_xyzw', 'rotation_quaternion'):
        if _object_field(obj, key) is not None:
            quat = _object_field(obj, key)
            break
    if quat is not None:
        R = q_xyzw_to_matrix(quat)
        local = (corners - center.reshape(1, 3)) @ R
        return local, corners

    # Euler/Rotator fallback: vehicle boxes are assumed to be upright in UE5 world Z-up.
    extent_value = _object_field(obj, 'extent')
    if extent_value is None:
        raise ValueError('UE5 object has neither quaternion nor extent for Euler-yaw inference')
    extent = _as_float_array(extent_value, (3,))
    if min(abs(float(extent[0])), abs(float(extent[1]))) < 1e-6:
        raise ValueError('Invalid UE5 horizontal extent')

    bottom = corners[:4]
    edge_vecs = [bottom[(i+1) % 4] - bottom[i] for i in range(4)]
    edge_lengths = [float(np.linalg.norm(v[:2])) for v in edge_vecs]
    long_edge_i = int(np.argmax(edge_lengths))
    if edge_lengths[long_edge_i] < 1e-6:
        raise ValueError('Cannot infer UE5 horizontal edge direction')
    edge_angle = math.atan2(float(edge_vecs[long_edge_i][1]), float(edge_vecs[long_edge_i][0]))

    # UE5 extent is half-size along local axes. Normally the larger horizontal
    # extent identifies the long axis. For nearly square footprints (e.g. people),
    # test both hypotheses and let the rotation metadata disambiguate them.
    ex, ey = abs(float(extent[0])), abs(float(extent[1]))
    preferred_x = ex >= ey
    axis_hypotheses = [preferred_x] if abs(ex - ey) / max(ex, ey, 1e-9) >= 0.08 else [preferred_x, not preferred_x]
    candidates = _rotation_candidates(_object_field(obj, 'rotation'))
    if not candidates:
        raise ValueError('UE5 Euler rotation field missing/unrecognized')

    scored: list[tuple[float, str, float]] = []
    for long_axis_is_x in axis_hypotheses:
        target_yaw = edge_angle if long_axis_is_x else edge_angle - math.pi / 2.0
        for name, angle in candidates:
            # Compare modulo pi because an edge is an undirected axis; retain the
            # signed yaw value from metadata for the actual local-coordinate transform.
            diff = abs(_wrap_angle(angle - target_yaw))
            diff_pi = abs(_wrap_angle(angle - (target_yaw + math.pi)))
            scored.append((min(diff, diff_pi), name, angle))
    scored.sort(key=lambda x: x[0])
    best_diff, best_name, yaw = scored[0]
    if best_diff > math.radians(20.0):
        raise ValueError(
            f'Cannot reconcile UE5 rotation with cuboid edge: best={math.degrees(best_diff):.2f} deg ({best_name})'
        )

    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    local = (corners - center.reshape(1, 3)) @ Rz
    return local, corners


def corner_sign_patterns(local: np.ndarray, eps: float = 1e-4) -> list[tuple[int, int, int]]:
    if local.shape != (8, 3):
        raise ValueError(f'Expected (8,3) local corners, got {local.shape}')
    out = []
    for row in local:
        if np.any(np.abs(row) < eps):
            raise ValueError('Corner too close to a local cuboid axis')
        out.append(tuple(1 if float(v) > 0 else -1 for v in row))
    return out


def inspect_ue5_corner_order(root: Path) -> tuple[list[tuple[int, int, int]], dict[str, Any]]:
    files = sorted((root / 'labels').glob('*.json'))
    if not files:
        raise FileNotFoundError(f'No UE5 JSON files in {root / "labels"}')
    per_index = [Counter() for _ in range(8)]
    used = 0
    seen = 0
    bottom_votes = 0
    top_votes = 0
    errors = Counter()
    for jf in files:
        try:
            data = json.loads(jf.read_text(encoding='utf-8'))
        except Exception:
            errors['json_read_error'] += 1
            continue
        for obj in iter_ue5_objects(data):
            seen += 1
            try:
                local, world = ue5_local_coordinates(obj)
                patterns = corner_sign_patterns(local)
                if len(set(patterns)) != 8 or set(patterns) != CANONICAL_PATTERN_SET:
                    raise ValueError('The eight points are not the eight unique cuboid corners')
                for i, pat in enumerate(patterns):
                    per_index[i][pat] += 1
                if np.mean(local[:4, 2]) < np.mean(local[4:, 2]):
                    bottom_votes += 1
                else:
                    top_votes += 1
                used += 1
            except Exception as exc:
                errors[f'invalid_corner_order_object:{type(exc).__name__}'] += 1
    if used == 0:
        raise RuntimeError('Cannot infer UE5 keypoint order; no valid objects with auditable 3D corners.')

    raw_patterns: list[tuple[int, int, int]] = []
    agreement: dict[str, float] = {}
    for i, counts in enumerate(per_index):
        if not counts:
            raise RuntimeError(f'No corner-order evidence for UE5 keypoint {i}')
        pat, n = counts.most_common(1)[0]
        ratio = n / sum(counts.values())
        agreement[str(i)] = round(ratio, 6)
        if ratio < 0.95:
            raise RuntimeError(
                f'UE5 raw corner index {i} is inconsistent ({ratio:.1%}); refusing unsafe remapping.'
            )
        raw_patterns.append(pat)
    if len(set(raw_patterns)) != 8 or set(raw_patterns) != CANONICAL_PATTERN_SET:
        raise RuntimeError('UE5 raw corner order does not contain all eight unique cuboid corners.')

    canonical_order = [raw_patterns.index(p) for p in CANONICAL_PATTERNS]
    report = {
        'json_files': len(files),
        'objects_seen': seen,
        'objects_audited': used,
        'audit_coverage': round(used / max(seen, 1), 6),
        'raw_index_pattern_agreement': agreement,
        'raw_index_local_sign_patterns': [list(x) for x in raw_patterns],
        'canonical_dair_order_to_raw_indices': canonical_order,
        'bottom_first_raw_objects': bottom_votes,
        'top_first_raw_objects': top_votes,
        'rotation_audit': 'quaternion when available; otherwise infer UE5 yaw from extent + cuboid edge + rotation metadata',
        'errors': dict(errors),
    }
    return raw_patterns, report


def resolve_ue5_image(root: Path, data: dict[str, Any], jf: Path) -> Path | None:
    rel = data.get('image')
    if isinstance(rel, dict):
        rel = rel.get('path') or rel.get('file') or rel.get('image_path')
    if rel:
        p = Path(str(rel).replace('\\', '/'))
        candidates = [p] if p.is_absolute() else [root / p, root / 'rgb' / p]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    stem = str(data.get('frame', jf.stem))
    for folder in (root / 'rgb', root / 'images', root):
        for ext in ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'):
            p = folder / f'{stem}{ext}'
            if p.is_file():
                return p
    return None


def inside_image(points: np.ndarray, width: int, height: int) -> bool:
    return bool(
        np.isfinite(points).all()
        and np.all(points[:, 0] >= 0) and np.all(points[:, 0] <= width - 1)
        and np.all(points[:, 1] >= 0) and np.all(points[:, 1] <= height - 1)
    )


def projected_bbox(points: np.ndarray, width: int, height: int) -> tuple[float, float, float, float] | None:
    if not np.isfinite(points).all():
        return None
    x1, y1 = float(points[:, 0].min()), float(points[:, 1].min())
    x2, y2 = float(points[:, 0].max()), float(points[:, 1].max())
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(width - 1.0, x2), min(height - 1.0, y2)
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    return x1, y1, x2, y2


def parse_bbox(value: Any) -> tuple[float, float, float, float] | None:
    try:
        if isinstance(value, dict):
            vals = [value[k] for k in ('xmin', 'ymin', 'xmax', 'ymax')]
        else:
            vals = list(value)
        if len(vals) != 4:
            return None
        box = tuple(float(v) for v in vals)
        if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
            return None
        return box  # type: ignore[return-value]
    except Exception:
        return None


def clip_bbox(box: tuple[float, float, float, float] | None, width: int, height: int):
    if box is None:
        return None
    x1, y1, x2, y2 = box
    out = (max(0.0, x1), max(0.0, y1), min(width - 1.0, x2), min(height - 1.0, y2))
    return out if out[2] - out[0] >= 1.0 and out[3] - out[1] >= 1.0 else None


def make_pose_line(cls: int, points: np.ndarray, width: int, height: int,
                   bbox_override: tuple[float, float, float, float] | None = None) -> str:
    box = clip_bbox(bbox_override, width, height) if bbox_override is not None else None
    if box is None:
        box = projected_bbox(points, width, height)
    if box is None:
        raise ValueError('Degenerate 2D bounding box')
    x1, y1, x2, y2 = box
    vals = [cls, (x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
            (x2 - x1) / width, (y2 - y1) / height]
    for x, y in points:
        vals.extend([float(x) / width, float(y) / height])
    return ' '.join([str(int(vals[0]))] + [f'{float(v):.6f}' for v in vals[1:]])


def make_record(source: str, image_path: Path, instances: list[dict[str, Any]],
                group: str = '', group_alt: str = '', group_mode: str = '') -> dict[str, Any]:
    cc = Counter(int(x['cls']) for x in instances)
    return {
        'source': source, 'image_path': image_path, 'stem': image_path.stem,
        'instances': instances, 'class_counts': cc,
        'group': group or f'frame:{image_path.stem}',
        'group_alt': group_alt,
        'group_mode': group_mode or ('image' if not group else 'metadata'),
    }


def _safe_fixed_array(value: Any, shape: tuple[int, ...], stats: Counter, key: str) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        stats[f'{key}_ragged'] += 1
        return None
    if arr.shape != shape:
        stats[f'{key}_bad_shape'] += 1
        return None
    if not np.isfinite(arr).all():
        stats[f'{key}_nonfinite'] += 1
        return None
    return arr


# -----------------------------------------------------------------------------
# UE5 / UrbanOmniView conversion
# -----------------------------------------------------------------------------
def _ue5_group(data: dict[str, Any], image_path: Path) -> tuple[str, str, str]:
    # Prefer sequence/scene identifiers; do not stringify an entire nested scene object.
    for key in ('scene_id', 'sequence_id', 'sequence', 'environment_id', 'level'):
        value = data.get(key)
        if value not in (None, '') and not isinstance(value, (dict, list)):
            return f'{key}:{value}', '', key
    scene = data.get('scene')
    if isinstance(scene, dict):
        for key in ('scene_id', 'sequence_id', 'id', 'name', 'map'):
            value = scene.get(key)
            if value not in (None, ''):
                return f'scene:{value}', '', 'scene'
    elif scene not in (None, ''):
        return f'scene:{scene}', '', 'scene'
    map_value = data.get('map')
    if map_value not in (None, '') and not isinstance(map_value, (dict, list)):
        return f'map:{map_value}', '', 'map'
    return f'frame:{image_path.stem}', '', 'image'


def ue5_instances(data: dict[str, Any], width: int, height: int, stats: Counter,
                  raw_patterns: list[tuple[int, int, int]]) -> list[dict[str, Any]]:
    out = []
    canonical_order = [raw_patterns.index(p) for p in CANONICAL_PATTERNS]
    for obj in iter_ue5_objects(data):
        source_class = str(_object_field(obj, 'label', _object_field(obj, 'type', _object_field(obj, 'class_name', '')))).strip()
        label = source_class.lower()
        if label not in UE5_CLASS_MAP:
            stats[f'ue5_ignored_class:{source_class or "<empty>"}'] += 1
            continue
        cls = UE5_CLASS_MAP[label]
        try:
            local, world_corners = ue5_local_coordinates(obj)
            patterns = corner_sign_patterns(local)
            if patterns != raw_patterns:
                stats['ue5_object_corner_order_mismatch'] += 1
                continue
            pts = None
            corners_image = _object_field(obj, 'corners_image')
            if corners_image is not None:
                raw = _safe_fixed_array(corners_image, (8, 2), stats, 'ue5_corners_image')
                if raw is not None:
                    pts = raw[canonical_order]
            corners_camera = _object_field(obj, 'corners_camera')
            camera_points = None
            if corners_camera is not None:
                camera_points_raw = _safe_fixed_array(corners_camera, (8, 3), stats, 'ue5_corners_camera')
                if camera_points_raw is not None:
                    camera_points = camera_points_raw[canonical_order]
            if pts is None and camera_points is not None:
                camera = data.get('camera', {})
                if 'K' in camera:
                    K = _safe_fixed_array(camera['K'], (3, 3), stats, 'ue5_camera_K')
                    if K is not None:
                        hom = (K @ camera_points.T).T
                        if np.all(hom[:, 2] > 1e-8):
                            pts = hom[:, :2] / hom[:, 2:3]
                        else:
                            stats['ue5_camera_points_nonpositive_depth'] += 1
            if pts is None:
                stats['ue5_invalid_keypoints'] += 1
                continue
            if not inside_image(pts, width, height):
                stats['ue5_corner_outside_image'] += 1
                continue
            if projected_bbox(pts, width, height) is None:
                stats['ue5_degenerate_box'] += 1
                continue
            bbox = parse_bbox(_object_field(obj, 'bbox_2d'))
            geometry3d = {
                'coordinate_frame': 'UE5_world',
                'units_metadata': data.get('units', 'unspecified_source_units'),
                'corners_3d_source': world_corners[canonical_order].tolist(),
            }
            if camera_points is not None:
                geometry3d['corners_3d_camera'] = camera_points.tolist()
                geometry3d['camera_coordinate_units_metadata'] = data.get('units', 'unspecified_source_units')
            stats[f'ue5_kept_source_class:{source_class}'] += 1
            out.append({
                'cls': cls, 'points': pts, 'bbox': bbox,
                'source_class': source_class,
                'geometry3d': geometry3d,
            })
        except Exception as exc:
            stats[f'ue5_invalid_object:{type(exc).__name__}'] += 1
    return out


def collect_ue5(root: Path, stats: Counter, raw_patterns: list[tuple[int, int, int]]) -> list[dict[str, Any]]:
    label_dir = root / 'labels'
    if not label_dir.is_dir():
        raise FileNotFoundError(f'UE5 labels directory not found: {label_dir}')
    records = []
    for jf in sorted(label_dir.glob('*.json')):
        try:
            data = json.loads(jf.read_text(encoding='utf-8'))
            image_path = resolve_ue5_image(root, data, jf)
            if image_path is None:
                stats['ue5_missing_image'] += 1
                continue
            w, h = read_image_size(image_path)
            instances = ue5_instances(data, w, h, stats, raw_patterns)
            group, group_alt, group_mode = _ue5_group(data, image_path)
            rec = make_record('uov', image_path, instances, group, group_alt, group_mode)

            # Preserve acquisition metadata for leakage-resistant chronological splitting.
            frame_value = data.get('frame', jf.stem)
            timestamp_value = data.get('timestamp_utc')
            engine_frame_value = data.get('engine_frame')
            if timestamp_value is None or engine_frame_value is None:
                raise ValueError('missing timestamp_utc or engine_frame metadata')
            timestamp_text = str(timestamp_value)
            timestamp_dt = datetime.fromisoformat(timestamp_text.replace('Z', '+00:00'))
            if timestamp_dt.tzinfo is None:
                raise ValueError('timestamp_utc must include an explicit timezone')
            rec.update({
                'frame_index': int(frame_value),
                'engine_frame': int(engine_frame_value),
                'timestamp_utc': timestamp_text,
                'timestamp_epoch': timestamp_dt.timestamp(),
            })
            records.append(rec)
        except Exception as exc:
            stats[f'ue5_sample_error:{type(exc).__name__}'] += 1
            print(f'[WARN] UE5 {jf.name}: {exc}')
    return records


# -----------------------------------------------------------------------------
# KITTI conversion
# -----------------------------------------------------------------------------
def read_p2(path: Path) -> np.ndarray:
    for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        if line.startswith('P2:'):
            vals = np.fromstring(line.split(':', 1)[1], sep=' ', dtype=np.float64)
            if vals.size == 12:
                return vals.reshape(3, 4)
    raise ValueError(f'P2 not found in {path}')


def kitti_corners(h: float, w: float, length: float, loc: np.ndarray,
                  yaw: float, width_sign: int = 1) -> np.ndarray:
    """Generate KITTI camera-frame corners in DAIR semantic index order.

    KITTI location is the bottom centre. Camera y points down, so bottom y=0,
    top y=-h. The local long-axis signs and width-side signs follow CANONICAL_PATTERNS.
    """
    sx = np.asarray([p[0] for p in CANONICAL_PATTERNS], dtype=np.float64)
    sy = np.asarray([p[1] for p in CANONICAL_PATTERNS], dtype=np.float64)
    sz = np.asarray([p[2] for p in CANONICAL_PATTERNS], dtype=np.float64)
    x_local = sx * length / 2.0
    z_local = width_sign * sy * w / 2.0
    y_local = np.where(sz < 0, 0.0, -h)
    c, s = math.cos(yaw), math.sin(yaw)
    rot_y = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    return (rot_y @ np.vstack([x_local, y_local, z_local])).T + loc.reshape(1, 3)


def project_camera(points: np.ndarray, P: np.ndarray) -> np.ndarray | None:
    hom = (P @ np.hstack([points, np.ones((8, 1))]).T).T
    if not np.isfinite(hom).all() or np.any(hom[:, 2] <= 1e-6):
        return None
    uv = hom[:, :2] / hom[:, 2:3]
    return uv if np.isfinite(uv).all() else None


def kitti_instances(label_path: Path, calib_path: Path, width: int, height: int,
                    stats: Counter, width_sign: int = 1) -> list[dict[str, Any]]:
    P2 = read_p2(calib_path)
    out = []
    for line in label_path.read_text(encoding='utf-8', errors='ignore').splitlines():
        f = line.split()
        if not f:
            continue
        if len(f) < 15:
            stats['kitti_malformed_line'] += 1
            continue
        typ = f[0]
        if typ not in KITTI_CLASS_MAP:
            stats[f'kitti_ignored_class:{typ}'] += 1
            continue
        try:
            h, w, length = map(float, f[8:11])
            loc = np.asarray(list(map(float, f[11:14])), dtype=np.float64)
            yaw = float(f[14])
            if min(h, w, length) <= 0 or loc.shape != (3,) or not np.isfinite(loc).all():
                raise ValueError('invalid 3D geometry')
            pts3d = kitti_corners(h, w, length, loc, yaw, width_sign=width_sign)
            pts2d = project_camera(pts3d, P2)
            if pts2d is None:
                stats['kitti_projection_invalid'] += 1
                continue
            if not inside_image(pts2d, width, height):
                stats['kitti_corner_outside_image'] += 1
                continue
            if projected_bbox(pts2d, width, height) is None:
                stats['kitti_degenerate_box'] += 1
                continue
            bbox = parse_bbox(f[4:8])
            stats[f'kitti_kept_source_class:{typ}'] += 1
            out.append({
                'cls': KITTI_CLASS_MAP[typ], 'points': pts2d, 'bbox': bbox,
                'source_class': typ,
                'geometry3d': {
                    'coordinate_frame': 'rectified_camera', 'units': 'm',
                    'corners_3d_camera': pts3d.tolist(),
                },
            })
        except Exception:
            stats['kitti_invalid_geometry_or_projection'] += 1
    return out


def find_image(image_dir: Path, stem: str) -> Path | None:
    for ext in ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'):
        p = image_dir / f'{stem}{ext}'
        if p.is_file():
            return p
    return None


def collect_kitti(root: Path, stats: Counter, width_sign: int = 1) -> list[dict[str, Any]]:
    base = root / 'training'
    image_dir, label_dir, calib_dir = base / 'image_2', base / 'label_2', base / 'calib'
    if not image_dir.is_dir() or not label_dir.is_dir() or not calib_dir.is_dir():
        raise FileNotFoundError(f'KITTI training image_2/label_2/calib not found under {base}')
    records = []
    for label_path in sorted(label_dir.glob('*.txt')):
        stem = label_path.stem
        image_path = find_image(image_dir, stem)
        calib_path = calib_dir / f'{stem}.txt'
        if image_path is None or not calib_path.is_file():
            stats['kitti_incomplete_triplet'] += 1
            continue
        try:
            w, h = read_image_size(image_path)
            instances = kitti_instances(label_path, calib_path, w, h, stats, width_sign)
            records.append(make_record('kitti', image_path, instances, group=f'frame:{stem}', group_mode='image'))
        except Exception as exc:
            stats[f'kitti_sample_error:{type(exc).__name__}'] += 1
            print(f'[WARN] KITTI {stem}: {exc}')
    return records


# -----------------------------------------------------------------------------
# DAIR-V2X-I conversion
# -----------------------------------------------------------------------------
def _first_existing_file(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.is_file():
            return p
    return None


def resolve_dair_meta_file(meta_root: Path, dair_root: Path, rel: str) -> Path | None:
    p = Path(str(rel).replace('\\', '/'))
    if p.is_absolute():
        return p if p.is_file() else None
    roots = [meta_root, dair_root, dair_root / 'single-infrastructure-side', dair_root / 'infrastructure-side']
    return _first_existing_file([base / p for base in roots])


def resolve_dair_image(meta_root: Path, image_roots: list[Path], dair_root: Path, rel: str) -> Path | None:
    p = Path(str(rel).replace('\\', '/'))
    if p.is_absolute():
        return p if p.is_file() else None
    candidates = [meta_root / p, dair_root / p]
    candidates.extend(base / p for base in image_roots)
    candidates.extend(base / p.name for base in image_roots)
    return _first_existing_file(candidates)


def find_dair_layout(root: Path) -> tuple[Path, list[Path], Path]:
    possible_meta = [root, root / 'single-infrastructure-side', root / 'infrastructure-side']
    meta_root = next((p for p in possible_meta if (p / 'data_info.json').is_file()), None)
    if meta_root is None:
        raise FileNotFoundError(
            f'Cannot find DAIR data_info.json under {root}, {root / "single-infrastructure-side"}, '
            f'or {root / "infrastructure-side"}'
        )
    image_roots = [
        meta_root / 'image', meta_root / 'images',
        root / 'single-infrastructure-side-image', root / 'infrastructure-side-image',
        root / 'image', root / 'images',
    ]
    return meta_root, image_roots, meta_root / 'data_info.json'


def dair_corners_virtual_lidar(obj: dict[str, Any]) -> np.ndarray:
    """Return DAIR official 8 corners in Virtual-LiDAR coordinates (metres)."""
    d = obj.get('3d_dimensions', {})
    p = obj.get('3d_location', {})
    h, w, length = float(d['h']), float(d['w']), float(d['l'])
    x, y, z = float(p['x']), float(p['y']), float(p['z'])
    yaw = float(obj['rotation'])
    if min(h, w, length) <= 0 or not np.isfinite([h, w, length, x, y, z, yaw]).all():
        raise ValueError('Invalid DAIR cuboid dimensions/location/rotation')
    local = np.array([
        [ length/2,  w/2, -h/2],
        [ length/2, -w/2, -h/2],
        [-length/2, -w/2, -h/2],
        [-length/2,  w/2, -h/2],
        [ length/2,  w/2,  h/2],
        [ length/2, -w/2,  h/2],
        [-length/2, -w/2,  h/2],
        [-length/2,  w/2,  h/2],
    ], dtype=np.float64)
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return (Rz @ local.T).T + np.array([x, y, z], dtype=np.float64)


def project_dair_object(obj: dict[str, Any], K: np.ndarray, R: np.ndarray,
                        t: np.ndarray) -> np.ndarray | None:
    """Project DAIR official bottom-first cuboid corners into the camera image."""
    try:
        corners_virtual_lidar = dair_corners_virtual_lidar(obj)
    except Exception:
        return None
    corners_camera = (R @ corners_virtual_lidar.T).T + t.reshape(1, 3)
    hom = (K @ corners_camera.T).T
    if not np.isfinite(hom).all() or np.any(hom[:, 2] <= 1e-6):
        return None
    uv = hom[:, :2] / hom[:, 2:3]
    return uv if np.isfinite(uv).all() else None


def _dair_group_fields(row: dict[str, Any], image_path: Path) -> tuple[str, str, str]:
    intersection = row.get('intersection_loc')
    batch = next((row.get(k) for k in ('batch_id', 'sequence_id', 'sequence', 'scene_id')
                  if row.get(k) not in (None, '')), None)
    if intersection not in (None, '') and str(intersection).strip():
        return f'intersection:{str(intersection).strip()}', (f'batch:{batch}' if batch not in (None, '') else ''), 'intersection_loc'
    if batch not in (None, ''):
        return f'batch:{batch}', '', 'batch_or_sequence'
    # Explicitly marked image-level fallback; never misreport this as leakage-safe grouping.
    return f'frame:{image_path.stem}', '', 'image_level_no_group_metadata'


def _dair_label_path(row: dict[str, Any]) -> str | None:
    # Prefer camera-time labels; DAIR's camera labels still store 3D boxes in Virtual-LiDAR coordinates.
    keys = (
        'label_camera_std_path', 'label_camera_path',
        'label_virtuallidar_path', 'label_lidar_std_path', 'label_lidar_path',
    )
    for key in keys:
        if row.get(key):
            return str(row[key])
    return None


def collect_dair(dair_root: Path, stats: Counter, min_iou: float = 0.0) -> list[dict[str, Any]]:
    meta_root, image_roots, data_info = find_dair_layout(dair_root)
    rows = json.loads(data_info.read_text(encoding='utf-8'))
    if not isinstance(rows, list):
        raise ValueError('DAIR data_info.json must contain a list')
    records = []
    for row in rows:
        try:
            image_rel = row.get('image_path')
            label_rel = _dair_label_path(row)
            intr_rel = row.get('calib_camera_intrinsic_path') or row.get('calib_camera_intrinsic')
            ext_rel = row.get('calib_virtuallidar_to_camera_path') or row.get('calib_virtuallidar_to_camera')
            if not all([image_rel, label_rel, intr_rel, ext_rel]):
                stats['dair_incomplete_record'] += 1
                continue
            image_path = resolve_dair_image(meta_root, image_roots, dair_root, str(image_rel))
            label_path = resolve_dair_meta_file(meta_root, dair_root, str(label_rel))
            intr_path = resolve_dair_meta_file(meta_root, dair_root, str(intr_rel))
            ext_path = resolve_dair_meta_file(meta_root, dair_root, str(ext_rel))
            if not all([image_path, label_path, intr_path, ext_path]):
                stats['dair_incomplete_record'] += 1
                continue

            w_img, h_img = read_image_size(image_path)
            intr = json.loads(intr_path.read_text(encoding='utf-8'))
            ext = json.loads(ext_path.read_text(encoding='utf-8'))
            K = np.asarray(intr.get('cam_K', intr.get('camera_matrix')), dtype=np.float64).reshape(3, 3)
            R = np.asarray(ext['rotation'], dtype=np.float64).reshape(3, 3)
            t = np.asarray(ext['translation'], dtype=np.float64).reshape(3)
            objects = json.loads(label_path.read_text(encoding='utf-8'))
            if isinstance(objects, dict) and 'objects' in objects:
                objects = objects['objects']
            instances = []
            for obj in objects:
                typ = str(obj.get('type', '')).strip()
                mapped_type = next((name for name in DAIR_CLASS_MAP if name.lower() == typ.lower()), None)
                if mapped_type is None:
                    stats[f'dair_ignored_class:{typ or "<empty>"}'] += 1
                    continue
                pts = None
                lidar_corners = None
                camera_corners = None
                try:
                    lidar_corners = dair_corners_virtual_lidar(obj)
                    camera_corners = (R @ lidar_corners.T).T + t.reshape(1, 3)
                    hom = (K @ camera_corners.T).T
                    if np.isfinite(hom).all() and np.all(hom[:, 2] > 1e-6):
                        pts = hom[:, :2] / hom[:, 2:3]
                except Exception:
                    pts = None
                if pts is None:
                    stats['dair_invalid_projection_or_geometry'] += 1
                    continue
                if not inside_image(pts, w_img, h_img):
                    stats['dair_corner_outside_image'] += 1
                    continue
                box = projected_bbox(pts, w_img, h_img)
                if box is None:
                    stats['dair_degenerate_box'] += 1
                    continue

                official_box = parse_bbox(obj.get('2d_box'))
                if official_box is not None:
                    clipped_official = clip_bbox(official_box, w_img, h_img)
                    if clipped_official is not None:
                        gx1, gy1, gx2, gy2 = clipped_official
                        ix1, iy1 = max(box[0], gx1), max(box[1], gy1)
                        ix2, iy2 = min(box[2], gx2), min(box[3], gy2)
                        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                        area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
                        area_b = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
                        iou = inter / max(area_a + area_b - inter, 1e-9)
                        cls_name = CLASS_NAMES[DAIR_CLASS_MAP[mapped_type]]
                        stats['dair_projection_iou_sum'] += iou
                        stats['dair_projection_iou_n'] += 1
                        stats[f'dair_projection_iou_sum:{cls_name}'] += iou
                        stats[f'dair_projection_iou_n:{cls_name}'] += 1
                        stats[f'dair_projection_iou_ge50:{cls_name}'] += int(iou >= 0.50)
                        stats[f'dair_projection_iou_ge70:{cls_name}'] += int(iou >= 0.70)
                        if iou < min_iou:
                            stats['dair_low_projection_iou_filtered'] += 1
                            continue
                    else:
                        official_box = None
                if official_box is None:
                    stats['dair_official_2d_bbox_unavailable'] += 1
                stats[f'dair_kept_source_class:{typ}'] += 1
                instances.append({
                    'cls': DAIR_CLASS_MAP[mapped_type], 'points': pts, 'bbox': official_box,
                    'source_class': typ,
                    'geometry3d': {
                        'coordinate_frame': 'virtual_lidar',
                        'units': 'm',
                        'corners_3d_virtual_lidar': lidar_corners.tolist(),
                        'corners_3d_camera': camera_corners.tolist(),
                    },
                })

            group, group_alt, group_mode = _dair_group_fields(row, image_path)
            if group_mode == 'image_level_no_group_metadata':
                stats['dair_missing_group_metadata'] += 1
            records.append(make_record('dair', image_path, instances, group, group_alt, group_mode))
        except Exception as exc:
            stats[f'dair_sample_error:{type(exc).__name__}'] += 1
            print(f'[WARN] DAIR record {row.get("image_path", "<unknown>")}: {exc}')
    return records


# -----------------------------------------------------------------------------
# Deterministic multi-label splitting
# -----------------------------------------------------------------------------
def allocate_counts(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    raw = [n * r for r in ratios]
    counts = [int(math.floor(x)) for x in raw]
    left = n - sum(counts)
    order = sorted(range(3), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in order[:left]:
        counts[i] += 1
    return tuple(counts)  # type: ignore[return-value]


def _record_vector(record: dict[str, Any]) -> np.ndarray:
    cc = record['class_counts']
    return np.asarray([1.0, cc.get(0, 0), cc.get(1, 0), cc.get(2, 0)], dtype=np.float64)


def _rarity_score(record: dict[str, Any], totals: np.ndarray) -> float:
    vec = _record_vector(record)
    score = 0.0
    for c in range(3):
        if vec[c + 1] > 0:
            score += 1.0 / max(totals[c + 1], 1.0)
    return score + 1e-12 * float(vec[1:].sum())


def stratified_image_split(records: list[dict[str, Any]], ratios=(0.8, 0.1, 0.1), seed=42):
    """Greedy multi-label split balancing image counts and per-class instance counts."""
    if not records:
        return {'train': [], 'val': [], 'test': []}
    rng = random.Random(seed)
    names = ['train', 'val', 'test']
    ntr, nva, nte = allocate_counts(len(records), tuple(ratios))
    slots = [ntr, nva, nte]
    total = np.sum([_record_vector(r) for r in records], axis=0)
    targets = np.asarray(ratios, dtype=np.float64)[:, None] * total[None, :]
    current = np.zeros((3, 4), dtype=np.float64)
    out = {name: [] for name in names}

    rows = records[:]
    rng.shuffle(rows)
    rows.sort(key=lambda r: _rarity_score(r, total), reverse=True)
    for rec in rows:
        vec = _record_vector(rec)
        choices = [i for i in range(3) if slots[i] > 0]
        scored = []
        for s in choices:
            image_deficit = (targets[s, 0] - current[s, 0]) / max(targets[s, 0], 1.0)
            cls_deficits = []
            for c in range(3):
                if vec[c + 1] > 0:
                    cls_deficits.append((targets[s, c + 1] - current[s, c + 1]) / max(targets[s, c + 1], 1.0))
            class_deficit = float(np.mean(cls_deficits)) if cls_deficits else image_deficit
            score = 0.60 * image_deficit + 0.40 * class_deficit + rng.random() * 1e-8
            scored.append((score, s))
        chosen = max(scored)[1]
        out[names[chosen]].append(rec)
        current[chosen] += vec
        slots[chosen] -= 1
    for split in out:
        out[split].sort(key=lambda r: r['image_path'].name)
    return out


def make_group_vectors(groups: list[tuple[str, list[dict[str, Any]]]]) -> np.ndarray:
    vectors = np.zeros((len(groups), 4), dtype=np.float64)
    for i, (_, rows) in enumerate(groups):
        for r in rows:
            vectors[i] += _record_vector(r)
    return vectors


def group_split_objective(assign: tuple[int, ...], vectors: np.ndarray,
                          ratios=(0.8, 0.1, 0.1)) -> float:
    total = vectors.sum(axis=0)
    ratio_arr = np.asarray(ratios, dtype=np.float64)
    target = ratio_arr[:, None] * total[None, :]
    got = np.zeros((3, 4), dtype=np.float64)
    for i, split_idx in enumerate(assign):
        got[split_idx] += vectors[i]
    active = np.flatnonzero(ratio_arr > 0)
    if any(got[s, 0] == 0 for s in active):
        return 1e9
    # A zero-ratio split (e.g. UE5 train/val with a previously locked test set)
    # must remain empty and must not make the assignment impossible.
    if any(got[s, 0] > 0 for s in range(3) if s not in active):
        return 1e9
    image_err = np.sum(((got[active, 0] - target[active, 0]) / np.maximum(target[active, 0], 1.0)) ** 2)
    class_err = np.sum(((got[active, 1:] - target[active, 1:]) / np.maximum(target[active, 1:], 1.0)) ** 2)
    return float(image_err + 0.25 * class_err)


def split_by_group(records: list[dict[str, Any]], group_field: str = 'group',
                   ratios=(0.8, 0.1, 0.1), seed=42):
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        grouped[str(r.get(group_field) or f'frame:{r["stem"]}')].append(r)
    ratio_arr = np.asarray(ratios, dtype=np.float64)
    active = [int(x) for x in np.flatnonzero(ratio_arr > 0)]
    if len(grouped) < len(active):
        return None, {}
    items = list(grouped.items())
    rng = random.Random(seed)
    rng.shuffle(items)
    items.sort(key=lambda x: len(x[1]), reverse=True)
    vectors = make_group_vectors(items)
    n = len(items)
    best_assign = None
    best_score = float('inf')
    if n <= 12:
        for ass in itertools.product(active, repeat=n):
            if set(ass) != set(active):
                continue
            score = group_split_objective(ass, vectors, ratios)
            if score < best_score:
                best_score, best_assign = score, ass
    else:
        # Greedy assignment using the full-dataset targets, followed by local improvement.
        total = vectors.sum(axis=0)
        targets = ratio_arr[:, None] * total[None, :]
        current = np.zeros((3, 4), dtype=np.float64)
        ass: list[int] = []
        for i, vec in enumerate(vectors):
            empty_splits = [s for s in active if not np.any(current[s])]
            remaining_groups = n - i
            # Do not force the largest groups into validation/test. Fill the best-fit
            # split first, reserving only the final required groups for still-empty splits.
            choices = empty_splits if remaining_groups <= len(empty_splits) else active
            scored = []
            for split_idx in choices:
                after = current[split_idx] + vec
                denom = np.maximum(targets[split_idx], 1.0)
                delta = np.sum(((after - targets[split_idx]) / denom) ** 2
                               - ((current[split_idx] - targets[split_idx]) / denom) ** 2)
                scored.append((float(delta), split_idx))
            chosen = min(scored)[1]
            ass.append(chosen)
            current[chosen] += vec

        changed = True
        while changed:
            changed = False
            current_score = group_split_objective(tuple(ass), vectors, ratios)
            counts = Counter(ass)
            for i in range(len(ass)):
                old = ass[i]
                if counts[old] <= 1:
                    continue
                for new_split in active:
                    if new_split == old:
                        continue
                    trial = ass[:]
                    trial[i] = new_split
                    score = group_split_objective(tuple(trial), vectors, ratios)
                    if score + 1e-10 < current_score:
                        ass = trial
                        current_score = score
                        counts[old] -= 1
                        counts[new_split] += 1
                        changed = True
                        break
                if changed:
                    break
        best_assign = tuple(ass)

    if best_assign is None:
        return None, {}
    out = {'train': [], 'val': [], 'test': []}
    names = ['train', 'val', 'test']
    assignment = {}
    for split_idx, (group, rows) in zip(best_assign, items):
        assignment[group] = names[split_idx]
        out[names[split_idx]].extend(rows)
    for split in out:
        out[split].sort(key=lambda r: r['image_path'].name)
    return out, assignment


def locked_ue5_test_ids(previous_output: Path) -> set[str]:
    test_dir = previous_output / 'images' / 'test'
    if not test_dir.is_dir():
        return set()
    ids = set()
    for p in test_dir.iterdir():
        if p.is_file() or p.is_symlink():
            stem = p.stem
            if stem.startswith('uov_'):
                ids.add(stem[4:])
    return ids



def split_ue5_chronological(
    records: list[dict[str, Any]],
    ratios=(0.8, 0.1, 0.1),
    search_radius: int = 100,
    align_group_size: int = 4,
):
    """Split UrbanOmniView chronologically, selecting natural timestamp gaps near targets.

    The JSON audit shows monotonically increasing frame/engine_frame/timestamp values.
    We therefore keep the acquisition order intact. Boundaries are selected near the
    80% and 90% targets by maximizing the timestamp gap in a local search window.
    When possible, boundaries are aligned to four-frame groups to avoid cutting a
    likely capture burst in half. No images are discarded.
    """
    if len(records) < 10:
        raise ValueError('Too few UrbanOmniView records for chronological splitting')

    rows = sorted(records, key=lambda r: int(r['frame_index']))
    frames = [int(r['frame_index']) for r in rows]
    times = [float(r['timestamp_epoch']) for r in rows]
    engine_frames = [int(r['engine_frame']) for r in rows]

    if len(set(frames)) != len(frames):
        raise ValueError('Duplicate UrbanOmniView frame_index values detected')
    if any(times[i] <= times[i - 1] for i in range(1, len(times))):
        raise ValueError('UrbanOmniView timestamp_utc is not strictly increasing by frame')
    if any(engine_frames[i] <= engine_frames[i - 1] for i in range(1, len(engine_frames))):
        raise ValueError('UrbanOmniView engine_frame is not strictly increasing by frame')

    n = len(rows)
    ratio_sum = float(sum(ratios))
    if ratio_sum <= 0:
        raise ValueError('Invalid UrbanOmniView split ratios')
    train_target = int(round(n * ratios[0] / ratio_sum))
    val_target = int(round(n * (ratios[0] + ratios[1]) / ratio_sum))
    targets = [train_target, val_target]

    boundaries = []
    boundary_report = {}
    previous_boundary = 0

    for boundary_number, target in enumerate(targets, start=1):
        lo = max(previous_boundary + 1, target - search_radius)
        hi = min(n - 1, target + search_radius)
        candidate_indices = list(range(lo, hi + 1))

        aligned = [
            i for i in candidate_indices
            if align_group_size <= 1
            or (
                frames[i] % align_group_size == 0
                and frames[i - 1] == frames[i] - 1
            )
        ]
        used_alignment = bool(aligned)
        if aligned:
            candidate_indices = aligned

        # Maximize the natural acquisition pause; use closeness to target as tie-breaker.
        best_i = max(
            candidate_indices,
            key=lambda i: (times[i] - times[i - 1], -abs(i - target))
        )
        gap_seconds = times[best_i] - times[best_i - 1]
        engine_gap = engine_frames[best_i] - engine_frames[best_i - 1]

        boundaries.append(best_i)
        boundary_report[f'boundary_{boundary_number}'] = {
            'target_index': target,
            'actual_index': best_i,
            'previous_frame': frames[best_i - 1],
            'next_frame': frames[best_i],
            'timestamp_gap_seconds': round(gap_seconds, 3),
            'engine_frame_gap': engine_gap,
            'aligned_to_group_size': align_group_size if used_alignment else None,
            'search_radius': search_radius,
        }
        previous_boundary = best_i

    b1, b2 = boundaries
    if not (0 < b1 < b2 < n):
        raise RuntimeError(f'Invalid chronological split boundaries: {boundaries}')

    out = {
        'train': rows[:b1],
        'val': rows[b1:b2],
        'test': rows[b2:],
    }
    return out, boundary_report, 'chronological_natural_gap'


def split_ue5_preserving_old_test(records: list[dict[str, Any]], previous_output: Path,
                                  seed: int, preserve_previous: bool = False):
    # A fresh 80/10/10 split is the default. Reuse the old UE5 test set only on explicit request.
    locked_ids = locked_ue5_test_ids(previous_output) if preserve_previous else set()
    by_stem = {r['stem']: r for r in records}
    locked = [by_stem[x] for x in sorted(locked_ids) if x in by_stem]
    if not locked:
        # If scene/sequence groups are available, keep complete scenes in one split.
        group_modes = {r.get('group_mode') for r in records}
        meaningful = group_modes - {'image', None}
        if meaningful:
            grouped, assignment = split_by_group(records, ratios=(0.8, 0.1, 0.1), seed=seed)
            if grouped is not None:
                return grouped, 0, assignment, 'scene_or_sequence_group'
        return stratified_image_split(records, seed=seed), 0, {}, 'image_level'

    locked_set = {r['stem'] for r in locked}
    remaining = [r for r in records if r['stem'] not in locked_set]
    # Keep the old test set intact; divide the remaining 90% in an 8:1 train:val ratio.
    grouped = None
    if any(r.get('group_mode') not in ('image', None) for r in remaining):
        grouped, _ = split_by_group(remaining, ratios=(8/9, 1/9, 0.0), seed=seed)
    if grouped is None:
        grouped = stratified_image_split(remaining, ratios=(8/9, 1/9, 0.0), seed=seed)
    out = {'train': grouped['train'], 'val': grouped['val'], 'test': locked}
    for split in out:
        out[split].sort(key=lambda r: r['image_path'].name)
    return out, len(locked), {}, 'preserved_previous_test'


def split_dair(records: list[dict[str, Any]], seed=42, allow_image_fallback=True):
    # Strictest available grouping first: intersection, then batch/sequence.
    if records and all(r.get('group_mode') == 'intersection_loc' for r in records):
        grouped, assignment = split_by_group(records, seed=seed)
        if grouped is not None:
            return grouped, assignment, 'intersection_loc'

    # Some DAIR-V2X-I releases omit intersection_loc but retain batch/sequence ids.
    if records and all(r.get('group_mode') == 'batch_or_sequence' for r in records):
        grouped, assignment = split_by_group(records, seed=seed)
        if grouped is not None:
            return grouped, assignment, 'batch_or_sequence'

    # If intersection groups exist but are too few, try their secondary batch ids.
    if records and all(r.get('group_alt') for r in records):
        alternate = [dict(r, group=r['group_alt']) for r in records]
        grouped, assignment = split_by_group(alternate, seed=seed)
        if grouped is not None:
            original_by_key = {(r['source'], r['stem']): r for r in records}
            grouped = {s: [original_by_key[(r['source'], r['stem'])] for r in rows]
                       for s, rows in grouped.items()}
            return grouped, assignment, 'batch_or_sequence_fallback'

    if allow_image_fallback:
        print('[WARN] DAIR has fewer than three usable intersection/batch groups or lacks group metadata; using image-level split. Check leakage risk in dataset_summary.json.')
        return stratified_image_split(records, seed=seed), {}, 'image_level_fallback'
    raise RuntimeError('DAIR cannot be split into three leakage-resistant groups. Add group metadata or allow image-level fallback.')


# -----------------------------------------------------------------------------
# Output / validation
# -----------------------------------------------------------------------------
def stage_image(src: Path, dst: Path, copy_images: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(f'Output collision: {dst}')
    if copy_images:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def draw_preview(image_path: Path, instances: list[dict[str, Any]], out_path: Path):
    if cv2 is None:
        return False
    im = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if im is None:
        return False
    for inst in instances:
        pts = np.round(np.asarray(inst['points'])).astype(int)
        color = COLORS[int(inst['cls'])]
        for a, b in BOX_EDGES:
            cv2.line(im, tuple(pts[a]), tuple(pts[b]), color, 2, cv2.LINE_AA)
        for i, (x, y) in enumerate(pts):
            cv2.circle(im, (int(x), int(y)), 4, color, -1, cv2.LINE_AA)
            cv2.putText(im, str(i), (int(x) + 3, int(y) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        x0, y0 = pts.min(axis=0)
        cv2.putText(im, CLASS_NAMES[int(inst['cls'])], (int(x0), max(15, int(y0) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), im))


def split_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    cc = Counter()
    image_class = Counter()
    for r in records:
        cc.update(r['class_counts'])
        for c in r['class_counts']:
            image_class[c] += 1
    return {
        'images': len(records),
        'instances': int(sum(cc.values())),
        'classes_instances': {CLASS_NAMES[i]: int(cc[i]) for i in range(3)},
        'images_containing_class': {CLASS_NAMES[i]: int(image_class[i]) for i in range(3)},
        'images_with_targets': int(sum(bool(r['instances']) for r in records)),
        'empty_label_images': int(sum(not r['instances'] for r in records)),
    }


def validate_output(out: Path) -> dict[str, int]:
    bad = []
    lines_total = 0
    label_files = sorted((out / 'labels').glob('*/*.txt'))
    for label in label_files:
        for line_no, line in enumerate(label.read_text(encoding='utf-8').splitlines(), 1):
            f = line.split()
            lines_total += 1
            if len(f) != 21:
                bad.append((str(label), line_no, f'columns={len(f)}'))
                continue
            try:
                cls = int(f[0])
                vals = np.asarray(f[1:], dtype=np.float64)
            except Exception:
                bad.append((str(label), line_no, 'non-numeric'))
                continue
            if cls not in (0, 1, 2) or not np.isfinite(vals).all():
                bad.append((str(label), line_no, 'class/nonfinite'))
                continue
            if np.any(vals < -1e-6) or np.any(vals > 1.000001):
                bad.append((str(label), line_no, 'normalized value out of range'))
            if vals[2] <= 0 or vals[3] <= 0:
                bad.append((str(label), line_no, 'non-positive bbox size'))
    if bad:
        raise RuntimeError('Output label validation failed: ' + repr(bad[:20]))
    return {'label_files': len(label_files), 'label_lines': lines_total}


def main() -> int:
    ap = argparse.ArgumentParser(description='Build unified KITTI + UE5 UrbanOmniView + DAIR-V2X-I YOLO-Pose dataset.')
    ap.add_argument('--kitti-root', type=Path, default=DEFAULT_KITTI)
    ap.add_argument('--ue5-root', type=Path, default=DEFAULT_UE5)
    ap.add_argument('--dair-root', type=Path, default=DEFAULT_DAIR)
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dair-min-iou', type=float, default=0.0,
                    help='Optional minimum IoU between projected 3D envelope and official DAIR 2D bbox; default 0 keeps all valid in-frame objects.')
    ap.add_argument('--copy-images', action='store_true', help='Copy images instead of symlinking.')
    ap.add_argument('--preview-per-source-split', type=int, default=8)
    ap.add_argument('--limit-per-source-split', type=int, default=0,
                    help='Smoke test only: keep at most N images per source per split; 0 means all.')
    ap.add_argument('--dry-run', action='store_true', help='Convert and report counts, but do not write output files.')
    ap.add_argument('--uov-boundary-search', type=int, default=100,
                    help='Search +/- this many frames around the 80%%/90%% UrbanOmniView split targets for natural timestamp gaps.')
    ap.add_argument('--uov-align-group-size', type=int, default=4,
                    help='Prefer UrbanOmniView split boundaries aligned to this many consecutive frames; default 4.')
    ap.add_argument('--kitti-width-sign', type=int, choices=(-1, 1), default=1,
                    help='KITTI local width-side sign; +1 follows the DAIR canonical side order.')
    ap.add_argument('--strict-dair-groups', action='store_true',
                    help='Fail rather than image-split if DAIR has insufficient intersection/batch groups.')
    args = ap.parse_args()

    kitti_root = args.kitti_root.expanduser().resolve()
    ue5_root = args.ue5_root.expanduser().resolve()
    dair_root = args.dair_root.expanduser().resolve()
    out = args.out.expanduser().resolve()

    for p in (kitti_root, ue5_root, dair_root):
        if not p.is_dir():
            raise FileNotFoundError(f'Source root does not exist: {p}')
    if out.exists() and not args.dry_run:
        raise FileExistsError(f'Output already exists: {out}. Choose a new --out; no files were overwritten.')
    if not (0 <= args.dair_min_iou < 1):
        raise ValueError('--dair-min-iou must be in [0,1)')
    if args.uov_boundary_search < 0:
        raise ValueError('--uov-boundary-search must be >= 0')
    if args.uov_align_group_size < 1:
        raise ValueError('--uov-align-group-size must be >= 1')

    print('========== STEP 1: UE5 CORNER ORDER AUDIT ==========')
    raw_patterns, order_report = inspect_ue5_corner_order(ue5_root)
    print(json.dumps(order_report, ensure_ascii=False, indent=2))
    print('Canonical output order: DAIR official bottom-first 0..3, corresponding top 4..7')

    stats = Counter()
    print('\n========== STEP 2: CONVERT ALL THREE SOURCES ==========')
    kitti = collect_kitti(kitti_root, stats, width_sign=args.kitti_width_sign)
    ue5 = collect_ue5(ue5_root, stats, raw_patterns)
    dair = collect_dair(dair_root, stats, min_iou=args.dair_min_iou)
    print(f'KITTI labelled images: {len(kitti)}')
    print(f'UE5 image/JSON pairs:  {len(ue5)}')
    print(f'DAIR image/label pairs: {len(dair)}')
    print(f'Total candidate images: {len(kitti) + len(ue5) + len(dair)}')
    print('\n========== CONVERSION FILTER / ERROR COUNTS ==========')
    if stats:
        for key in sorted(stats):
            print(f'  {key}: {stats[key]}')
    else:
        print('  No conversion filters/errors recorded.')
    if not kitti or not ue5 or not dair:
        raise RuntimeError('At least one source produced zero usable records. Check paths/layout before proceeding.')

    print('\n========== STEP 3: UNIFIED SPLIT (80/10/10 TARGET) ==========')
    kitti_split = stratified_image_split(kitti, seed=args.seed)
    ue5_split, ue5_boundary_report, ue5_split_mode = split_ue5_chronological(
        ue5,
        ratios=(0.8, 0.1, 0.1),
        search_radius=args.uov_boundary_search,
        align_group_size=args.uov_align_group_size,
    )
    dair_split, dair_group_assignment, dair_split_mode = split_dair(
        dair, seed=args.seed + 2, allow_image_fallback=not args.strict_dair_groups
    )
    print('UE5 previous random test split preserved: no (fresh chronological split)')
    print(f'UE5 split mode: {ue5_split_mode}')
    print('UE5 chronological boundaries:')
    for boundary_name, info in sorted(ue5_boundary_report.items()):
        print(
            f"  {boundary_name}: target={info['target_index']}, "
            f"actual={info['actual_index']}, "
            f"frames={info['previous_frame']}->{info['next_frame']}, "
            f"timestamp_gap={info['timestamp_gap_seconds']:.3f}s, "
            f"engine_gap={info['engine_frame_gap']}, "
            f"aligned_group={info['aligned_to_group_size']}"
        )
    print(f'DAIR split mode: {dair_split_mode}')
    if dair_group_assignment:
        print(f'DAIR groups assigned: {len(dair_group_assignment)}')
        for g, s in sorted(dair_group_assignment.items()):
            print(f'  {g}: {s}')

    source_splits = {'kitti': kitti_split, 'uov': ue5_split, 'dair': dair_split}
    if args.limit_per_source_split > 0:
        for source in source_splits:
            for split in ('train', 'val', 'test'):
                source_splits[source][split] = source_splits[source][split][:args.limit_per_source_split]

    unified = {
        split: source_splits['kitti'][split] + source_splits['uov'][split] + source_splits['dair'][split]
        for split in ('train', 'val', 'test')
    }

    dair_iou_by_class = {}
    for class_name in CLASS_NAMES:
        n_iou = int(stats[f'dair_projection_iou_n:{class_name}'])
        if n_iou:
            dair_iou_by_class[class_name] = {
                'n': n_iou,
                'mean_iou': float(stats[f'dair_projection_iou_sum:{class_name}'] / n_iou),
                'iou_ge_0_50_rate': float(stats[f'dair_projection_iou_ge50:{class_name}'] / n_iou),
                'iou_ge_0_70_rate': float(stats[f'dair_projection_iou_ge70:{class_name}'] / n_iou),
            }

    summary = {
        'task': 'Unified KITTI + UE5 UrbanOmniView + DAIR-V2X-I / YOLO11-Pose 3D cuboid keypoints',
        'classes': {str(i): CLASS_NAMES[i] for i in range(3)},
        'class_definitions': {
            '0': 'Car: passenger cars, trucks, vans and buses',
            '1': 'Person: pedestrians',
            '2': 'Bike: bicycles, motorcycles and scooters',
        },
        'class_mapping': {
            'car': 'Car/Truck/Van/Bus',
            'person': 'Pedestrian/Person',
            'bike': 'Cyclist/Bicycle/Bike/Motorcyclist/Motorcycle/Scooter',
        },
        'excluded_source_classes': [
            'Tricyclist', 'Trafficcone', 'Barrowlist', 'all unmapped classes',
        ],
        'keypoint_order': 'DAIR official: 0..3 bottom corners (+L,+W), (+L,-W), (-L,-W), (-L,+W); 4..7 corresponding top corners',
        'label_format': '21 columns: cls cx cy w h x1 y1 ... x8 y8',
        'geometry_3d_sidecar': {
            'file': 'geometry_3d_manifest.jsonl',
            'purpose': 'preserve the eight 3D corners in canonical order for future 3D-space training',
            'units_policy': 'source units are retained explicitly; no unverified UE5 scale conversion is applied',
            'coordinate_policy': 'source-native frame plus camera-frame corners where available; not silently fused into one global frame',
        },
        'kpt_shape': [8, 2],
        'flip_idx': FLIP_IDX,
        'seed': args.seed,
        'split_policy': {
            'target_ratio': {'train': 0.8, 'val': 0.1, 'test': 0.1},
            'KITTI': 'labelled training/ only; deterministic multi-label image-level split; official testing/ excluded because labels are not public',
            'UE5': (
                f'{ue5_split_mode}; chronological order; natural timestamp-gap boundaries '
                f'near 80%/90% targets; preferred four-frame alignment; '
                f'search_radius={args.uov_boundary_search}'
            ),
            'DAIR': dair_split_mode,
            'note': '80/10/10 is a target per source; grouped DAIR/UE5 splits may deviate to prevent sequence/intersection leakage.',
        },
        'source_counts': {source: {s: split_summary(source_splits[source][s]) for s in ('train', 'val', 'test')} for source in source_splits},
        'unified_counts': {s: split_summary(unified[s]) for s in ('train', 'val', 'test')},
        'ue5_corner_order_audit': order_report,
        'ue5_chronological_boundaries': ue5_boundary_report,
        'dair_group_assignment': dair_group_assignment,
        'dair_projection_iou_mean': (stats['dair_projection_iou_sum'] / stats['dair_projection_iou_n'] if stats['dair_projection_iou_n'] else None),
        'dair_projection_iou_by_class': dair_iou_by_class,
        'conversion_stats': dict(stats),
        'dair_min_iou': args.dair_min_iou,
        'kitti_width_sign': args.kitti_width_sign,
        'image_staging': 'copy' if args.copy_images else 'symlink',
    }

    print('\n========== SPLIT COUNTS ==========')
    for source, splits in summary['source_counts'].items():
        print(f'[{source}]')
        for split, info in splits.items():
            print(f'  {split:5s}: images={info["images"]:5d}, instances={info["instances"]:7d}, classes={info["classes_instances"]}')
    print('[UNIFIED]')
    for split, info in summary['unified_counts'].items():
        print(f'  {split:5s}: images={info["images"]:5d}, instances={info["instances"]:7d}, classes={info["classes_instances"]}')
    if summary['dair_projection_iou_mean'] is not None:
        print(f'DAIR mean projected-3D-bbox vs official-2D-bbox IoU: {summary["dair_projection_iou_mean"]:.4f}')

    if args.dry_run:
        print('\nDRY RUN complete: no output files were created.')
        print('Review source counts, corner audit, DAIR split mode and conversion warnings before the full build.')
        return 0

    out.mkdir(parents=True, exist_ok=False)
    for split in ('train', 'val', 'test'):
        (out / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / split).mkdir(parents=True, exist_ok=True)

    manifest = []
    geometry_manifest = []
    preview_count = Counter()
    output_names = set()
    for split in ('train', 'val', 'test'):
        for rec in unified[split]:
            src = rec['image_path']
            stem = f"{rec['source']}_{src.stem}"
            if stem in output_names:
                digest = hashlib.sha1(str(src.resolve()).encode('utf-8')).hexdigest()[:8]
                stem = f'{stem}_{digest}'
            output_names.add(stem)
            dst_img = out / 'images' / split / f'{stem}{src.suffix.lower()}'
            dst_lab = out / 'labels' / split / f'{stem}.txt'
            stage_image(src, dst_img, args.copy_images)
            w_img, h_img = read_image_size(src)
            lines = [
                make_pose_line(int(inst['cls']), np.asarray(inst['points']), w_img, h_img, inst.get('bbox'))
                for inst in rec['instances']
            ]
            dst_lab.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')

            preview_key = (rec['source'], split)
            if preview_count[preview_key] < args.preview_per_source_split and rec['instances']:
                draw_preview(src, rec['instances'], out / 'preview' / rec['source'] / split / f'{stem}.jpg')
                preview_count[preview_key] += 1

            source_class_counts = Counter(
                str(inst.get('source_class', '<unknown>')) for inst in rec['instances']
            )
            manifest.append({
                'source': rec['source'], 'split': split, 'group': rec['group'],
                'group_mode': rec.get('group_mode', ''),
                'frame_index': rec.get('frame_index', ''),
                'engine_frame': rec.get('engine_frame', ''),
                'timestamp_utc': rec.get('timestamp_utc', ''),
                'image': str(dst_img), 'label': str(dst_lab),
                'instances': len(rec['instances']),
                'car': rec['class_counts'].get(0, 0),
                'person': rec['class_counts'].get(1, 0),
                'bike': rec['class_counts'].get(2, 0),
                'source_class_counts': json.dumps(dict(source_class_counts), ensure_ascii=False, sort_keys=True),
            })
            geometry_objects = []
            for inst in rec['instances']:
                geom = inst.get('geometry3d')
                if geom is not None:
                    geometry_objects.append({
                        'class_id': int(inst['cls']),
                        'class_name': CLASS_NAMES[int(inst['cls'])],
                        'source_class': inst.get('source_class', '<unknown>'),
                        'corners_2d_image': np.asarray(inst['points']).tolist(),
                        **geom,
                    })
            geometry_manifest.append({
                'source': rec['source'], 'split': split,
                'frame_index': rec.get('frame_index'),
                'engine_frame': rec.get('engine_frame'),
                'timestamp_utc': rec.get('timestamp_utc'),
                'image': str(dst_img), 'label_2d': str(dst_lab),
                'objects': geometry_objects,
            })

    yaml_text = (
        f'path: {out}\n'
        'train: images/train\n'
        'val: images/val\n'
        'test: images/test\n\n'
        'nc: 3\n'
        "names: ['car', 'person', 'bike']\n"
        'kpt_shape: [8, 2]\n'
        f'flip_idx: {FLIP_IDX}\n'
    )
    (out / 'data.yaml').write_text(yaml_text, encoding='utf-8')
    with (out / 'split_manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'source', 'split', 'group', 'group_mode',
                'frame_index', 'engine_frame', 'timestamp_utc',
                'image', 'label', 'instances', 'car', 'person', 'bike',
                'source_class_counts',
            ],
        )
        writer.writeheader()
        writer.writerows(manifest)
    with (out / 'geometry_3d_manifest.jsonl').open('w', encoding='utf-8') as f:
        for row in geometry_manifest:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    validation = validate_output(out)
    summary['output_root'] = str(out)
    summary['output_validation'] = validation
    summary['manifest_rows'] = len(manifest)
    summary['geometry_manifest_rows'] = len(geometry_manifest)
    summary['geometry_manifest_objects'] = int(sum(len(row['objects']) for row in geometry_manifest))
    (out / 'dataset_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print('\n========== BUILD COMPLETE ==========')
    print(f'Output: {out}')
    print(f'Data YAML: {out / "data.yaml"}')
    print(f'Manifest: {out / "split_manifest.csv"}')
    print(f'3D geometry sidecar: {out / "geometry_3d_manifest.jsonl"}')
    print(f'Summary: {out / "dataset_summary.json"}')
    print(f'Preview: {out / "preview"}')
    print(f'Validated label files: {validation["label_files"]}; instances: {validation["label_lines"]}')
    print('Original source files were not modified.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'\nERROR: {exc}', file=sys.stderr)
        raise
