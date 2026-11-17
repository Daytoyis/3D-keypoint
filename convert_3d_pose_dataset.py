#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert UrbanOmniView UE5 JSON annotations + KITTI 3D labels into one
YOLO11-Pose V1 dataset (8 keypoints x 2 coordinates = 21 columns).

This script is intentionally non-destructive:
- Original UrbanOmniView JSON/images are never modified.
- Original KITTI labels/calibration/images are never modified.
- Images are symlinked by default (use --copy-images to copy them).
- Instances whose 3D cuboid has any projected corner outside the image are
  skipped, because V1 kpt_shape=[8,2] has no per-keypoint visibility mask.

Important corner-order handling:
- The script audits UrbanOmniView corners_world using each object's quaternion.
- It infers the eight local corner sign-patterns from the actual JSON files.
- KITTI corners are generated in the same inferred local-corner order.
- The released repository contains conflicting prose about top/bottom order;
  this script follows the actual raw JSON annotations rather than hard-coding
  a prose convention.

Default paths match the user's server:
  UrbanOmniView: /mnt/sdb/lys/yolo11/datasets/UrbanOmniView
  KITTI:         /mnt/sdb/lys/MonoCoP/data/KITTIDataset
  Output:        /mnt/sdb/lys/yolo11/datasets/YOLO11_3D_Pose_V1
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

DEFAULT_URBAN = Path("/mnt/sdb/lys/yolo11/datasets/UrbanOmniView")
DEFAULT_KITTI = Path("/mnt/sdb/lys/MonoCoP/data/KITTIDataset")
DEFAULT_OUT = Path("/mnt/sdb/lys/yolo11/datasets/YOLO11_3D_Pose_V1")

CLASS_NAMES = ["car", "person", "bike"]
URBAN_CLASS_MAP = {
    "car": 0, "vehicle": 0, "truck": 0, "van": 0, "bus": 0,
    "person": 1, "pedestrian": 1,
    "bike": 2, "bicycle": 2, "cyclist": 2, "motorcycle": 2, "scooter": 2,
}
KITTI_CLASS_MAP = {
    "Car": 0,
    "Van": 0,
    "Truck": 0,
    "Pedestrian": 1,
    "Cyclist": 2,
    # Person_sitting, Tram, Misc and DontCare are deliberately excluded.
}

# Expected set of eight cuboid corner signs in the object's local frame:
# (longitudinal X, lateral Y, vertical Z).
# The order within this list is NOT assumed; it is inferred from JSON.
EXPECTED_CORNER_PATTERNS = {
    (-1, -1, -1), (+1, -1, -1), (+1, +1, -1), (-1, +1, -1),
    (-1, -1, +1), (+1, -1, +1), (+1, +1, +1), (-1, +1, +1),
}

BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def q_xyzw_to_matrix(q: list[float] | np.ndarray) -> np.ndarray:
    """Quaternion (x,y,z,w) to a 3x3 rotation matrix."""
    x, y, z, w = [float(v) for v in q]
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if norm < 1e-12:
        raise ValueError("zero-length quaternion")
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def sign3(v: np.ndarray, eps: float = 1e-5) -> tuple[int, int, int]:
    out = []
    for x in v:
        if abs(float(x)) < eps:
            raise ValueError("corner lies too close to a local axis")
        out.append(1 if x > 0 else -1)
    return tuple(out)  # type: ignore[return-value]


def inspect_urban_corner_order(root: Path, max_json_files: int = 0) -> tuple[list[tuple[int, int, int]], dict[str, Any]]:
    """Infer the stable index->local-sign mapping from raw UrbanOmniView JSON."""
    label_dir = root / "labels"
    json_files = sorted(label_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON annotations found in {label_dir}")
    if max_json_files > 0:
        json_files = json_files[:max_json_files]

    per_index: list[Counter] = [Counter() for _ in range(8)]
    used_objects = 0
    files_read = 0
    z_order_votes = Counter()
    errors = Counter()

    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            errors["json_read_error"] += 1
            continue
        files_read += 1
        for obj in data.get("objects", []):
            try:
                cw = np.asarray(obj["corners_world"], dtype=np.float64)
                center = np.asarray(obj["center"], dtype=np.float64)
                quat = obj["quat_xyzw"]
                if cw.shape != (8, 3) or center.shape != (3,) or len(quat) != 4:
                    errors["bad_world_corner_shape"] += 1
                    continue
                R = q_xyzw_to_matrix(quat)
                # Row-vector equivalent of R.T @ (corner-center).
                local = (cw - center) @ R
                patterns = [sign3(local[i]) for i in range(8)]
                if len(set(patterns)) != 8:
                    errors["duplicate_corner_pattern"] += 1
                    continue
                for i, pat in enumerate(patterns):
                    per_index[i][pat] += 1
                if np.mean(local[:4, 2]) < np.mean(local[4:, 2]):
                    z_order_votes["first_half_bottom"] += 1
                else:
                    z_order_votes["first_half_top"] += 1
                used_objects += 1
            except Exception:
                errors["world_order_audit_error"] += 1

    if used_objects < 1:
        raise RuntimeError(
            "Could not infer corner order from corners_world + center + quat_xyzw. "
            "Do not proceed with KITTI conversion until this is resolved."
        )

    patterns: list[tuple[int, int, int]] = []
    consistency: dict[str, float] = {}
    for i, counts in enumerate(per_index):
        if not counts:
            raise RuntimeError(f"No usable world-corner order evidence for keypoint {i}.")
        pat, count = counts.most_common(1)[0]
        ratio = count / sum(counts.values())
        consistency[str(i)] = round(ratio, 5)
        if ratio < 0.95:
            raise RuntimeError(
                f"UrbanOmniView corner index {i} is inconsistent "
                f"({ratio:.1%} agreement). Refusing to guess the KITTI order."
            )
        patterns.append(pat)

    if len(set(patterns)) != 8 or set(patterns) != EXPECTED_CORNER_PATTERNS:
        raise RuntimeError(
            "The inferred UrbanOmniView corner patterns are not the expected "
            "eight cuboid corners. Inspect corner_order_audit before converting."
        )

    total_z_votes = sum(z_order_votes.values())
    first_bottom = z_order_votes["first_half_bottom"]
    order_name = (
        "indices_0_3_bottom_indices_4_7_top"
        if first_bottom >= total_z_votes / 2
        else "indices_0_3_top_indices_4_7_bottom"
    )

    report = {
        "json_files_found": len(sorted(label_dir.glob("*.json"))),
        "json_files_read": files_read,
        "objects_used_for_order_audit": used_objects,
        "index_pattern_agreement": consistency,
        "index_local_sign_patterns": [list(p) for p in patterns],
        "first_half_bottom_votes": z_order_votes["first_half_bottom"],
        "first_half_top_votes": z_order_votes["first_half_top"],
        "inferred_vertical_order": order_name,
        "audit_errors": dict(errors),
    }
    return patterns, report


def read_image_size(path: Path) -> tuple[int, int]:
    if cv2 is not None:
        im = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if im is None:
            raise ValueError(f"OpenCV cannot read image: {path}")
        h, w = im.shape[:2]
        return int(w), int(h)
    try:
        from PIL import Image
        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception as exc:
        raise RuntimeError(
            "Image reading requires opencv-python or Pillow in the active environment."
        ) from exc


def resolve_urban_image(root: Path, data: dict[str, Any], json_path: Path) -> Path | None:
    rel = data.get("image")
    if rel:
        rel_path = Path(str(rel).replace("\\", "/"))
        candidate = rel_path if rel_path.is_absolute() else root / rel_path
        if candidate.is_file():
            return candidate
    stem = str(data.get("frame", json_path.stem))
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
        candidate = root / "rgb" / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def points_from_urban_object(
    obj: dict[str, Any],
    camera: dict[str, Any],
    stats: Counter | None = None
) -> np.ndarray | None:
    """
    Prefer valid corners_image.
    If unavailable or malformed, project corners_camera using K.
    Never crash because of one malformed object.
    """

    raw_image = obj.get("corners_image")

    # 1. Try original image coordinates
    if raw_image is not None:
        try:
            pts = np.asarray(raw_image, dtype=np.float64)

            if pts.shape == (8, 2) and np.isfinite(pts).all():
                return pts

            if stats is not None:
                if pts.shape != (8, 2):
                    stats["urban_corners_image_bad_shape"] += 1
                else:
                    stats["urban_corners_image_nonfinite"] += 1

        except (TypeError, ValueError, OverflowError):
            if stats is not None:
                stats["urban_corners_image_ragged"] += 1
    else:
        if stats is not None:
            stats["urban_corners_image_missing"] += 1

    # 2. Fallback: project camera coordinates using intrinsic matrix K
    if "corners_camera" in obj and "K" in camera:
        try:
            cam = np.asarray(
                obj["corners_camera"], dtype=np.float64
            )
            K = np.asarray(camera["K"], dtype=np.float64)

            if (
                cam.shape == (8, 3)
                and K.shape == (3, 3)
                and np.isfinite(cam).all()
                and np.isfinite(K).all()
            ):
                homog = (K @ cam.T).T

                if (
                    np.isfinite(homog).all()
                    and np.all(homog[:, 2] > 1e-9)
                ):
                    pts = homog[:, :2] / homog[:, 2:3]

                    if np.isfinite(pts).all():
                        if stats is not None:
                            stats["urban_camera_projection_fallback"] += 1
                        return pts

        except (TypeError, ValueError, OverflowError):
            pass

        if stats is not None:
            stats["urban_camera_projection_failed"] += 1

    else:
        if stats is not None:
            stats["urban_camera_projection_unavailable"] += 1

    return None


def points_inside_image(points: np.ndarray, width: int, height: int) -> bool:
    return bool(
        np.isfinite(points).all()
        and np.all(points[:, 0] >= 0)
        and np.all(points[:, 1] >= 0)
        and np.all(points[:, 0] <= width - 1)
        and np.all(points[:, 1] <= height - 1)
    )


def bbox_from_projected_points(points: np.ndarray, width: int, height: int) -> tuple[float, float, float, float] | None:
    """Use the tight axis-aligned envelope of all eight projected cuboid corners."""
    x1 = max(0.0, float(np.min(points[:, 0])))
    y1 = max(0.0, float(np.min(points[:, 1])))
    x2 = min(float(width - 1), float(np.max(points[:, 0])))
    y2 = min(float(height - 1), float(np.max(points[:, 1])))
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    return x1, y1, x2, y2


def make_yolo_pose_line(cls: int, points: np.ndarray, width: int, height: int) -> str:
    box = bbox_from_projected_points(points, width, height)
    if box is None:
        raise ValueError("degenerate projected bounding box")
    x1, y1, x2, y2 = box
    vals = [
        cls,
        ((x1 + x2) / 2) / width,
        ((y1 + y2) / 2) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    ]
    for x, y in points:
        vals.extend([float(x) / width, float(y) / height])
    return " ".join(
        str(int(v)) if i == 0 else f"{float(v):.6f}"
        for i, v in enumerate(vals)
    )


def draw_preview(image_path: Path, instances: list[dict[str, Any]], output_path: Path) -> bool:
    if cv2 is None:
        return False
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return False
    colors = [(0, 220, 0), (0, 180, 255), (255, 180, 0)]
    for inst in instances:
        pts = np.asarray(inst["points"], dtype=np.float64)
        cls = int(inst["cls"])
        color = colors[cls % len(colors)]
        p = np.round(pts).astype(int)
        for a, b in BOX_EDGES:
            cv2.line(image, tuple(p[a]), tuple(p[b]), color, 2, cv2.LINE_AA)
        for i, xy in enumerate(p):
            cv2.circle(image, tuple(xy), 4, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(
                image, str(i), (int(xy[0]) + 4, int(xy[1]) - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
            )
    max_width = 1600
    if image.shape[1] > max_width:
        scale = max_width / image.shape[1]
        image = cv2.resize(
            image, (max_width, int(image.shape[0] * scale)),
            interpolation=cv2.INTER_AREA
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), image))


def stage_image(src: Path, dst: Path, copy_images: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(f"Output image already exists: {dst}")
    if copy_images:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def read_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing KITTI split file: {path}")
    return [x.strip() for x in path.read_text().splitlines() if x.strip()]


def find_kitti_image(image_dir: Path, stem: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG"):
        p = image_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def read_p2(calib_path: Path) -> np.ndarray:
    for line in calib_path.read_text().splitlines():
        if line.startswith("P2:"):
            vals = np.asarray([float(x) for x in line.split()[1:]], dtype=np.float64)
            if vals.size != 12:
                raise ValueError(f"Invalid P2 matrix in {calib_path}")
            return vals.reshape(3, 4)
    raise ValueError(f"No P2 matrix in {calib_path}")


def kitti_cuboid_camera(
    h: float,
    w: float,
    length: float,
    location: np.ndarray,
    rotation_y: float,
    uov_patterns: list[tuple[int, int, int]],
    kitti_width_sign: int = -1,
) -> np.ndarray:
    """
    Generate KITTI 3D corners in the same index order inferred from Urban JSON.

    KITTI local frame: x along object length, y down, z lateral.
    Urban local frame: x forward/length, y right/width, z up.
    Thus Urban +Y (right) maps to KITTI -Z (the standard KITTI local lateral
    axis has the opposite sign for a forward-facing object); Urban +Z (up)
    maps to KITTI -Y (down).

    location is the KITTI bottom-face center in rectified camera coordinates.
    """
    # The sign of the KITTI local lateral axis is exposed for auditability.
    # Default -1 corresponds to Urban local +Y (right) -> KITTI local -Z.
    x_local = np.asarray([sx * length / 2.0 for sx, sy, sz in uov_patterns])
    z_local = np.asarray([kitti_width_sign * sy * w / 2.0 for sx, sy, sz in uov_patterns])
    y_local = np.asarray([0.0 if sz < 0 else -h for sx, sy, sz in uov_patterns])

    c, s = math.cos(rotation_y), math.sin(rotation_y)
    R_y = np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ], dtype=np.float64)

    local = np.vstack([x_local, y_local, z_local])
    camera = (R_y @ local).T + location.reshape(1, 3)
    return camera


def project_camera_points(points_camera: np.ndarray, P2: np.ndarray) -> np.ndarray | None:
    hom = (P2 @ np.hstack([points_camera, np.ones((8, 1), dtype=np.float64)]).T).T
    if not np.isfinite(hom).all() or np.any(hom[:, 2] <= 1e-6):
        return None
    return hom[:, :2] / hom[:, 2:3]


def urban_instances(data: dict[str, Any], width: int, height: int, stats: Counter) -> list[dict[str, Any]]:
    instances = []
    camera = data.get("camera", {})
    for obj in data.get("objects", []):
        label = str(obj.get("label", "")).strip().lower()
        if label not in URBAN_CLASS_MAP:
            stats["urban_unknown_class"] += 1
            continue
        cls = URBAN_CLASS_MAP[label]
        points = points_from_urban_object(obj, camera, stats)
        if points is None or points.shape != (8, 2) or not np.isfinite(points).all():
            stats["urban_missing_or_invalid_keypoints"] += 1
            continue
        if not points_inside_image(points, width, height):
            stats["urban_keypoint_outside_image"] += 1
            continue
        if bbox_from_projected_points(points, width, height) is None:
            stats["urban_degenerate_box"] += 1
            continue
        instances.append({"cls": cls, "points": points})
        stats[f"urban_class_{CLASS_NAMES[cls]}"] += 1
    return instances


def kitti_instances(
    label_path: Path,
    P2: np.ndarray,
    width: int,
    height: int,
    uov_patterns: list[tuple[int, int, int]],
    stats: Counter,
    kitti_width_sign: int,
) -> list[dict[str, Any]]:
    instances = []
    for line in label_path.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 15:
            stats["kitti_malformed_label_line"] += 1
            continue
        typ = fields[0]
        if typ not in KITTI_CLASS_MAP:
            stats[f"kitti_ignored_class_{typ}"] += 1
            continue
        cls = KITTI_CLASS_MAP[typ]
        try:
            h, w, length = map(float, fields[8:11])
            location = np.asarray(list(map(float, fields[11:14])), dtype=np.float64)
            rotation_y = float(fields[14])
            if min(h, w, length) <= 0 or location.shape != (3,):
                raise ValueError("invalid dimensions/location")
        except Exception:
            stats["kitti_invalid_3d_parameters"] += 1
            continue

        corners_camera = kitti_cuboid_camera(
            h, w, length, location, rotation_y, uov_patterns,
            kitti_width_sign=kitti_width_sign
        )
        points = project_camera_points(corners_camera, P2)
        if points is None:
            stats["kitti_corner_behind_camera"] += 1
            continue
        if not points_inside_image(points, width, height):
            stats["kitti_keypoint_outside_image"] += 1
            continue
        if bbox_from_projected_points(points, width, height) is None:
            stats["kitti_degenerate_box"] += 1
            continue
        instances.append({"cls": cls, "points": points})
        stats[f"kitti_class_{CLASS_NAMES[cls]}"] += 1
    return instances


def write_one_sample(
    image_path: Path,
    source: str,
    split: str,
    instances: list[dict[str, Any]],
    out: Path,
    copy_images: bool,
    preview_limit: int,
    preview_counts: Counter,
    manifest_rows: list[dict[str, str]],
) -> None:
    ext = image_path.suffix.lower()
    stem = f"{source}_{image_path.stem}"
    dst_image = out / "images" / split / f"{stem}{ext}"
    dst_label = out / "labels" / split / f"{stem}.txt"

    stage_image(image_path, dst_image, copy_images)
    width, height = read_image_size(image_path)
    lines = [
        make_yolo_pose_line(int(inst["cls"]), np.asarray(inst["points"]), width, height)
        for inst in instances
    ]
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    dst_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    key = (source, split)
    if preview_limit > 0 and preview_counts[key] < preview_limit:
        preview_path = out / "preview" / source / split / f"{stem}_preview.jpg"
        if draw_preview(image_path, instances, preview_path):
            preview_counts[key] += 1

    manifest_rows.append({
        "source": source,
        "split": split,
        "image": str(dst_image),
        "label": str(dst_label),
        "instances": str(len(instances)),
    })


def split_urban_pairs(root: Path, seed: int, val_ratio: float, test_ratio: float) -> dict[str, list[tuple[Path, Path]]]:
    label_dir = root / "labels"
    pairs = []
    for jf in sorted(label_dir.glob("*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        image_path = resolve_urban_image(root, data, jf)
        if image_path is not None:
            pairs.append((image_path, jf))
    if not pairs:
        raise RuntimeError(f"No matching UrbanOmniView rgb/JSON pairs found under {root}")
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be >=0 and sum to <1")

    rng = random.Random(seed)
    rng.shuffle(pairs)
    n = len(pairs)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    test = pairs[:n_test]
    val = pairs[n_test:n_test + n_val]
    train = pairs[n_test + n_val:]
    return {
        "train": sorted(train, key=lambda x: x[0].name),
        "val": sorted(val, key=lambda x: x[0].name),
        "test": sorted(test, key=lambda x: x[0].name),
    }


def split_kitti_pairs(root: Path) -> dict[str, list[tuple[Path, Path, Path]]]:
    train_root = root / "training"
    image_dir = train_root / "image_2"
    label_dir = train_root / "label_2"
    calib_dir = train_root / "calib"
    imagesets = root / "ImageSets"

    splits = {}
    for split in ("train", "val"):
        ids = read_ids(imagesets / f"{split}.txt")
        rows = []
        for stem in ids:
            image = find_kitti_image(image_dir, stem)
            label = label_dir / f"{stem}.txt"
            calib = calib_dir / f"{stem}.txt"
            if image is None or not label.is_file() or not calib.is_file():
                continue
            rows.append((image, label, calib))
        splits[split] = rows
    return splits


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert UrbanOmniView JSON + KITTI 3D labels to YOLO11-Pose V1."
    )
    parser.add_argument("--urban-root", type=Path, default=DEFAULT_URBAN)
    parser.add_argument("--kitti-root", type=Path, default=DEFAULT_KITTI)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.10,
                        help="UrbanOmniView validation fraction (default 0.10).")
    parser.add_argument("--test-ratio", type=float, default=0.10,
                        help="UrbanOmniView held-out test fraction (default 0.10).")
    parser.add_argument("--limit-per-split", type=int, default=0,
                        help="Smoke test: stage at most N images per source/split; 0 means all.")
    parser.add_argument("--max-preview-per-split", type=int, default=5)
    parser.add_argument("--copy-images", action="store_true",
                        help="Copy images instead of creating symlinks.")
    parser.add_argument("--no-preview", action="store_true",
                        help="Do not draw preview images.")
    parser.add_argument("--audit-only", action="store_true",
                        help="Only inspect UrbanOmniView corner order; do not create a dataset.")
    parser.add_argument("--kitti-width-sign", type=int, choices=(-1, 1), default=-1,
                        help="KITTI local width-axis sign mapping; default -1 maps Urban +right to KITTI -Z.")
    args = parser.parse_args()

    urban_root = args.urban_root.expanduser().resolve()
    kitti_root = args.kitti_root.expanduser().resolve()
    out = args.out.expanduser().resolve()

    if not urban_root.is_dir():
        raise FileNotFoundError(f"UrbanOmniView root not found: {urban_root}")
    if not kitti_root.is_dir() and not args.audit_only:
        raise FileNotFoundError(f"KITTI root not found: {kitti_root}")

    patterns, order_report = inspect_urban_corner_order(urban_root)
    print("===== UrbanOmniView corner-order audit =====")
    print(json.dumps(order_report, ensure_ascii=False, indent=2))
    print("Inferred corner order (index -> local sign):")
    for i, p in enumerate(patterns):
        print(f"  {i}: {p}")

    if args.audit_only:
        return 0

    if out.exists():
        raise FileExistsError(
            f"Output directory already exists: {out}\n"
            "Choose a new --out path; the script will not overwrite existing data."
        )
    out.mkdir(parents=True, exist_ok=False)
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = Counter()
    preview_counts = Counter()
    manifest_rows: list[dict[str, str]] = []
    per_source: dict[str, dict[str, Any]] = {
        "urbanomniview": {"images": Counter(), "instances": Counter()},
        "kitti": {"images": Counter(), "instances": Counter()},
    }

    # ---------------- UrbanOmniView UE5 JSON ----------------
    urban_splits = split_urban_pairs(
        urban_root, args.seed, args.val_ratio, args.test_ratio
    )
    print("\n===== UrbanOmniView split sizes (custom deterministic split) =====")
    for split, pairs in urban_splits.items():
        print(f"{split}: {len(pairs)}")
        if args.limit_per_split > 0:
            urban_splits[split] = pairs[:args.limit_per_split]

    for split, pairs in urban_splits.items():
        for image_path, json_path in pairs:
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                width, height = read_image_size(image_path)
            except Exception:
                stats["urban_image_or_json_read_error"] += 1
                continue
            camera = data.get("camera", {})
            # Prefer the actual image dimensions; report camera/image mismatches.
            if camera.get("width") and camera.get("height"):
                if int(camera["width"]) != width or int(camera["height"]) != height:
                    stats["urban_camera_image_size_mismatch"] += 1
            instances = urban_instances(data, width, height, stats)
            write_one_sample(
                image_path, "uov", split, instances, out,
                args.copy_images,
                0 if args.no_preview else args.max_preview_per_split,
                preview_counts, manifest_rows,
            )
            per_source["urbanomniview"]["images"][split] += 1
            per_source["urbanomniview"]["instances"][split] += len(instances)

    # ---------------- KITTI 3D labels + P2 ----------------
    kitti_splits = split_kitti_pairs(kitti_root)
    print("\n===== KITTI ImageSets split sizes =====")
    for split, rows in kitti_splits.items():
        print(f"{split}: {len(rows)}")
        if args.limit_per_split > 0:
            kitti_splits[split] = rows[:args.limit_per_split]

    for split, rows in kitti_splits.items():
        for image_path, label_path, calib_path in rows:
            try:
                width, height = read_image_size(image_path)
                P2 = read_p2(calib_path)
                instances = kitti_instances(
                    label_path, P2, width, height, patterns, stats,
                    kitti_width_sign=args.kitti_width_sign,
                )
            except Exception:
                stats["kitti_image_label_or_calib_error"] += 1
                continue
            write_one_sample(
                image_path, "kitti", split, instances, out,
                args.copy_images,
                0 if args.no_preview else args.max_preview_per_split,
                preview_counts, manifest_rows,
            )
            per_source["kitti"]["images"][split] += 1
            per_source["kitti"]["instances"][split] += len(instances)

    # Save the corner-order audit as a standalone artifact too.
    (out / "corner_order_audit.json").write_text(
        json.dumps(order_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Dataset YAML for Ultralytics YOLO Pose.
    yaml_text = (
        f"path: {out}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "nc: 3\n"
        "names: ['car', 'person', 'bike']\n"
        "kpt_shape: [8, 2]\n"
    )
    (out / "data.yaml").write_text(yaml_text, encoding="utf-8")

    # Manifest.
    if manifest_rows:
        import csv
        with (out / "split_manifest.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["source", "split", "image", "label", "instances"]
            )
            writer.writeheader()
            writer.writerows(manifest_rows)

    summary = {
        "task": "UrbanOmniDetect V1 / YOLO11-Pose 8 projected cuboid corners",
        "label_format": "YOLO Pose: cls cx cy w h x1 y1 ... x8 y8 (21 columns)",
        "kpt_shape": [8, 2],
        "class_names": CLASS_NAMES,
        "urbanomniview_root": str(urban_root),
        "kitti_root": str(kitti_root),
        "output_root": str(out),
        "image_staging": "copy" if args.copy_images else "symlink",
        "urban_split": {
            "method": "custom deterministic shuffle; not claimed to be the official split",
            "seed": args.seed,
            "train_fraction": round(1.0 - args.val_ratio - args.test_ratio, 4),
            "val_fraction": args.val_ratio,
            "test_fraction": args.test_ratio,
        },
        "kitti_split": "existing ImageSets/train.txt and ImageSets/val.txt",
        "urban_corner_order_audit": order_report,
        "kitti_width_sign": args.kitti_width_sign,
        "kitti_class_mapping": {
            "Car": "car", "Van": "car", "Truck": "car",
            "Pedestrian": "person", "Cyclist": "bike",
            "Person_sitting/Tram/Misc/DontCare": "ignored",
        },
        "instance_filter": (
            "An instance is skipped if any of its 8 projected keypoints lies "
            "outside the image; V1 kpt_shape=[8,2] has no visibility mask."
        ),
        "per_source": {
            source: {
                "images": dict(values["images"]),
                "instances": dict(values["instances"]),
            }
            for source, values in per_source.items()
        },
        "skip_and_diagnostic_counters": dict(stats),
        "preview_counts": {
            f"{source}/{split}": count
            for (source, split), count in preview_counts.items()
        },
        "smoke_test_limit_per_split": args.limit_per_split,
    }
    (out / "conversion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n===== Conversion complete =====")
    print(f"Output: {out}")
    print(f"data.yaml: {out / 'data.yaml'}")
    print(f"Summary: {out / 'conversion_summary.json'}")
    print(f"Manifest: {out / 'split_manifest.csv'}")
    print(f"Preview: {out / 'preview'}")
    print("\nPer-source image counts:")
    for source, values in per_source.items():
        print(f"  {source}: {dict(values['images'])}")
    print("\nPer-source instance counts:")
    for source, values in per_source.items():
        print(f"  {source}: {dict(values['instances'])}")
    print("\nImportant: inspect preview cuboids and conversion_summary.json before training.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
