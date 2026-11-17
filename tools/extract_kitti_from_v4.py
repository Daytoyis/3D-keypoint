#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a KITTI-only YOLO11-Pose 3D-cuboid dataset using the KITTI conversion
and split logic extracted from build_unified_3d_pose_dataset_v4_chronological.py.

This is a standalone KITTI builder. It does not require UrbanOmniView/UE5 or
DAIR-V2X-I, and it does not read an already-built 3Source dataset.

v4 rules retained:
- KITTI training/image_2 + label_2 + calib only. Official testing/ is excluded
  because public 3D ground-truth labels are unavailable.
- Class mapping: Car/Van/Truck -> 0 (car); Pedestrian -> 1 (person);
  Cyclist -> 2 (bike). All unmapped source classes are ignored.
- Canonical cuboid corner order:
  0..3 bottom: (+L,+W), (+L,-W), (-L,-W), (-L,+W);
  4..7 corresponding top corners.
- Projection uses KITTI calibration P2 and the v4 camera-frame cuboid geometry.
- YOLO pose labels have 21 columns:
  cls cx cy w h x1 y1 ... x8 y8
- v4 horizontal flip indices: [1, 0, 3, 2, 5, 4, 7, 6].
- KITTI split remains v4 deterministic multi-label image-level stratification,
  with 80/10/10 target ratios and seed 42 by default. The chronological split
  in v4 applies to UrbanOmniView, not KITTI.
- Preserves v4 3D geometry sidecar, split manifest, previews and validation.
- Images are symlinked by default; --copy-images makes physical copies.
- Existing output directories are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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


# Defaults and taxonomy copied from v4.
DEFAULT_KITTI = Path("/mnt/sdb/lys/MonoCoP/data/KITTIDataset")
DEFAULT_OUT = Path(
    "/mnt/sdb/lys/yolo11/datasets/KITTI_3D_Pose_Standalone_v4"
)

CLASS_NAMES = ["car", "person", "bike"]
KITTI_CLASS_MAP = {
    "Car": 0,
    "Van": 0,
    "Truck": 0,
    "Pedestrian": 1,
    "Cyclist": 2,
}

# DAIR official canonical cuboid corner identity, exactly as in v4.
CANONICAL_PATTERNS = [
    (1, 1, -1),
    (1, -1, -1),
    (-1, -1, -1),
    (-1, 1, -1),
    (1, 1, 1),
    (1, -1, 1),
    (-1, -1, 1),
    (-1, 1, 1),
]

FLIP_IDX = [1, 0, 3, 2, 5, 4, 7, 6]
BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]
COLORS = [(40, 220, 70), (40, 170, 255), (255, 165, 35)]


# ---------------------------------------------------------------------------
# Basic image / geometry helpers retained from v4
# ---------------------------------------------------------------------------
def read_image_size(path: Path) -> tuple[int, int]:
    if cv2 is not None:
        im = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if im is None:
            raise ValueError(f"Cannot read image: {path}")
        h, w = im.shape[:2]
        return int(w), int(h)
    from PIL import Image
    with Image.open(path) as im:
        return int(im.width), int(im.height)


def inside_image(points: np.ndarray, width: int, height: int) -> bool:
    return bool(
        np.isfinite(points).all()
        and np.all(points[:, 0] >= 0)
        and np.all(points[:, 0] <= width - 1)
        and np.all(points[:, 1] >= 0)
        and np.all(points[:, 1] <= height - 1)
    )


def projected_bbox(
    points: np.ndarray, width: int, height: int
) -> tuple[float, float, float, float] | None:
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
            vals = [value[k] for k in ("xmin", "ymin", "xmax", "ymax")]
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


def clip_bbox(
    box: tuple[float, float, float, float] | None,
    width: int,
    height: int,
):
    if box is None:
        return None
    x1, y1, x2, y2 = box
    out = (
        max(0.0, x1),
        max(0.0, y1),
        min(width - 1.0, x2),
        min(height - 1.0, y2),
    )
    return (
        out
        if out[2] - out[0] >= 1.0 and out[3] - out[1] >= 1.0
        else None
    )


def make_pose_line(
    cls: int,
    points: np.ndarray,
    width: int,
    height: int,
    bbox_override: tuple[float, float, float, float] | None = None,
) -> str:
    # Same v4 rule: prefer the original KITTI 2D bbox; if unusable, derive
    # the bbox from the eight projected 3D corners.
    box = (
        clip_bbox(bbox_override, width, height)
        if bbox_override is not None
        else None
    )
    if box is None:
        box = projected_bbox(points, width, height)
    if box is None:
        raise ValueError("Degenerate 2D bounding box")

    x1, y1, x2, y2 = box
    vals = [
        cls,
        (x1 + x2) / (2 * width),
        (y1 + y2) / (2 * height),
        (x2 - x1) / width,
        (y2 - y1) / height,
    ]
    for x, y in points:
        vals.extend([float(x) / width, float(y) / height])
    return " ".join(
        [str(int(vals[0]))] + [f"{float(v):.6f}" for v in vals[1:]]
    )


def make_record(
    image_path: Path, instances: list[dict[str, Any]]
) -> dict[str, Any]:
    cc = Counter(int(x["cls"]) for x in instances)
    return {
        "source": "kitti",
        "image_path": image_path,
        "stem": image_path.stem,
        "instances": instances,
        "class_counts": cc,
        "group": f"frame:{image_path.stem}",
        "group_mode": "image",
    }


# ---------------------------------------------------------------------------
# KITTI conversion: extracted from v4 without changing the geometry logic
# ---------------------------------------------------------------------------
def read_p2(path: Path) -> np.ndarray:
    for line in path.read_text(
        encoding="utf-8", errors="ignore"
    ).splitlines():
        if line.startswith("P2:"):
            vals = np.fromstring(
                line.split(":", 1)[1], sep=" ", dtype=np.float64
            )
            if vals.size == 12:
                return vals.reshape(3, 4)
    raise ValueError(f"P2 not found in {path}")


def kitti_corners(
    h: float,
    w: float,
    length: float,
    loc: np.ndarray,
    yaw: float,
    width_sign: int = 1,
) -> np.ndarray:
    """Generate KITTI camera-frame corners in v4's DAIR semantic index order.

    KITTI location is the bottom centre. Camera y points down, so bottom y=0,
    top y=-h. The local long-axis and width-side signs follow
    CANONICAL_PATTERNS.
    """
    sx = np.asarray([p[0] for p in CANONICAL_PATTERNS], dtype=np.float64)
    sy = np.asarray([p[1] for p in CANONICAL_PATTERNS], dtype=np.float64)
    sz = np.asarray([p[2] for p in CANONICAL_PATTERNS], dtype=np.float64)

    x_local = sx * length / 2.0
    z_local = width_sign * sy * w / 2.0
    y_local = np.where(sz < 0, 0.0, -h)

    c, s = math.cos(yaw), math.sin(yaw)
    rot_y = np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )
    return (rot_y @ np.vstack([x_local, y_local, z_local])).T + loc.reshape(1, 3)


def project_camera(
    points: np.ndarray, P: np.ndarray
) -> np.ndarray | None:
    hom = (P @ np.hstack([points, np.ones((8, 1))]).T).T
    if not np.isfinite(hom).all() or np.any(hom[:, 2] <= 1e-6):
        return None
    uv = hom[:, :2] / hom[:, 2:3]
    return uv if np.isfinite(uv).all() else None


def kitti_instances(
    label_path: Path,
    calib_path: Path,
    width: int,
    height: int,
    stats: Counter,
    width_sign: int = 1,
) -> list[dict[str, Any]]:
    P2 = read_p2(calib_path)
    out = []

    for line in label_path.read_text(
        encoding="utf-8", errors="ignore"
    ).splitlines():
        f = line.split()
        if not f:
            continue
        if len(f) < 15:
            stats["kitti_malformed_line"] += 1
            continue

        typ = f[0]
        if typ not in KITTI_CLASS_MAP:
            stats[f"kitti_ignored_class:{typ}"] += 1
            continue

        try:
            h, w, length = map(float, f[8:11])
            loc = np.asarray(list(map(float, f[11:14])), dtype=np.float64)
            yaw = float(f[14])

            if (
                min(h, w, length) <= 0
                or loc.shape != (3,)
                or not np.isfinite(loc).all()
            ):
                raise ValueError("invalid 3D geometry")

            pts3d = kitti_corners(
                h, w, length, loc, yaw, width_sign=width_sign
            )
            pts2d = project_camera(pts3d, P2)

            if pts2d is None:
                stats["kitti_projection_invalid"] += 1
                continue
            if not inside_image(pts2d, width, height):
                stats["kitti_corner_outside_image"] += 1
                continue
            if projected_bbox(pts2d, width, height) is None:
                stats["kitti_degenerate_box"] += 1
                continue

            bbox = parse_bbox(f[4:8])
            stats[f"kitti_kept_source_class:{typ}"] += 1
            out.append(
                {
                    "cls": KITTI_CLASS_MAP[typ],
                    "points": pts2d,
                    "bbox": bbox,
                    "source_class": typ,
                    "geometry3d": {
                        "coordinate_frame": "rectified_camera",
                        "units": "m",
                        "corners_3d_camera": pts3d.tolist(),
                    },
                }
            )
        except Exception:
            stats["kitti_invalid_geometry_or_projection"] += 1

    return out


def find_image(image_dir: Path, stem: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        p = image_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def collect_kitti(
    root: Path, stats: Counter, width_sign: int = 1
) -> list[dict[str, Any]]:
    base = root / "training"
    image_dir = base / "image_2"
    label_dir = base / "label_2"
    calib_dir = base / "calib"

    if (
        not image_dir.is_dir()
        or not label_dir.is_dir()
        or not calib_dir.is_dir()
    ):
        raise FileNotFoundError(
            f"KITTI training image_2/label_2/calib not found under {base}"
        )

    records = []
    for label_path in sorted(label_dir.glob("*.txt")):
        stem = label_path.stem
        image_path = find_image(image_dir, stem)
        calib_path = calib_dir / f"{stem}.txt"

        if image_path is None or not calib_path.is_file():
            stats["kitti_incomplete_triplet"] += 1
            continue

        try:
            width, height = read_image_size(image_path)
            instances = kitti_instances(
                label_path,
                calib_path,
                width,
                height,
                stats,
                width_sign,
            )
            records.append(make_record(image_path, instances))
        except Exception as exc:
            stats[f"kitti_sample_error:{type(exc).__name__}"] += 1
            print(f"[WARN] KITTI {stem}: {exc}")

    return records


# ---------------------------------------------------------------------------
# v4 deterministic multi-label image-level split: unchanged
# ---------------------------------------------------------------------------
def allocate_counts(
    n: int, ratios: tuple[float, float, float]
) -> tuple[int, int, int]:
    raw = [n * r for r in ratios]
    counts = [int(math.floor(x)) for x in raw]
    left = n - sum(counts)
    order = sorted(
        range(3),
        key=lambda i: raw[i] - counts[i],
        reverse=True,
    )
    for i in order[:left]:
        counts[i] += 1
    return tuple(counts)  # type: ignore[return-value]


def _record_vector(record: dict[str, Any]) -> np.ndarray:
    cc = record["class_counts"]
    return np.asarray(
        [1.0, cc.get(0, 0), cc.get(1, 0), cc.get(2, 0)],
        dtype=np.float64,
    )


def _rarity_score(
    record: dict[str, Any], totals: np.ndarray
) -> float:
    vec = _record_vector(record)
    score = 0.0
    for c in range(3):
        if vec[c + 1] > 0:
            score += 1.0 / max(totals[c + 1], 1.0)
    return score + 1e-12 * float(vec[1:].sum())


def stratified_image_split(
    records: list[dict[str, Any]],
    ratios=(0.8, 0.1, 0.1),
    seed=42,
):
    """v4 greedy multi-label split balancing image and instance counts."""
    if not records:
        return {"train": [], "val": [], "test": []}

    rng = random.Random(seed)
    names = ["train", "val", "test"]
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
            image_deficit = (
                (targets[s, 0] - current[s, 0])
                / max(targets[s, 0], 1.0)
            )
            cls_deficits = []
            for c in range(3):
                if vec[c + 1] > 0:
                    cls_deficits.append(
                        (targets[s, c + 1] - current[s, c + 1])
                        / max(targets[s, c + 1], 1.0)
                    )
            class_deficit = (
                float(np.mean(cls_deficits))
                if cls_deficits
                else image_deficit
            )
            score = (
                0.60 * image_deficit
                + 0.40 * class_deficit
                + rng.random() * 1e-8
            )
            scored.append((score, s))

        chosen = max(scored)[1]
        out[names[chosen]].append(rec)
        current[chosen] += vec
        slots[chosen] -= 1

    for split in out:
        out[split].sort(key=lambda r: r["image_path"].name)
    return out


# ---------------------------------------------------------------------------
# v4 output helpers
# ---------------------------------------------------------------------------
def stage_image(src: Path, dst: Path, copy_images: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(f"Output collision: {dst}")
    if copy_images:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def draw_preview(
    image_path: Path,
    instances: list[dict[str, Any]],
    out_path: Path,
):
    if cv2 is None:
        return False
    im = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if im is None:
        return False

    for inst in instances:
        pts = np.round(np.asarray(inst["points"])).astype(int)
        color = COLORS[int(inst["cls"])]
        for a, b in BOX_EDGES:
            cv2.line(
                im, tuple(pts[a]), tuple(pts[b]),
                color, 2, cv2.LINE_AA
            )
        for i, (x, y) in enumerate(pts):
            cv2.circle(
                im, (int(x), int(y)), 4, color, -1, cv2.LINE_AA
            )
            cv2.putText(
                im, str(i), (int(x) + 3, int(y) - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA
            )
        x0, y0 = pts.min(axis=0)
        cv2.putText(
            im,
            CLASS_NAMES[int(inst["cls"])],
            (int(x0), max(15, int(y0) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), im))


def split_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    cc = Counter()
    image_class = Counter()
    for r in records:
        cc.update(r["class_counts"])
        for c in r["class_counts"]:
            image_class[c] += 1

    return {
        "images": len(records),
        "instances": int(sum(cc.values())),
        "classes_instances": {
            CLASS_NAMES[i]: int(cc[i]) for i in range(3)
        },
        "images_containing_class": {
            CLASS_NAMES[i]: int(image_class[i]) for i in range(3)
        },
        "images_with_targets": int(
            sum(bool(r["instances"]) for r in records)
        ),
        "empty_label_images": int(
            sum(not r["instances"] for r in records)
        ),
    }


def validate_output(out: Path) -> dict[str, int]:
    bad = []
    lines_total = 0
    label_files = sorted((out / "labels").glob("*/*.txt"))

    for label in label_files:
        for line_no, line in enumerate(
            label.read_text(encoding="utf-8").splitlines(), 1
        ):
            f = line.split()
            lines_total += 1
            if len(f) != 21:
                bad.append((str(label), line_no, f"columns={len(f)}"))
                continue
            try:
                cls = int(f[0])
                vals = np.asarray(f[1:], dtype=np.float64)
            except Exception:
                bad.append((str(label), line_no, "non-numeric"))
                continue
            if cls not in (0, 1, 2) or not np.isfinite(vals).all():
                bad.append((str(label), line_no, "class/nonfinite"))
                continue
            if np.any(vals < -1e-6) or np.any(vals > 1.000001):
                bad.append((str(label), line_no, "normalized value out of range"))
            if vals[2] <= 0 or vals[3] <= 0:
                bad.append((str(label), line_no, "non-positive bbox size"))

    if bad:
        raise RuntimeError(
            "Output label validation failed: " + repr(bad[:20])
        )
    return {
        "label_files": len(label_files),
        "label_lines": lines_total,
    }


# ---------------------------------------------------------------------------
# Main: KITTI-only build
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Build a standalone KITTI-only YOLO11-Pose 3D-cuboid dataset "
            "using the conversion rules from v4."
        )
    )
    ap.add_argument("--kitti-root", type=Path, default=DEFAULT_KITTI)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--kitti-width-sign",
        type=int,
        choices=(-1, 1),
        default=1,
        help="KITTI local width-side sign; v4 default is +1.",
    )
    ap.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of creating symlinks.",
    )
    ap.add_argument(
        "--preview-per-split",
        type=int,
        default=8,
        help="Number of labelled preview images saved per split; 0 disables previews.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Convert, split and report counts without writing output files.",
    )
    args = ap.parse_args()

    kitti_root = args.kitti_root.expanduser().resolve()
    out = args.out.expanduser().resolve()

    if not kitti_root.is_dir():
        raise FileNotFoundError(f"KITTI source root does not exist: {kitti_root}")
    if out.exists() and not args.dry_run:
        raise FileExistsError(
            f"Output already exists: {out}. Choose a new --out; "
            "no files were overwritten."
        )
    if args.preview_per_split < 0:
        raise ValueError("--preview-per-split must be >= 0")

    print("========== STEP 1: CONVERT KITTI ==========")
    stats = Counter()
    kitti = collect_kitti(
        kitti_root, stats, width_sign=args.kitti_width_sign
    )
    print(f"KITTI usable labelled images: {len(kitti)}")
    if stats:
        print("\n========== CONVERSION FILTER / ERROR COUNTS ==========")
        for key in sorted(stats):
            print(f"  {key}: {stats[key]}")
    else:
        print("No conversion filters/errors recorded.")

    if not kitti:
        raise RuntimeError(
            "KITTI produced zero usable records. Check root/layout/calibration."
        )

    print("\n========== STEP 2: KITTI SPLIT (80/10/10 TARGET) ==========")
    kitti_split = stratified_image_split(
        kitti, ratios=(0.8, 0.1, 0.1), seed=args.seed
    )
    source_splits = {"kitti": kitti_split}

    print("KITTI split mode: v4 deterministic multi-label image-level split")
    print(f"Random seed: {args.seed}")
    print("Target ratio: train=80%, val=10%, test=10%")
    print("Official KITTI testing/ is excluded (no public 3D ground-truth labels).")

    print("\n========== SPLIT COUNTS ==========")
    for split in ("train", "val", "test"):
        info = split_summary(kitti_split[split])
        print(
            f"{split:5s}: images={info['images']:5d}, "
            f"instances={info['instances']:7d}, "
            f"classes={info['classes_instances']}, "
            f"empty_labels={info['empty_label_images']}"
        )

    summary = {
        "task": "KITTI-only / YOLO11-Pose 3D cuboid keypoints",
        "source_script": "build_unified_3d_pose_dataset_v4_chronological.py",
        "classes": {str(i): CLASS_NAMES[i] for i in range(3)},
        "class_definitions": {
            "0": "Car: passenger cars, trucks and vans",
            "1": "Person: pedestrians",
            "2": "Bike: cyclists",
        },
        "class_mapping": {
            "car": "Car/Truck/Van",
            "person": "Pedestrian",
            "bike": "Cyclist",
        },
        "excluded_source_classes": [
            "DontCare",
            "Misc",
            "Tram",
            "Person_sitting",
            "all other unmapped classes",
        ],
        "keypoint_order": (
            "DAIR official: 0..3 bottom corners "
            "(+L,+W), (+L,-W), (-L,-W), (-L,+W); "
            "4..7 corresponding top corners"
        ),
        "label_format": "21 columns: cls cx cy w h x1 y1 ... x8 y8",
        "geometry_3d_sidecar": {
            "file": "geometry_3d_manifest.jsonl",
            "purpose": (
                "preserve the eight KITTI camera-frame 3D corners in "
                "canonical order for future 3D-space training"
            ),
            "coordinate_frame": "rectified_camera",
            "units": "m",
        },
        "kpt_shape": [8, 2],
        "flip_idx": FLIP_IDX,
        "seed": args.seed,
        "split_policy": {
            "target_ratio": {
                "train": 0.8,
                "val": 0.1,
                "test": 0.1,
            },
            "KITTI": (
                "labelled training/ only; deterministic multi-label "
                "image-level split; official testing/ excluded because "
                "public 3D ground-truth labels are unavailable"
            ),
            "note": (
                "The chronological split in v4 is for UrbanOmniView; "
                "KITTI retains the v4 deterministic multi-label image split."
            ),
        },
        "source_counts": {
            "kitti": {
                split: split_summary(kitti_split[split])
                for split in ("train", "val", "test")
            }
        },
        "conversion_stats": dict(stats),
        "kitti_width_sign": args.kitti_width_sign,
        "image_staging": "copy" if args.copy_images else "symlink",
        "original_source_files_modified": False,
    }

    if args.dry_run:
        print("\nDRY RUN complete: no output files were created.")
        print("Review conversion filters and split counts before the full build.")
        return 0

    print("\n========== STEP 3: WRITE KITTI-ONLY DATASET ==========")
    out.mkdir(parents=True, exist_ok=False)
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest = []
    geometry_manifest = []
    preview_count = Counter()
    output_names = set()

    for split in ("train", "val", "test"):
        for rec in kitti_split[split]:
            src = rec["image_path"]
            stem = f"kitti_{src.stem}"
            if stem in output_names:
                digest = hashlib.sha1(
                    str(src.resolve()).encode("utf-8")
                ).hexdigest()[:8]
                stem = f"{stem}_{digest}"
            output_names.add(stem)

            dst_img = out / "images" / split / f"{stem}{src.suffix.lower()}"
            dst_lab = out / "labels" / split / f"{stem}.txt"

            stage_image(src, dst_img, args.copy_images)
            width, height = read_image_size(src)

            lines = [
                make_pose_line(
                    int(inst["cls"]),
                    np.asarray(inst["points"]),
                    width,
                    height,
                    inst.get("bbox"),
                )
                for inst in rec["instances"]
            ]
            dst_lab.write_text(
                "\n".join(lines) + ("\n" if lines else ""),
                encoding="utf-8",
            )

            if (
                preview_count[split] < args.preview_per_split
                and rec["instances"]
            ):
                draw_preview(
                    src,
                    rec["instances"],
                    out / "preview" / "kitti" / split / f"{stem}.jpg",
                )
                preview_count[split] += 1

            source_class_counts = Counter(
                str(inst.get("source_class", "<unknown>"))
                for inst in rec["instances"]
            )
            manifest.append(
                {
                    "source": "kitti",
                    "split": split,
                    "group": rec["group"],
                    "group_mode": rec["group_mode"],
                    "frame_index": "",
                    "engine_frame": "",
                    "timestamp_utc": "",
                    "image": str(dst_img),
                    "label": str(dst_lab),
                    "instances": len(rec["instances"]),
                    "car": rec["class_counts"].get(0, 0),
                    "person": rec["class_counts"].get(1, 0),
                    "bike": rec["class_counts"].get(2, 0),
                    "source_class_counts": json.dumps(
                        dict(source_class_counts),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                }
            )

            geometry_objects = []
            for inst in rec["instances"]:
                geom = inst.get("geometry3d")
                if geom is not None:
                    geometry_objects.append(
                        {
                            "class_id": int(inst["cls"]),
                            "class_name": CLASS_NAMES[int(inst["cls"])],
                            "source_class": inst.get(
                                "source_class", "<unknown>"
                            ),
                            "corners_2d_image": np.asarray(
                                inst["points"]
                            ).tolist(),
                            **geom,
                        }
                    )

            geometry_manifest.append(
                {
                    "source": "kitti",
                    "split": split,
                    "frame_index": None,
                    "engine_frame": None,
                    "timestamp_utc": None,
                    "image": str(dst_img),
                    "label_2d": str(dst_lab),
                    "objects": geometry_objects,
                }
            )

    yaml_text = (
        f"path: {out}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "nc: 3\n"
        "names: ['car', 'person', 'bike']\n"
        "kpt_shape: [8, 2]\n"
        f"flip_idx: {FLIP_IDX}\n"
    )
    (out / "data.yaml").write_text(yaml_text, encoding="utf-8")

    with (out / "split_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source",
                "split",
                "group",
                "group_mode",
                "frame_index",
                "engine_frame",
                "timestamp_utc",
                "image",
                "label",
                "instances",
                "car",
                "person",
                "bike",
                "source_class_counts",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest)

    with (out / "geometry_3d_manifest.jsonl").open(
        "w", encoding="utf-8"
    ) as f:
        for row in geometry_manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    validation = validate_output(out)
    summary["output_root"] = str(out)
    summary["output_validation"] = validation
    summary["manifest_rows"] = len(manifest)
    summary["geometry_manifest_rows"] = len(geometry_manifest)
    summary["geometry_manifest_objects"] = int(
        sum(len(row["objects"]) for row in geometry_manifest)
    )
    (out / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n========== BUILD COMPLETE ==========")
    print(f"Output: {out}")
    print(f"Data YAML: {out / 'data.yaml'}")
    print(f"Manifest: {out / 'split_manifest.csv'}")
    print(f"3D geometry sidecar: {out / 'geometry_3d_manifest.jsonl'}")
    print(f"Summary: {out / 'dataset_summary.json'}")
    print(f"Preview: {out / 'preview'}")
    print(
        f"Validated label files: {validation['label_files']}; "
        f"instances: {validation['label_lines']}"
    )
    print("Original KITTI source files were not modified.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
