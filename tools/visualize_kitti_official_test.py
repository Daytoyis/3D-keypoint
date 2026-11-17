#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO11-Pose inference and 3D-corner visualization for KITTI official test set.

- Reads testing/image_2 and testing/calib.
- Draws 8 image-plane keypoints and 12 cuboid edges; no 2D rectangle.
- Saves per-detection pixel keypoints, calibration records, CSV and summary.
- KITTI official test labels are hidden: this script does inference/visualization,
  NOT local mAP evaluation.
- Assumes this project's label order: 0..3 ground corners, 4..7 top corners.
"""

import argparse
import csv
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

DEFAULT_WEIGHTS = "/mnt/sdb/lys/yolo11/runs/pose/train/weights/best.pt"
DEFAULT_IMAGE_DIR = "/mnt/sdb/lys/MonoCoP/data/KITTIDataset/testing/image_2"
DEFAULT_CALIB_DIR = "/mnt/sdb/lys/MonoCoP/data/KITTIDataset/testing/calib"
DEFAULT_OUTPUT = "/mnt/sdb/lys/yolo11/runs/pose/kitti_official_test"

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Must match the keypoint order used to create your training labels.
GROUND = [0, 1, 2, 3]
TOP = [4, 5, 6, 7]
GROUND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]
TOP_EDGES = [(4, 5), (5, 6), (6, 7), (7, 4)]
VERTICAL_EDGES = [(0, 4), (1, 5), (2, 6), (3, 7)]

# OpenCV BGR colors. Class IDs follow your data.yaml: 0 Car, 1 Person, 2 Bicycle.
COLORS = {
    0: (255, 156, 64),
    1: (64, 168, 255),
    2: (232, 99, 181),
}
FALLBACK_COLOR = (180, 210, 120)


def get_name(names, cls_id):
    if isinstance(names, dict):
        return str(names.get(int(cls_id), cls_id))
    if isinstance(names, (list, tuple)) and 0 <= int(cls_id) < len(names):
        return str(names[int(cls_id)])
    return str(cls_id)


def read_calibration(path):
    """Read KITTI calib TXT into {field: [float, ...]}."""
    result = {}
    if not path.is_file():
        return result
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            try:
                result[key.strip()] = [float(v) for v in value.split()]
            except ValueError:
                continue
    return result


def draw_edge(image, p1, p2, color, thickness=2):
    a = tuple(np.rint(p1).astype(int))
    b = tuple(np.rint(p2).astype(int))
    cv2.line(image, a, b, (10, 10, 10), thickness + 2, cv2.LINE_AA)
    cv2.line(image, a, b, color, thickness, cv2.LINE_AA)


def fill_polygon(image, points, color, alpha=0.12):
    overlay = image.copy()
    cv2.fillPoly(
        overlay, [np.rint(points).astype(np.int32)],
        color, lineType=cv2.LINE_AA
    )
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0, image)


def draw_cuboid(image, keypoints, color, show_indices=False, fill=True):
    points = np.asarray(keypoints, dtype=np.float64)
    if points.shape != (8, 2) or not np.isfinite(points).all():
        return False

    # Reorder explicitly: ground 0..3, top 4..7.
    points = points[GROUND + TOP]

    if fill:
        fill_polygon(image, points[:4], color)

    for a, b in GROUND_EDGES:
        draw_edge(image, points[a], points[b], color, 3)
    for a, b in TOP_EDGES + VERTICAL_EDGES:
        draw_edge(image, points[a], points[b], color, 2)

    for idx, (x, y) in enumerate(np.rint(points).astype(int)):
        cv2.circle(image, (x, y), 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, (x, y), 4, color, -1, cv2.LINE_AA)
        if show_indices:
            original_idx = GROUND[idx] if idx < 4 else TOP[idx - 4]
            pos = (x + 6, y - 6)
            cv2.putText(image, str(original_idx), pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, str(original_idx), pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (255, 255, 255), 1, cv2.LINE_AA)
    return True


def draw_label(image, points, text, color):
    x = max(4, int(np.nanmin(points[:, 0])))
    y = max(22, int(np.nanmin(points[:, 1])) - 8)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, color, 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--calib-dir", default=DEFAULT_CALIB_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-det", type=int, default=500)
    parser.add_argument("--limit", type=int, default=0,
                        help="Process first N images; 0 means all.")
    parser.add_argument("--show-indices", action="store_true")
    parser.add_argument("--no-fill", action="store_true")
    parser.add_argument("--no-save-images", action="store_true")
    parser.add_argument("--copy-calib", action="store_true")
    args = parser.parse_args()

    weights = Path(args.weights)
    image_dir = Path(args.image_dir)
    calib_dir = Path(args.calib_dir)
    output = Path(args.output)

    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not calib_dir.is_dir():
        raise FileNotFoundError(f"Calibration directory not found: {calib_dir}")

    images = sorted(p for p in image_dir.rglob("*")
                    if p.is_file() and p.suffix.lower() in EXTS)
    if args.limit > 0:
        images = images[:args.limit]
    if not images:
        raise RuntimeError(f"No images found in {image_dir}")

    vis_dir = output / "visualizations"
    calib_copy_dir = output / "calib"
    output.mkdir(parents=True, exist_ok=True)
    if not args.no_save_images:
        vis_dir.mkdir(parents=True, exist_ok=True)
    if args.copy_calib:
        calib_copy_dir.mkdir(parents=True, exist_ok=True)

    pred_path = output / "predictions.jsonl"
    calib_jsonl_path = output / "calibration.jsonl"
    csv_path = output / "per_image_detections.csv"
    summary_path = output / "summary.json"

    print("=" * 68)
    print("KITTI official test inference + 3D corner visualization")
    print(f"Weights : {weights}")
    print(f"Images  : {image_dir}")
    print(f"Calib   : {calib_dir}")
    print(f"Output  : {output}")
    print(f"Images  : {len(images)}")
    print("No local mAP: official KITTI testing labels are not available.")
    print("=" * 68)

    model = YOLO(str(weights))
    print("Model classes:", model.names)

    totals = Counter()
    image_presence = Counter()
    processed = 0
    missing_calib = 0
    invalid_pose = 0
    start = time.time()

    with pred_path.open("w", encoding="utf-8") as pred_f, \
         calib_jsonl_path.open("w", encoding="utf-8") as calib_f, \
         csv_path.open("w", newline="", encoding="utf-8") as csv_f:

        writer = csv.DictWriter(
            csv_f,
            fieldnames=["image", "detections", "car", "person",
                        "bicycle", "calib_found"]
        )
        writer.writeheader()

        for idx, image_path in enumerate(images, 1):
            stem = image_path.stem
            calib_path = calib_dir / f"{stem}.txt"
            calib = read_calibration(calib_path)
            calib_found = bool(calib)
            if not calib_found:
                missing_calib += 1

            calib_f.write(json.dumps({
                "image": image_path.name,
                "calib_file": str(calib_path),
                "calibration": calib,
            }, ensure_ascii=False) + "\n")

            if args.copy_calib and calib_path.is_file():
                shutil.copy2(calib_path, calib_copy_dir / calib_path.name)

            results = model.predict(
                source=str(image_path),
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                max_det=args.max_det,
                verbose=False,
            )
            if not results:
                writer.writerow({
                    "image": image_path.name, "detections": 0,
                    "car": 0, "person": 0, "bicycle": 0,
                    "calib_found": calib_found,
                })
                processed += 1
                continue

            result = results[0]
            canvas = result.orig_img.copy()
            if result.boxes is None or len(result.boxes) == 0:
                cls_ids = np.empty((0,), dtype=int)
                confs = np.empty((0,), dtype=float)
            else:
                cls_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
                confs = result.boxes.conf.detach().cpu().numpy()

            if result.keypoints is None:
                kpts = np.empty((0, 8, 2), dtype=float)
            else:
                kpts = result.keypoints.xy.detach().cpu().numpy()

            if len(cls_ids) != len(kpts):
                raise RuntimeError(
                    f"{image_path.name}: {len(cls_ids)} boxes but "
                    f"{len(kpts)} keypoint sets."
                )

            image_counts = Counter()
            image_classes_present = set()
            records = []

            for det_i, (cls_id, conf, points) in enumerate(zip(cls_ids, confs, kpts)):
                cls_id = int(cls_id)
                conf = float(conf)
                points = np.asarray(points, dtype=np.float64)

                if points.shape != (8, 2) or not np.isfinite(points).all():
                    invalid_pose += 1
                    continue

                name = get_name(result.names, cls_id)
                color = COLORS.get(cls_id, FALLBACK_COLOR)

                if not draw_cuboid(
                    canvas, points, color,
                    show_indices=args.show_indices,
                    fill=not args.no_fill,
                ):
                    invalid_pose += 1
                    continue

                draw_label(canvas, points, f"{name} {conf:.2f}", color)

                image_counts[cls_id] += 1
                totals[cls_id] += 1
                image_classes_present.add(cls_id)

                records.append({
                    "image": image_path.name,
                    "detection_index": det_i,
                    "class_id": cls_id,
                    "class_name": name,
                    "confidence": conf,
                    "keypoints_pixel_xy": np.round(points, 3).tolist(),
                    "calib_file": str(calib_path),
                    "calibration_available": calib_found,
                })

            for cls_id in image_classes_present:
                image_presence[cls_id] += 1

            for rec in records:
                pred_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if not args.no_save_images:
                save_path = vis_dir / f"{stem}.jpg"
                if not cv2.imwrite(str(save_path), canvas):
                    print(f"[WARN] Could not save {save_path}")

            writer.writerow({
                "image": image_path.name,
                "detections": len(records),
                "car": image_counts[0],
                "person": image_counts[1],
                "bicycle": image_counts[2],
                "calib_found": calib_found,
            })

            processed += 1
            if idx == 1 or idx % 100 == 0 or idx == len(images):
                elapsed = time.time() - start
                rate = processed / elapsed if elapsed else 0
                eta_min = ((len(images) - processed) / rate / 60) if rate else 0
                print(
                    f"[{idx}/{len(images)}] {image_path.name} | "
                    f"Detections={len(records)} | Total={sum(totals.values())} | "
                    f"{rate:.2f} img/s | ETA={eta_min:.1f} min"
                )

    elapsed = time.time() - start
    summary = {
        "weights": str(weights),
        "image_dir": str(image_dir),
        "calib_dir": str(calib_dir),
        "images_found": len(images),
        "images_processed": processed,
        "total_detections": int(sum(totals.values())),
        "class_detection_counts": {
            get_name(model.names, c): int(totals[c]) for c in sorted(totals)
        },
        "class_image_presence_counts": {
            get_name(model.names, c): int(image_presence[c])
            for c in sorted(image_presence)
        },
        "missing_calibration_files": missing_calib,
        "invalid_pose_results": invalid_pose,
        "confidence_threshold": args.conf,
        "imgsz": args.imgsz,
        "elapsed_seconds": round(elapsed, 2),
        "average_seconds_per_image": round(elapsed / processed, 4) if processed else None,
        "evaluation_note": (
            "KITTI official test labels are hidden; this run is inference and "
            "visualization only, not local mAP evaluation."
        ),
        "assumed_keypoint_order": {
            "ground": GROUND,
            "top": TOP,
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 68)
    print("FINISHED")
    print(f"Processed: {processed}/{len(images)}")
    print(f"Total detections: {sum(totals.values())}")
    print(f"Car: {totals[0]}")
    print(f"Person: {totals[1]}")
    print(f"Bicycle: {totals[2]}")
    print(f"Missing calib: {missing_calib}")
    print(f"Invalid pose: {invalid_pose}")
    print(f"Elapsed: {elapsed / 60:.2f} min")
    print(f"Visualizations: {vis_dir}")
    print(f"Predictions: {pred_path}")
    print(f"Calibration records: {calib_jsonl_path}")
    print(f"Per-image CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print("=" * 68)


if __name__ == "__main__":
    main()
