#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run full-image YOLO11-Pose inference on the standalone KITTI 3D-Pose dataset.

Outputs:
  predicted_2d_keypoints_all.json
  predicted_2d_keypoints_{train,val,test}.json
  summary.json
  per_image_counts.csv
  visualizations/{split}/...

Coordinates:
  - bbox_xyxy_px: pixel coordinates in the original image
  - bbox_xywh_norm: normalized center-x, center-y, width, height
  - keypoints_px: 8 predicted 2D keypoints in original-image pixels
  - keypoints_norm: 8 predicted 2D keypoints normalized by image width/height

The script only performs 2D inference. It does not read KITTI 3D labels or
geometry_3d_manifest.jsonl and does not perform 3D reconstruction.
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


DEFAULT_WEIGHTS = (
    "/mnt/sdb/lys/yolo11/runs/pose/train-3/weights/best.pt"
)
DEFAULT_DATASET = (
    "/mnt/sdb/lys/yolo11/datasets/KITTI_3D_Pose_Standalone_v4"
)
DEFAULT_OUT = (
    "/mnt/sdb/lys/yolo11/runs/kitti_full_2d_predictions"
)
SPLITS = ("train", "val", "test")


def parse_names(names):
    if isinstance(names, dict):
        return [str(names[k] if k in names else names[str(k)])
                for k in range(len(names))]
    if isinstance(names, (list, tuple)):
        return [str(x) for x in names]
    raise ValueError(f"Unsupported names format in data.yaml: {type(names)}")


def canonical_class_name(name):
    n = str(name).strip().lower()
    aliases = {
        "bike": "bicycle",
        "cyclist": "bicycle",
        "bicyclist": "bicycle",
        "pedestrian": "person",
    }
    return aliases.get(n, n)


def to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))


def draw_prediction(image, detections):
    """Draw predicted boxes and numbered keypoints; no GT is drawn."""
    canvas = image.copy()
    colors = {
        0: (40, 220, 40),    # car
        1: (0, 165, 255),    # person
        2: (255, 180, 0),    # bicycle
    }

    for det in detections:
        cid = int(det["class_id"])
        color = colors.get(cid, (0, 255, 255))
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox_xyxy_px"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        label = f'{det["class_name"]} {det["confidence"]:.2f}'
        ty = max(18, y1 - 6)
        cv2.putText(canvas, label, (max(0, x1), ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                    cv2.LINE_AA)

        for kp_id, (x, y) in enumerate(det["keypoints_px"]):
            px, py = int(round(x)), int(round(y))
            if 0 <= px < canvas.shape[1] and 0 <= py < canvas.shape[0]:
                cv2.circle(canvas, (px, py), 4, color, -1, cv2.LINE_AA)
                cv2.putText(canvas, str(kp_id), (px + 4, py - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1,
                            cv2.LINE_AA)
    return canvas


def get_image_files(split_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(
        [p for p in split_dir.iterdir()
         if p.is_file() and p.suffix.lower() in exts],
        key=lambda p: p.name
    )


def build_detection_records(result, model_names, expected_kpts=8):
    h, w = result.orig_shape
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return []

    xyxy = to_numpy(boxes.xyxy).astype(float)
    cls = to_numpy(boxes.cls).astype(int)
    conf = to_numpy(boxes.conf).astype(float)

    if result.keypoints is None:
        raise RuntimeError(
            f"Pose output is missing for image {result.path}; "
            "the selected weights may not be a pose model."
        )

    kpts = to_numpy(result.keypoints.xy)
    if kpts is None:
        raise RuntimeError(f"Keypoint coordinates are missing: {result.path}")
    kpts = np.asarray(kpts, dtype=float)

    if kpts.ndim != 3 or kpts.shape[1] != expected_kpts or kpts.shape[2] != 2:
        raise RuntimeError(
            f"Expected keypoints shape (N,{expected_kpts},2), got "
            f"{kpts.shape} for {result.path}. Check the model/data keypoint schema."
        )

    kpt_conf_obj = getattr(result.keypoints, "conf", None)
    kpt_conf = to_numpy(kpt_conf_obj) if kpt_conf_obj is not None else None

    records = []
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
        cid = int(cls[i])
        cname = str(model_names.get(cid, f"class_{cid}"))

        points_px = []
        points_norm = []
        for x, y in kpts[i]:
            points_px.append([float(x), float(y)])
            points_norm.append([
                float(x / max(w, 1)),
                float(y / max(h, 1))
            ])

        kp_conf_list = None
        if kpt_conf is not None:
            kp_conf_list = [float(v) for v in kpt_conf[i]]

        bw = x2 - x1
        bh = y2 - y1
        records.append({
            "class_id": cid,
            "class_name": cname,
            "confidence": float(conf[i]),
            "bbox_xyxy_px": [x1, y1, x2, y2],
            "bbox_xywh_norm": [
                float(((x1 + x2) / 2.0) / max(w, 1)),
                float(((y1 + y2) / 2.0) / max(h, 1)),
                float(bw / max(w, 1)),
                float(bh / max(h, 1)),
            ],
            "keypoints_px": points_px,
            "keypoints_norm": points_norm,
            "keypoint_confidence": kp_conf_list,
        })
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Full KITTI 2D keypoint inference using a trained YOLO Pose model."
    )
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--imgsz", type=int, default=640)
    # Low threshold retains candidates for later threshold tuning and evaluation.
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--save-vis-per-split", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--splits", nargs="+", choices=SPLITS, default=list(SPLITS),
        help="Default: infer train, val and test."
    )
    args = parser.parse_args()

    weights = Path(args.weights).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    yaml_path = dataset_root / "data.yaml"

    if not weights.is_file():
        raise FileNotFoundError(f"Missing model weights: {weights}")
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Missing dataset YAML: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    dataset_names = parse_names(data_cfg.get("names"))
    kpt_shape = data_cfg.get("kpt_shape")
    if not kpt_shape or int(kpt_shape[0]) != 8 or int(kpt_shape[1]) != 2:
        raise ValueError(
            f"Expected data.yaml kpt_shape [8, 2], got {kpt_shape}"
        )

    print("========== MODEL ==========")
    print("Weights:", weights)
    model = YOLO(str(weights))
    raw_model_names = model.names
    if isinstance(raw_model_names, dict):
        model_names = {int(k): str(v) for k, v in raw_model_names.items()}
    else:
        model_names = {i: str(v) for i, v in enumerate(raw_model_names)}

    print("Model classes:", model_names)
    print("Dataset classes:", dataset_names)

    if len(model_names) != len(dataset_names):
        raise ValueError(
            f"Class count mismatch: model={len(model_names)}, "
            f"dataset={len(dataset_names)}"
        )

    for cid, dataset_name in enumerate(dataset_names):
        model_name = model_names.get(cid)
        if model_name is None:
            raise ValueError(f"Model has no class id {cid}")
        if canonical_class_name(model_name) != canonical_class_name(dataset_name):
            raise ValueError(
                f"Class order/name mismatch at id {cid}: "
                f"model='{model_name}', dataset='{dataset_name}'. "
                "Do not run inference until the mapping is checked."
            )

    print("Class ID mapping: PASS")
    print("Keypoint schema: 8 x 2")
    print("Inference confidence threshold:", args.conf)
    print("Note: conf=0.001 intentionally retains low-confidence candidates.")

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    combined = {
        "metadata": {
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "weights": str(weights),
            "dataset_root": str(dataset_root),
            "data_yaml": str(yaml_path),
            "model_names": model_names,
            "dataset_names": dataset_names,
            "kpt_shape": [8, 2],
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "device": args.device,
            "batch_size": args.batch_size,
            "splits": args.splits,
            "note": (
                "Predictions only. No KITTI 3D labels or geometry sidecar "
                "were used as inference input. Train predictions are in-sample "
                "because the model was trained using the matching 3Source split."
            ),
        },
        "splits": {}
    }

    summary = {
        "weights": str(weights),
        "dataset_root": str(dataset_root),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "device": args.device,
        "class_names": model_names,
        "kpt_shape": [8, 2],
        "splits": {},
        "total_images": 0,
        "total_predictions": 0,
        "total_empty_images": 0,
        "total_by_class": {str(i): 0 for i in model_names},
    }

    csv_path = out_dir / "per_image_counts.csv"
    csv_rows = []

    for split in args.splits:
        split_dir = dataset_root / "images" / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split image directory: {split_dir}")

        image_files = get_image_files(split_dir)
        if not image_files:
            raise RuntimeError(f"No images found in {split_dir}")

        print(f"\n========== INFERENCE: {split.upper()} ==========")
        print("Images:", len(image_files))

        split_records = []
        split_counts = Counter()
        empty_images = 0
        vis_candidates = set(
            rng.sample(
                range(len(image_files)),
                min(args.save_vis_per_split, len(image_files))
            )
        ) if args.save_vis_per_split > 0 else set()

        # IMPORTANT:
        # Pass the DIRECTORY, not a Python list containing thousands of file paths.
        # Ultralytics treats a list of paths as an in-memory image collection in
        # some versions, which can create one enormous preprocessing batch and
        # cause CUDA OOM even when batch=args.batch_size is specified.
        # Passing the directory activates the normal batched image dataloader.
        result_stream = model.predict(
            source=str(split_dir),
            stream=True,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            batch=args.batch_size,
            verbose=False,
            save=False,
        )

        # Basenames are unique within each split and robust to symlink/path
        # normalization differences returned by the Ultralytics dataloader.
        name_to_index = {p.name: i for i, p in enumerate(image_files)}
        processed = 0
        seen_names = set()

        for result in result_stream:
            result_name = Path(result.path).name
            idx = name_to_index.get(result_name)
            if idx is None:
                raise RuntimeError(
                    f"Cannot map prediction to source image: {result.path}"
                )
            if result_name in seen_names:
                raise RuntimeError(
                    f"Duplicate inference result for image: {result_name}"
                )
            seen_names.add(result_name)

            h, w = result.orig_shape
            detections = build_detection_records(result, model_names, expected_kpts=8)
            if not detections:
                empty_images += 1

            for det in detections:
                split_counts[str(det["class_id"])] += 1

            record = {
                "image_id": image_files[idx].stem,
                "image_file": image_files[idx].name,
                "image_relative_path": str(
                    image_files[idx].relative_to(dataset_root)
                ),
                "width": int(w),
                "height": int(h),
                "num_detections": len(detections),
                "detections": detections,
            }
            split_records.append(record)

            csv_rows.append({
                "split": split,
                "image_id": record["image_id"],
                "image_file": record["image_file"],
                "width": w,
                "height": h,
                "num_detections": len(detections),
                "car": sum(d["class_id"] == 0 for d in detections),
                "person": sum(d["class_id"] == 1 for d in detections),
                "bike": sum(d["class_id"] == 2 for d in detections),
            })

            if idx in vis_candidates:
                vis_dir = out_dir / "visualizations" / split
                vis_dir.mkdir(parents=True, exist_ok=True)
                vis = draw_prediction(result.orig_img, detections)
                vis_path = vis_dir / image_files[idx].name
                if not cv2.imwrite(str(vis_path), vis):
                    print(f"WARNING: failed to write visualization: {vis_path}")

            processed += 1
            if processed % 500 == 0 or processed == len(image_files):
                print(
                    f"[{split}] {processed}/{len(image_files)} "
                    f"images; predictions={sum(split_counts.values())}; "
                    f"empty_images={empty_images}"
                )

        if processed != len(image_files):
            raise RuntimeError(
                f"Incomplete split inference for {split}: "
                f"processed {processed}/{len(image_files)} images"
            )

        # Preserve original image order in the JSON.
        split_records.sort(key=lambda x: x["image_file"])
        split_payload = {
            "split": split,
            "image_count": len(split_records),
            "prediction_count": int(sum(split_counts.values())),
            "empty_images": int(empty_images),
            "class_counts": dict(split_counts),
            "images": split_records,
        }
        combined["splits"][split] = split_payload
        write_json(
            out_dir / f"predicted_2d_keypoints_{split}.json",
            {
                "metadata": combined["metadata"],
                "split": split_payload,
            }
        )

        summary["splits"][split] = {
            "images": len(split_records),
            "predictions": int(sum(split_counts.values())),
            "empty_images": int(empty_images),
            "class_counts": dict(split_counts),
        }
        summary["total_images"] += len(split_records)
        summary["total_predictions"] += int(sum(split_counts.values()))
        summary["total_empty_images"] += int(empty_images)
        for cid in model_names:
            summary["total_by_class"][str(cid)] += int(split_counts[str(cid)])

    write_json(out_dir / "predicted_2d_keypoints_all.json", combined)
    write_json(out_dir / "summary.json", summary)

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "split", "image_id", "image_file", "width", "height",
            "num_detections", "car", "person", "bike"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n========== FULL INFERENCE COMPLETE ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Output directory:", out_dir)
    print("Combined JSON:", out_dir / "predicted_2d_keypoints_all.json")
    print("Summary:", out_dir / "summary.json")
    print("Per-image CSV:", csv_path)
    print("Visualizations:", out_dir / "visualizations")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
