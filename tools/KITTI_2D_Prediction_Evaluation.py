#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate full KITTI YOLO11-Pose predictions against standalone-v4 2D labels.

Ground-truth label convention (unchanged from v4):
  class cx cy w h x0 y0 ... x7 y7
  21 values per object; bbox and keypoints normalized to image width/height.
  Keypoints 0..3 are bottom corners; 4..7 are corresponding top corners.

Prediction input:
  predicted_2d_keypoints_{train,val,test}.json
written by kitti_full_2d_inference.py.

Evaluation:
  - class-aware greedy one-to-one matching by bbox IoU, confidence descending
  - Precision / Recall / F1 at configurable confidence thresholds
  - AP50 and AP50:95
  - keypoint pixel error and PCK@5/10/20 px on bbox-matched true positives
  - per-keypoint-index error analysis

This script does NOT use KITTI 3D labels or geometry_3d_manifest.jsonl.
"""

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np


DEFAULT_DATASET = (
    "/mnt/sdb/lys/yolo11/datasets/KITTI_3D_Pose_Standalone_v4"
)
DEFAULT_PRED_DIR = (
    "/mnt/sdb/lys/yolo11/runs/kitti_full_2d_predictions"
)
DEFAULT_OUT = (
    "/mnt/sdb/lys/yolo11/runs/kitti_2d_prediction_evaluation"
)
SPLITS = ("train", "val", "test")
CLASS_NAMES = {0: "car", 1: "person", 2: "bike"}
DEFAULT_CONF_THRESHOLDS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.25, 0.50]
DEFAULT_IOU_THRESHOLDS = [round(x, 2) for x in np.arange(0.50, 0.951, 0.05)]


def iter_json_array(path, array_key="images", chunk_size=1024 * 1024):
    """
    Stream objects from a large JSON array without loading the entire
    prediction JSON into RAM. Supports the compact JSON layout produced by
    kitti_full_2d_inference.py.
    """
    marker = '"' + array_key + '"'
    decoder = json.JSONDecoder()

    with Path(path).open("r", encoding="utf-8") as f:
        buf = ""
        while marker not in buf:
            chunk = f.read(chunk_size)
            if not chunk:
                raise ValueError(f'JSON array key "{array_key}" not found in {path}')
            buf += chunk

        pos = buf.index(marker) + len(marker)

        # Find the array opening bracket after the key.
        while True:
            colon = buf.find(":", pos)
            if colon >= 0:
                pos = colon + 1
                break
            chunk = f.read(chunk_size)
            if not chunk:
                raise ValueError(f"Malformed JSON near key {array_key}: {path}")
            buf += chunk

        while True:
            while pos < len(buf) and buf[pos].isspace():
                pos += 1
            if pos < len(buf):
                if buf[pos] != "[":
                    raise ValueError(
                        f'Expected "[" after "{array_key}" in {path}'
                    )
                pos += 1
                break
            chunk = f.read(chunk_size)
            if not chunk:
                raise ValueError(f"Unexpected EOF before {array_key} array")
            buf = buf[pos:] + chunk
            pos = 0

        while True:
            # Skip whitespace and commas between array elements.
            while True:
                while pos < len(buf) and (buf[pos].isspace() or buf[pos] == ","):
                    pos += 1
                if pos < len(buf):
                    break
                chunk = f.read(chunk_size)
                if not chunk:
                    raise ValueError(f"Unexpected EOF inside {array_key} array")
                buf = buf[pos:] + chunk
                pos = 0

            if buf[pos] == "]":
                return

            try:
                item, end = decoder.raw_decode(buf, pos)
            except json.JSONDecodeError:
                chunk = f.read(chunk_size)
                if not chunk:
                    raise ValueError(
                        f"Could not decode an item in {array_key} array of {path}"
                    )
                buf = buf[pos:] + chunk
                pos = 0
                continue

            yield item
            pos = end

            # Discard consumed data so memory remains bounded.
            if pos >= chunk_size:
                buf = buf[pos:]
                pos = 0


def read_gt_labels(label_path, width, height):
    """Read v4 21-column YOLO-Pose labels and convert coordinates to pixels."""
    objects = []
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing GT label file: {label_path}")

    with label_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            values = [float(x) for x in line.split()]
            if len(values) != 21:
                raise ValueError(
                    f"{label_path}:{line_no}: expected 21 columns, got {len(values)}"
                )

            cid = int(round(values[0]))
            if cid not in CLASS_NAMES:
                raise ValueError(
                    f"{label_path}:{line_no}: unexpected class id {cid}"
                )

            cx, cy, bw, bh = values[1:5]
            x1 = (cx - bw / 2.0) * width
            y1 = (cy - bh / 2.0) * height
            x2 = (cx + bw / 2.0) * width
            y2 = (cy + bh / 2.0) * height

            kpts = np.asarray(values[5:], dtype=np.float32).reshape(8, 2)
            kpts[:, 0] *= width
            kpts[:, 1] *= height

            objects.append({
                "class_id": cid,
                "bbox": np.asarray([x1, y1, x2, y2], dtype=np.float32),
                "keypoints": kpts,
            })
    return objects


def bbox_iou_matrix(pred_boxes, gt_boxes):
    """Pairwise IoU matrix for xyxy boxes."""
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)

    p = np.asarray(pred_boxes, dtype=np.float32)
    g = np.asarray(gt_boxes, dtype=np.float32)

    ix1 = np.maximum(p[:, None, 0], g[None, :, 0])
    iy1 = np.maximum(p[:, None, 1], g[None, :, 1])
    ix2 = np.minimum(p[:, None, 2], g[None, :, 2])
    iy2 = np.minimum(p[:, None, 3], g[None, :, 3])

    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih

    pa = np.maximum(0.0, p[:, 2] - p[:, 0]) * np.maximum(
        0.0, p[:, 3] - p[:, 1]
    )
    ga = np.maximum(0.0, g[:, 2] - g[:, 0]) * np.maximum(
        0.0, g[:, 3] - g[:, 1]
    )
    union = pa[:, None] + ga[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def greedy_match(iou_matrix, threshold):
    """
    Confidence-order greedy matching for one image and one class.
    Rows must already be sorted by descending prediction confidence.
    Returns a bool TP vector and assigned GT index (-1 for unmatched).
    """
    n_pred, n_gt = iou_matrix.shape
    tp = np.zeros(n_pred, dtype=bool)
    assignment = np.full(n_pred, -1, dtype=np.int32)
    if n_pred == 0 or n_gt == 0:
        return tp, assignment

    used = np.zeros(n_gt, dtype=bool)
    for pi in range(n_pred):
        overlaps = iou_matrix[pi].copy()
        overlaps[used] = -1.0
        gi = int(np.argmax(overlaps))
        if overlaps[gi] >= threshold:
            tp[pi] = True
            assignment[pi] = gi
            used[gi] = True
    return tp, assignment


def ap_101(conf, tp, gt_count):
    """101-point interpolated AP from confidence-ranked predictions."""
    if gt_count <= 0:
        return None
    if len(conf) == 0:
        return 0.0

    order = np.argsort(-conf, kind="mergesort")
    tp_sorted = tp[order].astype(np.float64)
    fp_sorted = 1.0 - tp_sorted

    tp_cum = np.cumsum(tp_sorted)
    fp_cum = np.cumsum(fp_sorted)
    recall = tp_cum / max(float(gt_count), 1.0)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    # Precision envelope.
    precision_envelope = np.maximum.accumulate(precision[::-1])[::-1]
    samples = []
    for r in np.linspace(0.0, 1.0, 101):
        valid = precision_envelope[recall >= r]
        samples.append(float(valid.max()) if valid.size else 0.0)
    return float(np.mean(samples))


def safe_div(a, b):
    return float(a / b) if b else 0.0


def summarize_errors(error_matrix, normalized_matrix):
    """
    error_matrix shape [matched_objects, 8], each value is a keypoint pixel
    Euclidean error. normalized_matrix is the same error divided by GT bbox
    diagonal. NaN rows are excluded.
    """
    valid = np.isfinite(error_matrix)
    vals = error_matrix[valid]
    nvals = normalized_matrix[np.isfinite(normalized_matrix)]

    if vals.size == 0:
        return {
            "evaluated_objects": 0,
            "evaluated_keypoints": 0,
            "mpjpe_px": None,
            "median_px": None,
            "p90_px": None,
            "nme_bbox_diagonal": None,
            "pck_5px": None,
            "pck_10px": None,
            "pck_20px": None,
        }

    return {
        "evaluated_objects": int(error_matrix.shape[0]),
        "evaluated_keypoints": int(vals.size),
        "mpjpe_px": float(np.mean(vals)),
        "median_px": float(np.median(vals)),
        "p90_px": float(np.percentile(vals, 90)),
        "nme_bbox_diagonal": float(np.mean(nvals)) if nvals.size else None,
        "pck_5px": float(np.mean(vals <= 5.0)),
        "pck_10px": float(np.mean(vals <= 10.0)),
        "pck_20px": float(np.mean(vals <= 20.0)),
    }


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_split(split, dataset_root, pred_dir, out_dir,
                   conf_thresholds, iou_thresholds, progress_every=500):
    label_dir = dataset_root / "labels" / split
    pred_json = pred_dir / f"predicted_2d_keypoints_{split}.json"
    if not pred_json.is_file():
        raise FileNotFoundError(f"Missing prediction JSON: {pred_json}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Missing label directory: {label_dir}")

    iou_thresholds = sorted(set(float(x) for x in iou_thresholds))
    if not iou_thresholds or any(x <= 0 or x > 1 for x in iou_thresholds):
        raise ValueError(f"Invalid IoU thresholds: {iou_thresholds}")
    if not any(abs(x - 0.5) < 1e-8 for x in iou_thresholds):
        raise ValueError("iou-thresholds must include 0.50 for keypoint matching")
    iou_key = min(iou_thresholds, key=lambda x: abs(x - 0.5))

    # Store compact NumPy chunks rather than retaining full prediction dicts.
    accum = {}
    for cid in CLASS_NAMES:
        accum[cid] = {
            "conf": [],
            "tp": {thr: [] for thr in iou_thresholds},
            "kp_error": [],
            "kp_nme": [],
            "gt_count": 0,
        }

    seen = set()
    image_count = 0
    total_gt = Counter()
    total_pred = Counter()

    print(f"\n========== EVALUATE {split.upper()} ==========")
    print("Prediction JSON:", pred_json)
    print("GT labels:", label_dir)

    for image_record in iter_json_array(pred_json, "images"):
        image_file = image_record.get("image_file")
        if not image_file:
            raise ValueError(f"Missing image_file in prediction record: {image_record.keys()}")
        if image_file in seen:
            raise ValueError(f"Duplicate prediction image record: {image_file}")
        seen.add(image_file)

        width = int(image_record["width"])
        height = int(image_record["height"])
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions for {image_file}: {width}x{height}")

        label_path = label_dir / (Path(image_file).stem + ".txt")
        gt_objects = read_gt_labels(label_path, width, height)
        detections = image_record.get("detections", [])

        gt_by_class = {
            cid: [g for g in gt_objects if g["class_id"] == cid]
            for cid in CLASS_NAMES
        }
        pred_by_class = {cid: [] for cid in CLASS_NAMES}

        for det in detections:
            cid = int(det["class_id"])
            if cid not in CLASS_NAMES:
                raise ValueError(f"Unexpected predicted class id {cid} in {image_file}")
            box = np.asarray(det["bbox_xyxy_px"], dtype=np.float32)
            kp = np.asarray(det["keypoints_px"], dtype=np.float32)
            if box.shape != (4,):
                raise ValueError(f"Bad bbox shape in {image_file}: {box.shape}")
            if kp.shape != (8, 2):
                raise ValueError(f"Expected 8x2 predicted keypoints in {image_file}, got {kp.shape}")
            pred_by_class[cid].append({
                "confidence": float(det["confidence"]),
                "bbox": box,
                "keypoints": kp,
            })

        for cid in CLASS_NAMES:
            gts = gt_by_class[cid]
            preds = sorted(
                pred_by_class[cid],
                key=lambda d: d["confidence"],
                reverse=True
            )
            a = accum[cid]
            a["gt_count"] += len(gts)
            total_gt[cid] += len(gts)
            total_pred[cid] += len(preds)

            n_pred = len(preds)
            conf = np.asarray(
                [p["confidence"] for p in preds], dtype=np.float32
            )
            pred_boxes = (
                np.stack([p["bbox"] for p in preds])
                if n_pred else np.zeros((0, 4), dtype=np.float32)
            )
            gt_boxes = (
                np.stack([g["bbox"] for g in gts])
                if gts else np.zeros((0, 4), dtype=np.float32)
            )
            ious = bbox_iou_matrix(pred_boxes, gt_boxes)

            tp_by_iou = {}
            assignment_by_iou = {}
            for thr in iou_thresholds:
                tp_vec, assignment = greedy_match(ious, thr)
                tp_by_iou[thr] = tp_vec
                assignment_by_iou[thr] = assignment
                a["tp"][thr].append(tp_vec)

            # Use IoU=0.50 bbox matches for keypoint evaluation.
            assignment = assignment_by_iou[iou_key]
            err = np.full((n_pred, 8), np.nan, dtype=np.float32)
            nme = np.full((n_pred, 8), np.nan, dtype=np.float32)
            for pi, gi in enumerate(assignment):
                if gi < 0:
                    continue
                pred_kp = preds[pi]["keypoints"]
                gt_kp = gts[int(gi)]["keypoints"]
                point_error = np.linalg.norm(pred_kp - gt_kp, axis=1)
                gt_box = gts[int(gi)]["bbox"]
                diag = float(np.linalg.norm(gt_box[2:4] - gt_box[0:2]))
                err[pi] = point_error
                nme[pi] = point_error / max(diag, 1e-6)

            a["conf"].append(conf)
            a["kp_error"].append(err)
            a["kp_nme"].append(nme)

        image_count += 1
        if image_count % progress_every == 0:
            print(f"[{split}] evaluated {image_count} images")

    if image_count == 0:
        raise RuntimeError(f"No image records found in {pred_json}")

    # Validate complete coverage against the split's image directory.
    image_dir = dataset_root / "images" / split
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    dataset_images = {
        p.name for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in valid_exts
    }
    missing = sorted(dataset_images - seen)
    extra = sorted(seen - dataset_images)
    if missing or extra:
        raise RuntimeError(
            f"{split} image coverage mismatch: prediction_records={len(seen)}, "
            f"dataset_images={len(dataset_images)}, missing={len(missing)}, "
            f"extra={len(extra)}; missing examples={missing[:5]}, extra examples={extra[:5]}"
        )

    # Concatenate compact arrays per class.
    final = {}
    for cid in CLASS_NAMES:
        a = accum[cid]
        conf = np.concatenate(a["conf"]) if a["conf"] else np.zeros(0, dtype=np.float32)
        tp = {
            thr: np.concatenate(a["tp"][thr]) if a["tp"][thr]
            else np.zeros(0, dtype=bool)
            for thr in iou_thresholds
        }
        kp_error = (
            np.concatenate(a["kp_error"], axis=0)
            if a["kp_error"] else np.zeros((0, 8), dtype=np.float32)
        )
        kp_nme = (
            np.concatenate(a["kp_nme"], axis=0)
            if a["kp_nme"] else np.zeros((0, 8), dtype=np.float32)
        )
        final[cid] = {
            "conf": conf,
            "tp": tp,
            "kp_error": kp_error,
            "kp_nme": kp_nme,
            "gt_count": int(a["gt_count"]),
            "pred_count": int(len(conf)),
        }

    # AP metrics.
    ap_rows = []
    ap_by_class = {}
    for cid, name in CLASS_NAMES.items():
        c = final[cid]
        per_iou = {}
        for thr in iou_thresholds:
            ap = ap_101(c["conf"], c["tp"][thr], c["gt_count"])
            per_iou[f"{thr:.2f}"] = ap
            ap_rows.append({
                "split": split,
                "class_id": cid,
                "class_name": name,
                "iou_threshold": f"{thr:.2f}",
                "ground_truth_objects": c["gt_count"],
                "predictions": c["pred_count"],
                "AP_101": ap,
            })
        vals = [v for v in per_iou.values() if v is not None]
        ap_by_class[name] = {
            "AP50": per_iou.get("0.50"),
            "AP50_95": float(np.mean(vals)) if vals else None,
            "AP_by_IoU": per_iou,
        }

    # Confidence-threshold detection and keypoint metrics.
    detection_rows = []
    keypoint_rows = []
    keypoint_index_rows = []
    best_f1 = {}

    for cid, name in CLASS_NAMES.items():
        c = final[cid]
        conf = c["conf"]
        tp50 = c["tp"][iou_key]
        gt_count = c["gt_count"]
        best_row = None

        for conf_thr in conf_thresholds:
            keep = conf >= conf_thr
            n_pred = int(keep.sum())
            n_tp = int(tp50[keep].sum())
            n_fp = n_pred - n_tp
            n_fn = gt_count - n_tp
            precision = safe_div(n_tp, n_tp + n_fp)
            recall = safe_div(n_tp, n_tp + n_fn)
            f1 = safe_div(2 * precision * recall, precision + recall)

            detection_rows.append({
                "split": split,
                "class_id": cid,
                "class_name": name,
                "confidence_threshold": conf_thr,
                "GT": gt_count,
                "predictions_kept": n_pred,
                "TP": n_tp,
                "FP": n_fp,
                "FN": n_fn,
                "precision": precision,
                "recall": recall,
                "F1": f1,
            })

            if best_row is None or (f1, conf_thr) > (
                best_row["F1"], best_row["confidence_threshold"]
            ):
                best_row = {
                    "confidence_threshold": conf_thr,
                    "F1": f1,
                    "precision": precision,
                    "recall": recall,
                }

            selected_err = c["kp_error"][keep & tp50]
            selected_nme = c["kp_nme"][keep & tp50]
            kp_summary = summarize_errors(selected_err, selected_nme)
            keypoint_rows.append({
                "split": split,
                "class_id": cid,
                "class_name": name,
                "confidence_threshold": conf_thr,
                **kp_summary,
            })

            for kpt_id in range(8):
                vals = selected_err[:, kpt_id]
                vals = vals[np.isfinite(vals)]
                keypoint_index_rows.append({
                    "split": split,
                    "class_id": cid,
                    "class_name": name,
                    "confidence_threshold": conf_thr,
                    "keypoint_id": kpt_id,
                    "mean_error_px": float(np.mean(vals)) if vals.size else None,
                    "median_error_px": float(np.median(vals)) if vals.size else None,
                    "p90_error_px": float(np.percentile(vals, 90)) if vals.size else None,
                })

        best_f1[name] = best_row

    split_summary = {
        "split": split,
        "images_evaluated": image_count,
        "image_coverage": {
            "prediction_images": len(seen),
            "dataset_images": len(dataset_images),
            "missing": 0,
            "extra": 0,
        },
        "classes": {},
    }
    for cid, name in CLASS_NAMES.items():
        c = final[cid]
        split_summary["classes"][name] = {
            "ground_truth_objects": c["gt_count"],
            "predictions": c["pred_count"],
            "AP": ap_by_class[name],
            "best_F1_threshold_within_sweep": (
                best_f1[name] if split == "val" else None
            ),
        }

    return {
        "summary": split_summary,
        "ap_rows": ap_rows,
        "detection_rows": detection_rows,
        "keypoint_rows": keypoint_rows,
        "keypoint_index_rows": keypoint_index_rows,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate KITTI full 2D keypoint predictions against v4 YOLO-Pose labels."
    )
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET)
    parser.add_argument("--pred-dir", default=DEFAULT_PRED_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument(
        "--conf-thresholds", nargs="+", type=float,
        default=DEFAULT_CONF_THRESHOLDS
    )
    parser.add_argument(
        "--iou-thresholds", nargs="+", type=float,
        default=DEFAULT_IOU_THRESHOLDS
    )
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    pred_dir = Path(args.pred_dir).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in (dataset_root / "data.yaml", pred_dir):
        if not p.exists():
            raise FileNotFoundError(p)

    conf_thresholds = sorted(set(float(x) for x in args.conf_thresholds))
    if not conf_thresholds or any(x < 0 or x > 1 for x in conf_thresholds):
        raise ValueError(f"Invalid confidence thresholds: {conf_thresholds}")

    all_results = {}
    all_ap, all_det, all_kp, all_kp_idx = [], [], [], []

    print("========== KITTI 2D PREDICTION EVALUATION ==========")
    print("Dataset:", dataset_root)
    print("Prediction directory:", pred_dir)
    print("Output:", out_dir)
    print("Classes:", CLASS_NAMES)
    print("GT labels: 21-column v4 YOLO-Pose labels")
    print("Confidence thresholds:", conf_thresholds)
    print("IoU thresholds:", args.iou_thresholds)
    print("Matching: class-aware, one-to-one, confidence-descending")
    print("Keypoint matching: bbox IoU >= 0.50")
    print("No 3D GT is used.")

    for split in args.splits:
        result = evaluate_split(
            split=split,
            dataset_root=dataset_root,
            pred_dir=pred_dir,
            out_dir=out_dir,
            conf_thresholds=conf_thresholds,
            iou_thresholds=args.iou_thresholds,
            progress_every=args.progress_every,
        )
        all_results[split] = result["summary"]
        all_ap.extend(result["ap_rows"])
        all_det.extend(result["detection_rows"])
        all_kp.extend(result["keypoint_rows"])
        all_kp_idx.extend(result["keypoint_index_rows"])

    write_csv(
        out_dir / "ap_metrics.csv",
        all_ap,
        ["split", "class_id", "class_name", "iou_threshold",
         "ground_truth_objects", "predictions", "AP_101"],
    )
    write_csv(
        out_dir / "detection_metrics_by_threshold.csv",
        all_det,
        ["split", "class_id", "class_name", "confidence_threshold",
         "GT", "predictions_kept", "TP", "FP", "FN",
         "precision", "recall", "F1"],
    )
    write_csv(
        out_dir / "keypoint_metrics_by_threshold.csv",
        all_kp,
        ["split", "class_id", "class_name", "confidence_threshold",
         "evaluated_objects", "evaluated_keypoints", "mpjpe_px",
         "median_px", "p90_px", "nme_bbox_diagonal",
         "pck_5px", "pck_10px", "pck_20px"],
    )
    write_csv(
        out_dir / "keypoint_index_metrics.csv",
        all_kp_idx,
        ["split", "class_id", "class_name", "confidence_threshold",
         "keypoint_id", "mean_error_px", "median_error_px", "p90_error_px"],
    )

    summary = {
        "dataset_root": str(dataset_root),
        "prediction_dir": str(pred_dir),
        "evaluation_output": str(out_dir),
        "class_mapping": CLASS_NAMES,
        "ground_truth_format": (
            "21 columns: class cx cy w h x0 y0 ... x7 y7; "
            "normalized coordinates; no visibility values"
        ),
        "keypoint_order": "0-3 bottom corners; 4-7 corresponding top corners",
        "matching": "class-aware greedy one-to-one bbox IoU matching, confidence descending",
        "keypoint_evaluation": "only bbox-matched true positives at IoU >= 0.50",
        "confidence_thresholds": conf_thresholds,
        "iou_thresholds": args.iou_thresholds,
        "splits": all_results,
        "interpretation": {
            "train": "In-sample diagnostic; model was trained using the matching 3Source train images.",
            "val": "Use for confidence-threshold selection and method development; it may have influenced best.pt selection.",
            "test": "Hold out for final independent evaluation; do not tune thresholds or methods on this split.",
            "AP": "Computed from predictions retained by inference conf=0.001 and max_det=300; AP is therefore conditional on that inference candidate set.",
            "keypoint_error": "Pixel errors are measured only after detection boxes are matched by IoU; they do not include missed ground-truth objects.",
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    report_lines = [
        "KITTI 2D PREDICTION EVALUATION",
        f"Dataset: {dataset_root}",
        f"Predictions: {pred_dir}",
        f"Confidence thresholds: {conf_thresholds}",
        f"IoU thresholds: {args.iou_thresholds}",
        "",
    ]
    for split, s in all_results.items():
        report_lines.append(f"===== {split.upper()} =====")
        report_lines.append(f"Images evaluated: {s['images_evaluated']}")
        for name, c in s["classes"].items():
            ap = c["AP"]
            bf = c["best_F1_threshold_within_sweep"]
            if bf is None:
                threshold_text = "threshold selection: use VAL only"
            else:
                threshold_text = (
                    f"best-F1-thr(val-sweep)={bf['confidence_threshold']}, "
                    f"F1={bf['F1']}"
                )
            report_lines.append(
                f"{name}: GT={c['ground_truth_objects']}, "
                f"pred={c['predictions']}, AP50={ap['AP50']}, "
                f"AP50:95={ap['AP50_95']}, {threshold_text}"
            )
        report_lines.append("")
    report_lines.extend([
        "Caution:",
        "- Train results are in-sample diagnostics.",
        "- Select thresholds using Val, not Test.",
        "- Test is reserved for final evaluation.",
        "- Keypoint errors are conditional on successful bbox matching.",
        "- No 3D ground-truth labels are used by this evaluation.",
    ])
    (out_dir / "report.txt").write_text("\n".join(report_lines), encoding="utf-8")

    print("\n========== EVALUATION COMPLETE ==========")
    print(json.dumps(all_results, ensure_ascii=False, indent=2))
    print("\nOutput:", out_dir)
    print("Summary:", out_dir / "summary.json")
    print("AP:", out_dir / "ap_metrics.csv")
    print("Detection:", out_dir / "detection_metrics_by_threshold.csv")
    print("Keypoints:", out_dir / "keypoint_metrics_by_threshold.csv")
    print("Per-keypoint:", out_dir / "keypoint_index_metrics.csv")
    print("Report:", out_dir / "report.txt")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
