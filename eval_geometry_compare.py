import csv
import json
import gc
from pathlib import Path
from itertools import combinations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

ROOT = Path("/mnt/sdb/lys/yolo11")
DATA = ROOT / "datasets/YOLO11_3D_Pose_3Source"

MODELS = {
    "VPCL": ROOT / "runs/pose/train-2/weights/best.pt",
    "Baseline": ROOT / "runs/pose/train-3/weights/best.pt",
}

OUT = ROOT / "runs/geometry_compare"

CONF = 0.001
NMS_IOU = 0.7
MATCH_IOU = 0.5
IMGSZ = 640
BATCH = 8
MAX_DET = 300

CLASS_NAMES = {
    0: "car",
    1: "person",
    2: "bike",
}

EDGE_GROUPS = [
    [(0, 1), (3, 2), (4, 5), (7, 6)],
    [(1, 2), (0, 3), (5, 6), (4, 7)],
    [(0, 4), (1, 5), (2, 6), (3, 7)],
]

TRIPLES = list(combinations(range(4), 3))

AXIS_NAMES = ["length", "width", "vertical"]


def load_gt(path, width, height):
    objects = []

    if not path.exists():
        raise FileNotFoundError(path)

    for line_no, line in enumerate(
        path.read_text().splitlines(), 1
    ):
        if not line.strip():
            continue

        v = np.asarray(
            [float(x) for x in line.split()],
            dtype=np.float64
        )

        if len(v) != 21:
            raise ValueError(
                f"Invalid label: {path}, line={line_no}, "
                f"columns={len(v)}, expected=21"
            )

        cls = int(v[0])
        cx, cy, bw, bh = v[1:5]

        box = np.array([
            (cx - bw / 2) * width,
            (cy - bh / 2) * height,
            (cx + bw / 2) * width,
            (cy + bh / 2) * height,
        ])

        points = v[5:21].reshape(8, 2)
        points *= np.array([width, height])

        objects.append({
            "cls": cls,
            "box": box,
            "points": points,
        })

    return objects


def box_iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))

    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])

    wh = np.maximum(rb - lt, 0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area_a = (
        np.maximum(a[:, 2] - a[:, 0], 0)
        * np.maximum(a[:, 3] - a[:, 1], 0)
    )

    area_b = (
        np.maximum(b[:, 2] - b[:, 0], 0)
        * np.maximum(b[:, 3] - b[:, 1], 0)
    )

    union = area_a[:, None] + area_b[None, :] - inter

    return inter / np.maximum(union, 1e-9)


def match_objects(gt, pred):
    matches = []

    for cls in range(3):
        gi = [
            i for i, x in enumerate(gt)
            if x["cls"] == cls
        ]

        pi = [
            i for i, x in enumerate(pred)
            if x["cls"] == cls
        ]

        if not gi or not pi:
            continue

        gt_boxes = np.asarray([gt[i]["box"] for i in gi])
        pr_boxes = np.asarray([pred[i]["box"] for i in pi])

        ious = box_iou_matrix(gt_boxes, pr_boxes)

        rows, cols = linear_sum_assignment(-ious)

        for r, c in zip(rows, cols):
            iou = float(ious[r, c])

            if iou >= MATCH_IOU:
                matches.append((
                    gi[r],
                    pi[c],
                    iou
                ))

    return matches


def vpcl_geometry(points, area):
    """
    Reproduce the VPCL projective incidence calculation.
    All eight keypoints are assumed valid.
    """

    points = np.asarray(points, dtype=np.float64)

    if points.shape != (8, 2):
        return None

    if not np.isfinite(points).all():
        return None

    center = points.mean(axis=0)
    scale = np.sqrt(max(float(area), 1.0))

    xy = (points - center) / scale

    homogeneous = np.column_stack([
        xy,
        np.ones(8)
    ])

    axis_losses = []
    valid_count = 0

    for edges in EDGE_GROUPS:

        lines = []
        edge_valid = []

        for start, end in edges:
            line = np.cross(
                homogeneous[start],
                homogeneous[end]
            )

            norm = np.linalg.norm(line)

            if not np.isfinite(norm) or norm <= 1e-6:
                lines.append(None)
                edge_valid.append(False)
            else:
                lines.append(line / norm)
                edge_valid.append(True)

        current_losses = []

        for triple in TRIPLES:

            if not all(edge_valid[j] for j in triple):
                continue

            matrix = np.stack([
                lines[j] for j in triple
            ])

            det = np.linalg.det(matrix)

            current_losses.append(float(det ** 2))

        axis_losses.append(
            float(np.mean(current_losses))
            if current_losses else None
        )

        valid_count += len(current_losses)

    all_losses = [
        axis_losses[i]
        for i in range(3)
        if axis_losses[i] is not None
    ]

    return {
        "loss": (
            float(np.mean(all_losses))
            if all_losses else None
        ),
        "axis": axis_losses,
        "valid_count": valid_count,
        "valid_fraction": valid_count / 12.0,
    }


def load_source_map():
    result = {}

    manifest = DATA / "split_manifest.csv"

    with manifest.open(
        "r", encoding="utf-8-sig"
    ) as f:
        for row in csv.DictReader(f):
            result[Path(row["image"]).name] = row["source"]

    return result


def infer_model(name, weight, image_files, gt_cache):

    print(f"\nEvaluating geometry: {name}")

    model = YOLO(str(weight))
    matches_by_image = {}
    seen = set()

    results = model.predict(
        source=str(DATA / "images" / "test"),
        stream=True,
        imgsz=IMGSZ,
        batch=BATCH,
        device=0,
        conf=CONF,
        iou=NMS_IOU,
        max_det=MAX_DET,
        augment=False,
        verbose=False,
        save=False,
    )

    for result in results:

        image_name = Path(result.path).name
        seen.add(image_name)

        height, width = result.orig_shape

        label_path = (
            DATA / "labels" / "test"
            / f"{Path(image_name).stem}.txt"
        )

        gt = load_gt(label_path, width, height)
        gt_cache[image_name] = gt

        predictions = []

        if (
            result.boxes is not None
            and len(result.boxes) > 0
            and result.keypoints is not None
        ):

            boxes = result.boxes.xyxy.cpu().numpy()
            classes = (
                result.boxes.cls.cpu().numpy().astype(int)
            )
            confs = result.boxes.conf.cpu().numpy()
            keypoints = result.keypoints.xy.cpu().numpy()

            for j in range(len(boxes)):

                if not np.isfinite(keypoints[j]).all():
                    continue

                predictions.append({
                    "cls": int(classes[j]),
                    "box": boxes[j],
                    "points": keypoints[j],
                    "conf": float(confs[j]),
                })

        matched = match_objects(gt, predictions)

        matches_by_image[image_name] = {
            gi: {
                "points": predictions[pi]["points"],
                "iou": iou,
                "conf": predictions[pi]["conf"],
            }
            for gi, pi, iou in matched
        }

        if len(seen) % 500 == 0:
            print(f"Processed {len(seen)} images")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    if len(seen) != len(image_files):
        raise RuntimeError(
            f"{name}: processed {len(seen)} / "
            f"{len(image_files)} images"
        )

    return matches_by_image


def safe_mean(values):
    values = [
        float(x) for x in values
        if x is not None and np.isfinite(x)
    ]

    return float(np.mean(values)) if values else None


def summarize(rows):

    output = {"count": len(rows)}

    fields = [
        "vpcl_geom",
        "baseline_geom",
        "gt_geom",
        "vpcl_nme",
        "baseline_nme",
        "vpcl_px_error",
        "baseline_px_error",
        "vpcl_pck005",
        "baseline_pck005",
        "vpcl_valid_fraction",
        "baseline_valid_fraction",
    ]

    for field in fields:

        values = [
            r[field] for r in rows
            if r[field] is not None
        ]

        output[field] = {
            "mean": safe_mean(values),
            "median": (
                float(np.median(values))
                if values else None
            ),
            "p90": (
                float(np.percentile(values, 90))
                if values else None
            ),
        }

    vp = output["vpcl_geom"]["mean"]
    bl = output["baseline_geom"]["mean"]

    if vp is not None and bl is not None and bl > 0:
        output["geometry_change_percent"] = (
            (vp - bl) / bl * 100
        )

    for axis in AXIS_NAMES:

        for model in ("vpcl", "baseline", "gt"):

            key = f"{model}_{axis}"

            output[key] = safe_mean([
                r[key] for r in rows
            ])

    return output


def main():

    OUT.mkdir(parents=True, exist_ok=True)

    image_dir = DATA / "images" / "test"

    image_files = sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp"
        }
    ])

    if not image_files:
        raise RuntimeError("No test images found")

    source_map = load_source_map()

    gt_cache = {}

    all_matches = {}

    for name, weight in MODELS.items():

        if not weight.exists():
            raise FileNotFoundError(weight)

        all_matches[name] = infer_model(
            name, weight, image_files, gt_cache
        )

    vpcl_matches = all_matches["VPCL"]
    baseline_matches = all_matches["Baseline"]

    vpcl_count = sum(
        len(x) for x in vpcl_matches.values()
    )

    baseline_count = sum(
        len(x) for x in baseline_matches.values()
    )

    rows = []

    for image_name, gt_objects in gt_cache.items():

        vp = vpcl_matches.get(image_name, {})
        bl = baseline_matches.get(image_name, {})

        common_ids = sorted(
            set(vp.keys()) & set(bl.keys())
        )

        source = source_map.get(image_name, "unknown")

        for gi in common_ids:

            gt = gt_objects[gi]

            vp_pred = vp[gi]
            bl_pred = bl[gi]

            area = (
                (gt["box"][2] - gt["box"][0])
                * (gt["box"][3] - gt["box"][1])
            )

            if area <= 1:
                continue

            vp_geom = vpcl_geometry(
                vp_pred["points"], area
            )

            bl_geom = vpcl_geometry(
                bl_pred["points"], area
            )

            gt_geom = vpcl_geometry(
                gt["points"], area
            )

            if (
                vp_geom is None
                or bl_geom is None
                or gt_geom is None
                or vp_geom["loss"] is None
                or bl_geom["loss"] is None
                or gt_geom["loss"] is None
            ):
                continue

            scale = np.sqrt(area)

            vp_error = np.linalg.norm(
                vp_pred["points"] - gt["points"],
                axis=1
            )

            bl_error = np.linalg.norm(
                bl_pred["points"] - gt["points"],
                axis=1
            )

            cls_name = CLASS_NAMES[gt["cls"]]

            row = {
                "image": image_name,
                "source": source,
                "class": cls_name,
                "gt_index": gi,

                "vpcl_iou": vp_pred["iou"],
                "baseline_iou": bl_pred["iou"],

                "vpcl_conf": vp_pred["conf"],
                "baseline_conf": bl_pred["conf"],

                "vpcl_geom": vp_geom["loss"],
                "baseline_geom": bl_geom["loss"],
                "gt_geom": gt_geom["loss"],

                "vpcl_nme": float(
                    np.mean(vp_error) / scale
                ),
                "baseline_nme": float(
                    np.mean(bl_error) / scale
                ),

                "vpcl_px_error": float(
                    np.mean(vp_error)
                ),
                "baseline_px_error": float(
                    np.mean(bl_error)
                ),

                "vpcl_pck005": float(
                    np.mean(vp_error <= 0.05 * scale)
                ),
                "baseline_pck005": float(
                    np.mean(bl_error <= 0.05 * scale)
                ),

                "vpcl_valid_fraction": vp_geom["valid_fraction"],
                "baseline_valid_fraction": bl_geom["valid_fraction"],
            }

            for i, axis in enumerate(AXIS_NAMES):

                row[f"vpcl_{axis}"] = vp_geom["axis"][i]
                row[f"baseline_{axis}"] = bl_geom["axis"][i]
                row[f"gt_{axis}"] = gt_geom["axis"][i]

            rows.append(row)

    if not rows:
        raise RuntimeError(
            "No common matched objects with valid geometry"
        )

    # Overall
    summary = {
        "images": len(image_files),
        "vpcl_matched_objects": vpcl_count,
        "baseline_matched_objects": baseline_count,
        "common_valid_objects": len(rows),
        "match_iou_threshold": MATCH_IOU,
        "confidence_threshold": CONF,
        "nms_iou": NMS_IOU,
        "overall": summarize(rows),
    }

    # By class
    summary["by_class"] = {}

    for cls in CLASS_NAMES.values():

        subset = [
            r for r in rows if r["class"] == cls
        ]

        if subset:
            summary["by_class"][cls] = summarize(subset)

    # By source
    summary["by_source"] = {}

    for source in sorted(set(r["source"] for r in rows)):

        subset = [
            r for r in rows if r["source"] == source
        ]

        summary["by_source"][source] = summarize(subset)

    with (OUT / "geometry_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            summary, f,
            ensure_ascii=False,
            indent=4
        )

    with (OUT / "geometry_per_instance.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\n========== GEOMETRY COMPARISON ==========")
    print("Images:", len(image_files))
    print("VPCL matched:", vpcl_count)
    print("Baseline matched:", baseline_count)
    print("Common valid objects:", len(rows))

    print("\nOverall:")
    print(json.dumps(
        summary["overall"],
        ensure_ascii=False,
        indent=2
    ))

    print("\nBy source:")
    print(json.dumps(
        summary["by_source"],
        ensure_ascii=False,
        indent=2
    ))

    print("\nResults saved to:", OUT)


if __name__ == "__main__":
    main()
