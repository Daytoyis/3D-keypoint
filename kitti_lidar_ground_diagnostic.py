#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KITTI diagnostic: compare fixed horizontal ground Y=1.65 m against
per-frame LiDAR-RANSAC road-plane estimation, using the SAME saved YOLO11-Pose
predicted 2D keypoints for both reconstructions.

Important:
- LiDAR is used ONLY to estimate the road plane for this diagnostic.
- The 3D object labels are accessed only AFTER reconstruction for evaluation.
- The plane-aware cuboid remains upright in KITTI camera coordinates (same
  cuboid convention as KITTI labels); the fitted plane constrains its bottom
  centre. This keeps 3D IoU comparable with the current baseline.
- This is a diagnostic upper-bound experiment for ground-plane information,
  NOT the final monocular inference method.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import least_squares

import kitti_yolo11_predicted_3d as base


def parse_calibration(path: Path):
    vals = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        arr = np.fromstring(value, sep=" ", dtype=float)
        if arr.size:
            vals[key.strip()] = arr
    if "R0_rect" not in vals or "Tr_velo_to_cam" not in vals:
        raise ValueError(f"Missing R0_rect or Tr_velo_to_cam in {path}")
    R0 = vals["R0_rect"].reshape(3, 3)
    Tr = vals["Tr_velo_to_cam"].reshape(3, 4)
    T = R0 @ Tr
    P2 = vals["P2"].reshape(3, 4)
    return P2, T


def load_velodyne(path: Path):
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4:
        raise ValueError(f"Unexpected KITTI velodyne binary length: {path}")
    return raw.reshape(-1, 4)[:, :3]


def to_camera(points_velo: np.ndarray, T_velo_to_rect: np.ndarray):
    hom = np.column_stack([points_velo, np.ones(len(points_velo))])
    return hom @ T_velo_to_rect.T


def select_ground_candidates(points_cam: np.ndarray,
                             x_range=18.0,
                             z_min=3.0,
                             z_max=55.0,
                             y_min=0.5,
                             y_max=3.2,
                             cell_x=0.45,
                             cell_z=0.9):
    """
    In rectified KITTI camera coordinates: X right, Y down, Z forward.
    For each X-Z grid cell retain its lowest visible return (largest Y),
    which is a practical ground-candidate heuristic before constrained RANSAC.
    """
    p = np.asarray(points_cam, dtype=float)
    m = (
        np.isfinite(p).all(axis=1)
        & (np.abs(p[:, 0]) <= x_range)
        & (p[:, 2] >= z_min) & (p[:, 2] <= z_max)
        & (p[:, 1] >= y_min) & (p[:, 1] <= y_max)
    )
    p = p[m]
    if len(p) < 30:
        return p

    ix = np.floor((p[:, 0] + x_range) / cell_x).astype(np.int64)
    iz = np.floor(p[:, 2] / cell_z).astype(np.int64)
    # Sort by grid cell then Y descending; keep the largest Y per cell.
    order = np.lexsort((-p[:, 1], iz, ix))
    q = p[order]
    qix, qiz = ix[order], iz[order]
    first = np.ones(len(q), dtype=bool)
    first[1:] = (qix[1:] != qix[:-1]) | (qiz[1:] != qiz[:-1])
    return q[first]


def fit_ground_plane_ransac(points_cam: np.ndarray, seed: int = 0,
                            iterations: int = 2500,
                            distance_threshold: float = 0.13,
                            max_tilt_deg: float = 15.0,
                            min_inliers: int = 35):
    """
    Fit n.X + d = 0. n is oriented so n_y > 0.
    Constrain the normal to be within max_tilt_deg of camera +Y.
    """
    pts = select_ground_candidates(points_cam)
    if len(pts) < min_inliers:
        raise ValueError(f"Too few ground candidates: {len(pts)}")

    rng = np.random.default_rng(seed)
    cos_limit = math.cos(math.radians(max_tilt_deg))
    best_idx = None
    best_score = (-1, float("inf"))

    for _ in range(iterations):
        ids = rng.choice(len(pts), 3, replace=False)
        a, b, c = pts[ids]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-8:
            continue
        n /= norm
        if n[1] < 0:
            n = -n
        if n[1] < cos_limit:
            continue
        d = -float(np.dot(n, a))
        residual = np.abs(pts @ n + d)
        inliers = residual < distance_threshold
        count = int(inliers.sum())
        if count < min_inliers:
            continue
        med = float(np.median(residual[inliers]))
        score = (count, -med)
        if score > best_score:
            best_score = score
            best_idx = inliers

    if best_idx is None:
        raise ValueError(
            f"RANSAC found no acceptable plane; candidates={len(pts)}. "
            "Check calibration/point-cloud alignment and relax thresholds cautiously."
        )

    # Least-squares refinement using the best inliers (SVD).
    inlier_pts = pts[best_idx]
    centroid = inlier_pts.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
    n = vh[-1]
    n /= np.linalg.norm(n)
    if n[1] < 0:
        n = -n
    if n[1] < cos_limit:
        raise ValueError("Refined plane normal violates the road-slope constraint")
    d = -float(np.dot(n, centroid))
    residual = np.abs(inlier_pts @ n + d)

    return {
        "normal": n,
        "d": d,
        "candidate_count": int(len(pts)),
        "inlier_count": int(len(inlier_pts)),
        "inlier_ratio": float(len(inlier_pts) / len(pts)),
        "median_abs_residual_m": float(np.median(residual)),
        "p90_abs_residual_m": float(np.percentile(residual, 90)),
        "tilt_deg": float(math.degrees(math.acos(np.clip(n[1], -1, 1)))),
    }


def image_rays(P: np.ndarray, uv: np.ndarray):
    M = P[:, :3]
    C = -np.linalg.solve(M, P[:, 3])
    D = np.linalg.solve(M, np.column_stack([uv, np.ones(len(uv))]).T).T
    return C, D


def intersect_plane(P: np.ndarray, uv: np.ndarray, plane):
    C, D = image_rays(P, uv)
    n = np.asarray(plane["normal"], dtype=float)
    d = float(plane["d"])
    den = D @ n
    if np.any(np.abs(den) < 1e-9):
        raise ValueError("A keypoint ray is parallel to fitted ground plane")
    t = -(float(np.dot(n, C)) + d) / den
    if np.any(t <= 0):
        raise ValueError("Fitted-ground intersection lies behind camera")
    xyz = C[None, :] + t[:, None] * D
    if np.any(xyz[:, 2] <= 1.0) or not np.isfinite(xyz).all():
        raise ValueError("Invalid fitted-ground ray intersection")
    return xyz


def ground_y_at_xz(plane, x, z):
    n = np.asarray(plane["normal"], dtype=float)
    if abs(n[1]) < 0.5:
        raise ValueError("Fitted plane is too vertical for road ground")
    return float(-(n[0] * x + n[2] * z + plane["d"]) / n[1])


def recover_box_with_plane(uv, P, plane, max_reproj_px=30.0):
    """
    Fit an upright KITTI cuboid, constraining its bottom centre to the fitted
    plane. The LiDAR plane is not used to infer object dimensions or labels.
    """
    uv = np.asarray(uv, dtype=float).reshape(8, 2)
    if not np.isfinite(uv).all():
        raise ValueError("Nonfinite 2D keypoints")

    bottom_pts = intersect_plane(P, uv[:4], plane)
    xz = bottom_pts[:, [0, 2]]
    center_guess = xz.mean(axis=0)
    h_init = base.estimate_height_from_top(P, bottom_pts, uv[4:])

    candidates = []
    for signs in base.reorder_signs():
        u = (signs[:, 0, None] * xz).mean(axis=0)
        v = (signs[:, 1, None] * xz).mean(axis=0)
        l0 = max(0.5, 2 * np.linalg.norm(u))
        w0 = max(0.4, 2 * np.linalg.norm(v))
        yaw0 = math.atan2(-u[1], u[0])
        p0 = np.array([center_guess[0], center_guess[1],
                       yaw0, l0, w0, h_init], dtype=float)
        low = np.array([-300., 1., -20*math.pi, 0.25, 0.25, 0.25])
        high = np.array([300., 250., 20*math.pi, 25., 12., 8.])
        p0 = np.clip(p0, low + 1e-5, high - 1e-5)

        def residual(p):
            cx, cz, yaw, length, width, height = p
            gy = ground_y_at_xz(plane, cx, cz)
            center = np.array([cx, gy, cz])
            try:
                pred = base.project(P, base.corners(height, width, length,
                                                    center, yaw, signs))
                return (pred - uv).ravel()
            except ValueError:
                return np.full(16, 1e3)

        fit = least_squares(
            residual, p0, bounds=(low, high), loss="soft_l1",
            f_scale=5.0, max_nfev=180, xtol=1e-9
        )
        cx, cz, yaw, length, width, height = fit.x
        gy = ground_y_at_xz(plane, cx, cz)
        center = np.array([cx, gy, cz])
        xyz = base.corners(height, width, length, center, yaw, signs)
        rms = float(np.sqrt(np.mean((base.project(P, xyz) - uv) ** 2)))
        penalty = 0.15 * max(0.0, width - length) ** 2
        candidates.append((rms + penalty, rms, height, width, length,
                           center, yaw, xyz))

    if not candidates:
        raise ValueError("All plane-constrained cuboid fits failed")
    candidates.sort(key=lambda q: q[0])
    _, rms, h, w, l, center, yaw, xyz = candidates[0]
    if rms > max_reproj_px:
        raise ValueError(f"Large reprojection error: {rms:.2f}px")
    if l < w:
        l, w = w, l
        yaw -= math.pi / 2
    box = base.Cuboid(
        "recovered_lidar_plane", h, w, l, center,
        float((yaw + math.pi) % (2 * math.pi) - math.pi),
        np.zeros(4), -1
    )
    return box, xyz, rms


def make_pred_rows(frame_data):
    rows = []
    for i, item in enumerate(frame_data):
        rows.append({
            "pred_index": int(i),
            "class_id": int(item["class_id"]),
            "bbox_xyxy": np.asarray(item["bbox_xyxy"], dtype=float),
            "score": float(item.get("confidence", 1.0)),
            "uv": np.asarray(item["keypoints_xy"], dtype=float),
            "kpt_conf": item.get("keypoint_confidence"),
        })
    return rows


def evaluate_one(box, xyz, reproj, gt, frame, det_idx, method, pred, match_iou):
    gt_xyz = base.corners(gt.h, gt.w, gt.l, gt.center, gt.yaw)
    center_delta = box.center - gt.center
    return {
        "frame": frame,
        "det_index": det_idx,
        "class_id": pred["class_id"],
        "confidence": pred["score"],
        "gt_type": gt.name,
        "gt_line_index": gt.index,
        "method": method,
        "match_iou_2d": match_iou,
        "reproj_rmse_px": reproj,
        "iou3d": base.iou3d(box, gt),
        "center_error_m": float(np.linalg.norm(center_delta)),
        "depth_z_error_m": float(center_delta[2]),
        "corner_rmse_m": base.corner_rmse(xyz, gt_xyz),
        "height_error_m": float(box.h - gt.h),
        "width_error_m": float(box.w - gt.w),
        "length_error_m": float(box.l - gt.l),
        "yaw_error_mod_pi_rad": base.angle_err_mod_pi(box.yaw, gt.yaw),
    }


def draw_compare(image_path, P, preds, gt, matches, out_path):
    im = Image.open(image_path).convert("RGB")
    dr = ImageDraw.Draw(im)
    for i, (pred, a, b) in enumerate(preds):
        for u, v in pred["uv"]:
            dr.ellipse((u-2, v-2, u+2, v+2), fill=(255, 190, 0))
        for box, color in ((a, (40, 130, 255)), (b, (255, 80, 210))):
            if box is None:
                continue
            xyz = base.corners(box.h, box.w, box.l, box.center, box.yaw)
            uv = base.project(P, xyz)
            for j, k in base.EDGE_PAIRS:
                dr.line([tuple(uv[j]), tuple(uv[k])], fill=color, width=2)
        if i in matches:
            tgt = gt[matches[i][0]]
            xyz = base.corners(tgt.h, tgt.w, tgt.l, tgt.center, tgt.yaw)
            uv = base.project(P, xyz)
            for j, k in base.EDGE_PAIRS:
                dr.line([tuple(uv[j]), tuple(uv[k])],
                        fill=(30, 220, 65), width=2)
    dr.text((6, 5), "BLUE=fixed Y=1.65 | MAGENTA=LiDAR plane | GREEN=KITTI GT",
            fill=(255,255,255), stroke_width=2, stroke_fill=(0,0,0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)


def main():
    ap = argparse.ArgumentParser(
        description="Compare fixed ground with per-frame LiDAR-RANSAC plane "
                    "using saved YOLO11-Pose predicted 2D keypoints."
    )
    ap.add_argument("--kitti-root", type=Path,
                    default=Path("/mnt/sdb/lys/MonoCoP/data/KITTIDataset"))
    ap.add_argument("--predictions-json", type=Path,
                    default=Path("runs/kitti_yolo11_predicted_3d/predicted_2d_keypoints.json"))
    ap.add_argument("--fixed-ground-y", type=float, default=1.65)
    ap.add_argument("--limit", type=int, default=0,
                    help="0 means all frames present in predictions JSON")
    ap.add_argument("--ransac-threshold", type=float, default=0.13)
    ap.add_argument("--max-tilt-deg", type=float, default=15.0)
    ap.add_argument("--min-inliers", type=int, default=35)
    ap.add_argument("--match-iou", type=float, default=0.25)
    ap.add_argument("--save-images", type=int, default=20)
    ap.add_argument("--out", type=Path,
                    default=Path("runs/kitti_lidar_ground_diagnostic"))
    args = ap.parse_args()

    kitti = base.locate_kitti(args.kitti_root)
    pred_json = json.loads(args.predictions_json.read_text(encoding="utf-8"))
    frames = sorted(pred_json.keys())
    if args.limit > 0:
        frames = frames[:args.limit]
    if not frames:
        raise RuntimeError("No frames found in predictions JSON")

    velodyne_dir = kitti / "velodyne"
    image_dir = kitti / "image_2"
    calib_dir = kitti / "calib"
    label_dir = kitti / "label_2"
    if not velodyne_dir.is_dir():
        raise FileNotFoundError(f"Missing KITTI velodyne directory: {velodyne_dir}")

    args.out.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    plane_rows = []
    frame_results = []
    image_count = 0
    counters = {
        "frames_in_predictions": len(frames),
        "frames_plane_success": 0,
        "frames_plane_failed": 0,
        "matched_detections": 0,
        "fixed_recon_success": 0,
        "lidar_plane_recon_success": 0,
    }
    failures = {}

    for fi, frame in enumerate(frames, 1):
        try:
            calib_path = calib_dir / f"{frame}.txt"
            velo_path = velodyne_dir / f"{frame}.bin"
            if not calib_path.is_file() or not velo_path.is_file():
                raise FileNotFoundError("Missing calibration or Velodyne bin")
            P, T = parse_calibration(calib_path)
            points = load_velodyne(velo_path)
            points_cam = to_camera(points, T)
            plane = fit_ground_plane_ransac(
                points_cam,
                seed=int(frame) if frame.isdigit() else fi,
                distance_threshold=args.ransac_threshold,
                max_tilt_deg=args.max_tilt_deg,
                min_inliers=args.min_inliers,
            )
            counters["frames_plane_success"] += 1
            for z in (5.0, 10.0, 20.0, 40.0):
                plane[f"ground_y_x0_z{int(z)}"] = ground_y_at_xz(plane, 0.0, z)
            plane_rows.append({"frame": frame, **{
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in plane.items()
            }})

            pred_rows = make_pred_rows(pred_json[frame])
            label_path = label_dir / f"{frame}.txt"
            targets = base.load_kitti_label(
                label_path,
                {"Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist"}
            ) if label_path.is_file() else []
            matches = base.associate_predictions_with_gt(
                pred_rows, targets, args.match_iou
            )

            frame_pred_results = []
            for i, pred in enumerate(pred_rows):
                fixed_box = plane_box = None
                fixed_xyz = plane_xyz = None
                fixed_rms = plane_rms = None
                try:
                    fixed_box, fixed_xyz, fixed_rms = base.recover_box(
                        pred["uv"], P, args.fixed_ground_y, max_reproj_px=30.0
                    )
                    counters["fixed_recon_success"] += 1
                except Exception as e:
                    failures["fixed:" + str(e)[:80]] = failures.get("fixed:" + str(e)[:80], 0) + 1
                try:
                    plane_box, plane_xyz, plane_rms = recover_box_with_plane(
                        pred["uv"], P, plane, max_reproj_px=30.0
                    )
                    counters["lidar_plane_recon_success"] += 1
                except Exception as e:
                    failures["lidar:" + str(e)[:80]] = failures.get("lidar:" + str(e)[:80], 0) + 1

                frame_pred_results.append((pred, fixed_box, plane_box))

                if i in matches:
                    counters["matched_detections"] += 1
                    gt_idx, iou2d = matches[i]
                    gt = targets[gt_idx]
                    if fixed_box is not None:
                        all_metrics.append(evaluate_one(
                            fixed_box, fixed_xyz, fixed_rms, gt, frame,
                            pred["pred_index"], "fixed_y_1.65", pred, iou2d
                        ))
                    if plane_box is not None:
                        all_metrics.append(evaluate_one(
                            plane_box, plane_xyz, plane_rms, gt, frame,
                            pred["pred_index"], "lidar_ransac_plane", pred, iou2d
                        ))

            frame_results.append({
                "frame": frame,
                "plane": {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in plane.items()
                },
                "num_predictions": len(pred_rows),
                "num_matches": len(matches),
            })

            image_path = image_dir / f"{frame}.png"
            if image_count < args.save_images and image_path.is_file():
                draw_compare(
                    image_path, P, frame_pred_results, targets, matches,
                    args.out / "visualizations" / f"{frame}.png"
                )
                image_count += 1

        except Exception as e:
            counters["frames_plane_failed"] += 1
            failures["frame:" + str(e)[:100]] = failures.get("frame:" + str(e)[:100], 0) + 1

        if fi % 10 == 0 or fi == len(frames):
            print(
                f"[{fi}/{len(frames)}] plane_ok={counters['frames_plane_success']} "
                f"plane_fail={counters['frames_plane_failed']} "
                f"fixed3d={counters['fixed_recon_success']} "
                f"lidar3d={counters['lidar_plane_recon_success']} "
                f"matched={counters['matched_detections']}",
                flush=True
            )

    # Paired comparison: keep only detections for which both methods succeeded.
    by_key = {}
    for row in all_metrics:
        key = (row["frame"], row["det_index"], row["gt_line_index"])
        by_key.setdefault(key, {})[row["method"]] = row

    paired = []
    for key, methods in by_key.items():
        if "fixed_y_1.65" in methods and "lidar_ransac_plane" in methods:
            a, b = methods["fixed_y_1.65"], methods["lidar_ransac_plane"]
            paired.append({
                "frame": key[0], "det_index": key[1],
                "iou3d_fixed": a["iou3d"], "iou3d_lidar_plane": b["iou3d"],
                "iou3d_change": b["iou3d"] - a["iou3d"],
                "center_error_fixed_m": a["center_error_m"],
                "center_error_lidar_plane_m": b["center_error_m"],
                "center_error_change_m": b["center_error_m"] - a["center_error_m"],
                "depth_abs_error_fixed_m": abs(a["depth_z_error_m"]),
                "depth_abs_error_lidar_plane_m": abs(b["depth_z_error_m"]),
                "depth_abs_error_change_m": abs(b["depth_z_error_m"]) - abs(a["depth_z_error_m"]),
                "corner_rmse_fixed_m": a["corner_rmse_m"],
                "corner_rmse_lidar_plane_m": b["corner_rmse_m"],
                "corner_rmse_change_m": b["corner_rmse_m"] - a["corner_rmse_m"],
                "height_abs_error_fixed_m": abs(a["height_error_m"]),
                "height_abs_error_lidar_plane_m": abs(b["height_error_m"]),
                "length_abs_error_fixed_m": abs(a["length_error_m"]),
                "length_abs_error_lidar_plane_m": abs(b["length_error_m"]),
            })

    def summarize(rows, method):
        q = [r for r in rows if r["method"] == method]
        out = {"n": len(q)}
        for metric in ("iou3d", "center_error_m", "depth_z_error_m",
                       "corner_rmse_m", "reproj_rmse_px",
                       "height_error_m", "width_error_m", "length_error_m"):
            vals = np.asarray([r[metric] for r in q], dtype=float)
            if len(vals):
                out[metric] = {
                    "median": float(np.median(vals)),
                    "mean": float(np.mean(vals)),
                    "p90": float(np.percentile(vals, 90)),
                }
        return out

    plane_tilts = np.asarray([r["tilt_deg"] for r in plane_rows], dtype=float)
    summary = {
        "description": "Same saved YOLO11 predicted 2D keypoints; compare fixed Y=1.65 with per-frame LiDAR RANSAC plane. LiDAR plane is diagnostic only.",
        "counters": counters,
        "plane_quality": {
            "n": int(len(plane_rows)),
            "tilt_deg_median": float(np.median(plane_tilts)) if len(plane_tilts) else None,
            "inliers_median": float(np.median([r["inlier_count"] for r in plane_rows])) if plane_rows else None,
            "median_abs_residual_m_median": float(np.median([r["median_abs_residual_m"] for r in plane_rows])) if plane_rows else None,
            "ground_y_at_x0_z10_m_median": float(np.median([r["ground_y_x0_z10"] for r in plane_rows])) if plane_rows else None,
        },
        "metrics_fixed": summarize(all_metrics, "fixed_y_1.65"),
        "metrics_lidar_plane": summarize(all_metrics, "lidar_ransac_plane"),
        "paired_comparison": {
            "n": len(paired),
            "median_iou3d_change": float(np.median([r["iou3d_change"] for r in paired])) if paired else None,
            "median_center_error_change_m": float(np.median([r["center_error_change_m"] for r in paired])) if paired else None,
            "median_depth_abs_error_change_m": float(np.median([r["depth_abs_error_change_m"] for r in paired])) if paired else None,
            "median_corner_rmse_change_m": float(np.median([r["corner_rmse_change_m"] for r in paired])) if paired else None,
            "fraction_iou_improved": float(np.mean([r["iou3d_change"] > 0 for r in paired])) if paired else None,
            "fraction_center_error_improved": float(np.mean([r["center_error_change_m"] < 0 for r in paired])) if paired else None,
        },
        "failures": failures,
        "caution": "This experiment uses LiDAR only to estimate the ground plane, then reconstructs the object from predicted 2D keypoints. GT 3D labels are read only after reconstruction for evaluation. The plane-aware solver constrains the cuboid bottom centre to the plane while retaining the upright KITTI cuboid convention.",
    }

    def write_csv(path, rows):
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write_csv(args.out / "metrics_both_methods.csv", all_metrics)
    write_csv(args.out / "paired_comparison.csv", paired)
    write_csv(args.out / "ground_planes.csv", plane_rows)
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out / "frame_plane_results.json").write_text(
        json.dumps(frame_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n========== SUMMARY ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved results to: {args.out}")


if __name__ == "__main__":
    main()
