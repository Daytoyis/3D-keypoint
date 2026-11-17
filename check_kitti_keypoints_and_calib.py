#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 KITTI YOLO11-Pose 的8个二维关键点排列，并审查 KITTI 相机标定文件。

默认路径按当前项目记录设置：
  KITTI 原始数据: /mnt/sdb/lys/MonoCoP/data/KITTIDataset
  YOLO Pose 数据: /mnt/sdb/lys/yolo11/datasets/KITTI_YOLO11Pose

检查内容：
1) 从 YOLO 标签读取每个目标的 8 个二维关键点；
2) 输出编号关键点可视化，检查 0-3 / 4-7 的上下关系及 0↔4 等配对；
3) 统计每个目标中两组点的平均图像 y 坐标，给出“哪组更靠下”的启发式结果；
4) 读取 KITTI calib，解析 P2、R0_rect、Tr_velo_to_cam；
5) 根据 P2=[M|p4] 计算 C=-M^{-1}p4，并输出 P2 左侧3x3矩阵；
6) 明确提示：P2 本身不提供相机相对于道路地面的高度/地面平面。

注意：
- 仅凭二维点的 y 坐标只能做上下关系的启发式检查，不能证明语义角点顺序。
- 最终顺序必须结合数据生成脚本的 corners 构造顺序及编号可视化确认。
- 本脚本不读取 label_2 中的3D框来参与恢复；label_2也不是本脚本的输入。
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_KITTI = Path("/mnt/sdb/lys/MonoCoP/data/KITTIDataset")
DEFAULT_POSE = Path("/mnt/sdb/lys/yolo11/datasets/KITTI_YOLO11Pose")


def parse_calib_file(path: Path):
    data = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        arr = np.fromstring(value.strip(), sep=" ", dtype=float)
        if arr.size:
            data[key.strip()] = arr
    return data


def inspect_calibration(calib_dir: Path, max_files: int = 5):
    files = sorted(calib_dir.glob("*.txt"))[:max_files]
    if not files:
        print(f"[CALIB][WARN] No *.txt found in: {calib_dir}")
        return []

    rows = []
    print("\n========== KITTI CALIBRATION ==========")
    for path in files:
        d = parse_calib_file(path)
        print(f"\n[{path.name}]")
        if "P2" not in d:
            print("  ERROR: P2 missing")
            continue

        P2 = d["P2"]
        if P2.size != 12:
            print(f"  ERROR: P2 has {P2.size} values, expected 12")
            continue
        P2 = P2.reshape(3, 4)
        M = P2[:, :3]
        p4 = P2[:, 3]

        try:
            C = -np.linalg.solve(M, p4)
            K_like = M.copy()
            det_M = float(np.linalg.det(M))
            cond_M = float(np.linalg.cond(M))
        except np.linalg.LinAlgError:
            C = np.full(3, np.nan)
            K_like = M
            det_M = float("nan")
            cond_M = float("inf")

        print("  P2 =")
        print(np.array2string(P2, precision=6, suppress_small=True))
        print("  Left 3x3 block M (K-like for rectified projection) =")
        print(np.array2string(K_like, precision=6, suppress_small=True))
        print(f"  fx~{M[0,0]:.6f}, fy~{M[1,1]:.6f}, cx~{M[0,2]:.6f}, cy~{M[1,2]:.6f}")
        print(f"  det(M)={det_M:.8g}, cond(M)={cond_M:.6g}")
        print(f"  P2-derived camera center C=-M^-1 p4 = {C.tolist()}")

        if "R0_rect" in d and d["R0_rect"].size == 9:
            R0 = d["R0_rect"].reshape(3, 3)
            ortho_err = float(np.linalg.norm(R0 @ R0.T - np.eye(3)))
            det_R0 = float(np.linalg.det(R0))
            print("  R0_rect =")
            print(np.array2string(R0, precision=6, suppress_small=True))
            print(f"  R0 orthogonality error={ortho_err:.3e}, det={det_R0:.6f}")
        else:
            ortho_err, det_R0 = float("nan"), float("nan")
            print("  R0_rect: missing or invalid")

        if "Tr_velo_to_cam" in d and d["Tr_velo_to_cam"].size == 12:
            Tr = d["Tr_velo_to_cam"].reshape(3, 4)
            print("  Tr_velo_to_cam present (LiDAR-to-camera extrinsic);")
            print("  this is NOT camera-to-road extrinsic.")
        else:
            print("  Tr_velo_to_cam: missing or invalid")

        print("  IMPORTANT: P2 gives the projection model; it does not by itself")
        print("             provide camera height above road or the road plane.")

        rows.append({
            "frame": path.stem,
            "fx_like": float(M[0,0]),
            "fy_like": float(M[1,1]),
            "cx_like": float(M[0,2]),
            "cy_like": float(M[1,2]),
            "camera_center_x": float(C[0]),
            "camera_center_y": float(C[1]),
            "camera_center_z": float(C[2]),
            "det_M": det_M,
            "cond_M": cond_M,
            "R0_orthogonality_error": ortho_err,
            "R0_det": det_R0,
            "has_Tr_velo_to_cam": "Tr_velo_to_cam" in d,
        })
    return rows


def load_yaml_paths(dataset_root: Path, split: str):
    yaml_candidates = list(dataset_root.glob("*.yaml")) + list(dataset_root.glob("*.yml"))
    for yp in yaml_candidates:
        try:
            import yaml
            cfg = yaml.safe_load(yp.read_text(encoding="utf-8")) or {}
            val = cfg.get(split)
            if val:
                p = Path(str(val))
                if not p.is_absolute():
                    p = (yp.parent / p).resolve()
                if p.is_dir():
                    # Common YOLO convention: images/train -> labels/train
                    if p.name == "images":
                        lp = p.parent / "labels"
                    else:
                        lp = p.parent / "labels" / p.name
                    if lp.is_dir():
                        return p, lp
                if p.is_file():
                    return p.parent, p.parent.parent / "labels" / p.parent.name
        except Exception:
            pass
    return None, None


def find_image_label_dirs(dataset_root: Path, split: str):
    img, lab = load_yaml_paths(dataset_root, split)
    if img and lab and img.exists() and lab.exists():
        return img, lab

    candidates = [
        (dataset_root / "images" / split, dataset_root / "labels" / split),
        (dataset_root / split / "images", dataset_root / split / "labels"),
        (dataset_root / "images", dataset_root / "labels"),
        (dataset_root / split, dataset_root / "labels" / split),
    ]
    for im, la in candidates:
        if im.is_dir() and la.is_dir():
            return im, la

    # Recursive fallback: find directories named images/<split> and labels/<split>.
    image_dirs = [p for p in dataset_root.rglob("*") if p.is_dir() and p.name == split and p.parent.name == "images"]
    for im in image_dirs:
        la = im.parent.parent / "labels" / split
        if la.is_dir():
            return im, la
    raise FileNotFoundError(
        f"Cannot auto-detect image/label directories under {dataset_root}. "
        "Pass --image-dir and --label-dir explicitly."
    )


def parse_pose_label(label_path: Path):
    objects = []
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        vals = line.split()
        if len(vals) < 21:
            print(f"[LABEL][WARN] {label_path.name}:{line_no}: {len(vals)} columns (<21), skipped")
            continue
        try:
            cls = int(float(vals[0]))
            bbox = np.asarray([float(x) for x in vals[1:5]], dtype=float)
            kpts = np.asarray([float(x) for x in vals[5:21]], dtype=float).reshape(8, 2)
        except Exception as e:
            print(f"[LABEL][WARN] parse error {label_path.name}:{line_no}: {e}")
            continue
        objects.append({"class_id": cls, "bbox": bbox, "kpts": kpts, "line_no": line_no})
    return objects


def find_font():
    for f in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if Path(f).exists():
            try:
                return ImageFont.truetype(f, 14)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_keypoints(image_path: Path, objects, out_path: Path):
    im = Image.open(image_path).convert("RGB")
    W, H = im.size
    draw = ImageDraw.Draw(im)
    font = find_font()
    for oi, obj in enumerate(objects):
        pts = obj["kpts"].copy()
        # YOLO pose labels are assumed normalized [0,1]; tolerate pixel labels only if clearly outside.
        if np.nanmax(np.abs(pts)) <= 1.5:
            pts[:, 0] *= W
            pts[:, 1] *= H

        # Draw cyclic faces and proposed vertical correspondences.
        for a, b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4)]:
            draw.line([tuple(pts[a]), tuple(pts[b])], fill=(255, 210, 0), width=2)
        for a, b in [(0,4),(1,5),(2,6),(3,7)]:
            draw.line([tuple(pts[a]), tuple(pts[b])], fill=(0, 230, 255), width=2)

        for k, (x, y) in enumerate(pts):
            r = 5
            draw.ellipse((x-r, y-r, x+r, y+r), fill=(255, 50, 50), outline=(255,255,255), width=1)
            draw.text((x+5, y-15), str(k), fill=(255,255,255), font=font,
                      stroke_width=2, stroke_fill=(0,0,0))
        draw.text((8, 8 + oi*18), f"obj{oi} class={obj['class_id']} line={obj['line_no']}",
                  fill=(255,255,255), font=font, stroke_width=2, stroke_fill=(0,0,0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)


def inspect_pose_dataset(image_dir: Path, label_dir: Path, out_dir: Path, max_images: int):
    image_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in image_exts])
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")

    rows = []
    total_objects = 0
    count_group0_lower = 0
    count_group4_lower = 0
    count_ties = 0
    missing_labels = 0

    vis_dir = out_dir / "keypoint_order_visualizations"
    vis_saved = 0

    print("\n========== YOLO11-POSE KEYPOINT LABELS ==========")
    print(f"Image directory: {image_dir}")
    print(f"Label directory: {label_dir}")
    print(f"Images found: {len(images)}")

    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing_labels += 1
            continue
        objs = parse_pose_label(label_path)
        if not objs:
            continue

        with Image.open(image_path) as im:
            W, H = im.size

        for oi, obj in enumerate(objs):
            pts = obj["kpts"].copy()
            if np.nanmax(np.abs(pts)) <= 1.5:
                pts[:, 0] *= W
                pts[:, 1] *= H
            # In image coordinates, larger v is lower on the image.
            mean_v_0_3 = float(np.mean(pts[:4, 1]))
            mean_v_4_7 = float(np.mean(pts[4:, 1]))
            if mean_v_0_3 > mean_v_4_7 + 1.0:
                heuristic = "0-3_lower_candidate_bottom"
                count_group0_lower += 1
            elif mean_v_4_7 > mean_v_0_3 + 1.0:
                heuristic = "4-7_lower_candidate_bottom"
                count_group4_lower += 1
            else:
                heuristic = "ambiguous"
                count_ties += 1

            total_objects += 1
            rows.append({
                "image": image_path.name,
                "label": label_path.name,
                "object_line": obj["line_no"],
                "class_id": obj["class_id"],
                "mean_y_0_3_px": mean_v_0_3,
                "mean_y_4_7_px": mean_v_4_7,
                "group0_minus_group1_y_px": mean_v_0_3 - mean_v_4_7,
                "vertical_order_heuristic": heuristic,
                "pair_0_4_y_delta": float(pts[4,1] - pts[0,1]),
                "pair_1_5_y_delta": float(pts[5,1] - pts[1,1]),
                "pair_2_6_y_delta": float(pts[6,1] - pts[2,1]),
                "pair_3_7_y_delta": float(pts[7,1] - pts[3,1]),
            })

        if vis_saved < max_images:
            draw_keypoints(image_path, objs, vis_dir / f"{image_path.stem}_keypoints.jpg")
            vis_saved += 1

    csv_path = out_dir / "keypoint_order_statistics.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"Objects inspected: {total_objects}")
    print(f"Group 0-3 lower in image: {count_group0_lower}")
    print(f"Group 4-7 lower in image: {count_group4_lower}")
    print(f"Ambiguous: {count_ties}")
    print(f"Images without label: {missing_labels}")
    print(f"Visualizations saved: {vis_saved} -> {vis_dir}")
    print(f"Statistics CSV: {csv_path}")
    print("\nInterpretation:")
    print("  Larger image y means lower in the image.")
    print("  If group 0-3 has larger mean y in most vehicles, it is consistent")
    print("  with 0-3 being bottom corners and 4-7 being top corners.")
    print("  This is only a heuristic; inspect numbered overlays and the label")
    print("  generation code to verify exact corner identities and pairings.")
    return rows


def inspect_generator_source(path: Path):
    print("\n========== OPTIONAL LABEL GENERATOR SOURCE ==========")
    if not path or not path.exists():
        print("No --generator-script supplied or file not found; source not inspected.")
        return
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    patterns = re.compile(r"corner|corners|keypoint|kpt|bottom|top|vertices|rotation", re.I)
    hits = [(i+1, line.strip()) for i, line in enumerate(lines) if patterns.search(line)]
    for line_no, line in hits[:100]:
        print(f"{line_no:5d}: {line}")
    if not hits:
        print("No obvious corner/keypoint construction lines found.")


def main():
    ap = argparse.ArgumentParser(description="Audit KITTI YOLO11-Pose keypoint order and camera calibration")
    ap.add_argument("--kitti-root", type=Path, default=DEFAULT_KITTI)
    ap.add_argument("--pose-dataset-root", type=Path, default=DEFAULT_POSE)
    ap.add_argument("--split", default="train")
    ap.add_argument("--image-dir", type=Path, default=None)
    ap.add_argument("--label-dir", type=Path, default=None)
    ap.add_argument("--calib-dir", type=Path, default=None)
    ap.add_argument("--generator-script", type=Path, default=None)
    ap.add_argument("--max-vis-images", type=int, default=30)
    ap.add_argument("--max-calib-files", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("runs/kitti_geometry_check"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    calib_dir = args.calib_dir or (args.kitti_root / "training" / "calib")
    inspect_calibration(calib_dir, args.max_calib_files)

    if args.image_dir and args.label_dir:
        image_dir, label_dir = args.image_dir, args.label_dir
    else:
        image_dir, label_dir = find_image_label_dirs(args.pose_dataset_root, args.split)

    rows = inspect_pose_dataset(image_dir, label_dir, args.out, args.max_vis_images)
    inspect_generator_source(args.generator_script)

    report = {
        "kitti_root": str(args.kitti_root),
        "pose_dataset_root": str(args.pose_dataset_root),
        "split": args.split,
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "calib_dir": str(calib_dir),
        "objects_inspected": len(rows),
        "visualization_dir": str(args.out / "keypoint_order_visualizations"),
        "statistics_csv": str(args.out / "keypoint_order_statistics.csv"),
        "warning": "Keypoint group vertical ordering is heuristic; verify numbered overlays and label-generation source. P2 does not supply camera-to-road height/plane."
    }
    (args.out / "check_report.json").write_text(
        __import__("json").dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n========== CHECK FINISHED ==========")
    print(f"Report: {args.out / 'check_report.json'}")


if __name__ == "__main__":
    main()
