#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO11 3D Pose Visualization

功能：
1. YOLO11 Pose模型推理
2. 8个3D投影角点绘制
3. 12条边绘制
4. 深色Halo描边
5. 底部平面半透明填充
6. 类别颜色区分
7. 保存图片

"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# =========================================================
# 1. 路径配置
# =========================================================

DEFAULT_WEIGHTS = (
    "/mnt/sdb/lys/yolo11/runs/pose/train/weights/best.pt"
)

DEFAULT_SOURCE = (
    "/mnt/sdb/lys/yolo11/datasets/"
    "YOLO11_3D_Pose_Full/images/test"
)

DEFAULT_OUTPUT = (
    "/mnt/sdb/lys/yolo11/runs/pose/visualize_3d"
)


# =========================================================
# 2. 八角点排列
# =========================================================

# 这是针对你自己的KITTI转换标签定义的顺序：
#
# 0  1  2  3：底部四个角点
# 4  5  6  7：顶部四个角点
#
# 上下对应关系：
# 0 <-> 4
# 1 <-> 5
# 2 <-> 6
# 3 <-> 7

GROUND_INDICES = [0, 1, 2, 3]

TOP_INDICES = [4, 5, 6, 7]


# =========================================================
# 3. 3D立方体边线
# =========================================================

GROUND_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
]

TOP_EDGES = [
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
]

VERTICAL_EDGES = [
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]


# =========================================================
# 4. 颜色配置
# 参考 UrbanOmniDetect/uod/style.py
# RGB格式，使用OpenCV时转换为BGR
# =========================================================

CLASS_COLORS = {
    0: (64, 156, 255),    # Car：蓝色
    1: (255, 168, 64),    # Person：橙色
    2: (181, 99, 232),    # Bike：紫色
}

CLASS_NAMES = {
    0: "Car",
    1: "Person",
    2: "Bike",
}


def rgb_to_bgr(color):
    return (
        int(color[2]),
        int(color[1]),
        int(color[0])
    )


# =========================================================
# 5. 半透明多边形填充
# 参考 UrbanOmniDetect fill_poly_alpha()
# =========================================================

def fill_poly_alpha(image, points, color_rgb, alpha=0.18):

    if alpha <= 0:
        return

    points = np.round(points).astype(np.int32)

    h, w = image.shape[:2]

    bx, by, bw, bh = cv2.boundingRect(points)

    x0 = max(bx, 0)
    y0 = max(by, 0)

    x1 = min(bx + bw, w)
    y1 = min(by + bh, h)

    if x1 <= x0 or y1 <= y0:
        return

    roi = image[y0:y1, x0:x1]

    overlay = roi.copy()

    local_points = points - np.array([x0, y0])

    cv2.fillPoly(
        overlay,
        [local_points],
        rgb_to_bgr(color_rgb),
        cv2.LINE_AA
    )

    cv2.addWeighted(
        overlay,
        alpha,
        roi,
        1 - alpha,
        0,
        roi
    )


# =========================================================
# 6. 抗锯齿线段绘制
# 参考 UrbanOmniDetect _stroke_segments()
# =========================================================

def stroke_segments(
    image,
    segments,
    color_rgb,
    thickness=2,
    halo=True
):

    color = rgb_to_bgr(color_rgb)

    # 深色轮廓层
    if halo:

        for p1, p2 in segments:

            a = tuple(np.round(p1).astype(int))
            b = tuple(np.round(p2).astype(int))

            cv2.line(
                image,
                a,
                b,
                (8, 10, 14),
                thickness + 2,
                cv2.LINE_AA
            )

    # 彩色主体层
    for p1, p2 in segments:

        a = tuple(np.round(p1).astype(int))
        b = tuple(np.round(p2).astype(int))

        cv2.line(
            image,
            a,
            b,
            color,
            thickness,
            cv2.LINE_AA
        )


# =========================================================
# 7. 核心函数：绘制3D立方体
# 参考 UrbanOmniDetect draw_cuboid()
# =========================================================

def draw_cuboid(
    image,
    keypoints,
    color_rgb,
    thickness=2,
    point_radius=4,
    fill=True,
    show_indices=False
):

    keypoints = np.asarray(
        keypoints,
        dtype=np.float64
    )

    if keypoints.shape != (8, 2):
        return False

    if not np.isfinite(keypoints).all():
        return False

    # -----------------------------------------------
    # A. 按照你的标签顺序排列
    # -----------------------------------------------

    # 先底部，再顶部
    k = keypoints[
        GROUND_INDICES + TOP_INDICES
    ]

    # -----------------------------------------------
    # B. 底部四边形半透明填充
    # -----------------------------------------------

    if fill:

        fill_poly_alpha(
            image,
            k[:4],
            color_rgb,
            alpha=0.18
        )

    # -----------------------------------------------
    # C. 绘制顶部、底部、垂直边
    # -----------------------------------------------

    top_segments = [
        (k[a], k[b])
        for a, b in TOP_EDGES
    ]

    ground_segments = [
        (k[a], k[b])
        for a, b in GROUND_EDGES
    ]

    vertical_segments = [
        (k[a], k[b])
        for a, b in VERTICAL_EDGES
    ]

    # 顶部四条边
    stroke_segments(
        image,
        top_segments,
        color_rgb,
        thickness=thickness,
        halo=True
    )

    # 四条垂直边
    stroke_segments(
        image,
        vertical_segments,
        color_rgb,
        thickness=thickness,
        halo=True
    )

    # 底部四条边
    stroke_segments(
        image,
        ground_segments,
        color_rgb,
        thickness=thickness + 1,
        halo=True
    )

    # -----------------------------------------------
    # D. 绘制八个角点
    # -----------------------------------------------

    for idx, point in enumerate(k):

        x, y = np.round(point).astype(int)

        # 白色外圈
        cv2.circle(
            image,
            (x, y),
            point_radius + 2,
            (255, 255, 255),
            -1,
            cv2.LINE_AA
        )

        # 类别颜色内圈
        cv2.circle(
            image,
            (x, y),
            point_radius,
            rgb_to_bgr(color_rgb),
            -1,
            cv2.LINE_AA
        )

        # 可选角点编号
        if show_indices:

            cv2.putText(
                image,
                str(GROUND_INDICES[idx]
                    if idx < 4 else TOP_INDICES[idx - 4]),
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

    return True


# =========================================================
# 8. 绘制类别及置信度
# 不绘制2D矩形框
# =========================================================

def draw_label(
    image,
    keypoints,
    class_id,
    confidence,
    names
):

    if isinstance(names, dict):
        name = names.get(class_id, str(class_id))
    else:
        name = (
            names[class_id]
            if class_id < len(names)
            else str(class_id)
        )

    text = f"{name} {confidence:.2f}"

    x = max(5, int(np.min(keypoints[:, 0])))
    y = max(22, int(np.min(keypoints[:, 1])) - 10)

    color = rgb_to_bgr(
        CLASS_COLORS.get(class_id, (150, 162, 176))
    )

    # 深色文字描边
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        4,
        cv2.LINE_AA
    )

    # 彩色文字
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA
    )


# =========================================================
# 9. 单张图片可视化
# =========================================================

def visualize_result(
    result,
    show_indices=False,
    fill=True
):

    image = result.orig_img.copy()

    if result.boxes is None or len(result.boxes) == 0:
        return image, 0

    if result.keypoints is None:
        raise RuntimeError(
            "模型没有输出关键点，请确认加载的是YOLO11 Pose模型。"
        )

    # 读取8个关键点
    keypoints = (
        result.keypoints.xy
        .detach()
        .cpu()
        .numpy()
    )

    # 读取类别和置信度
    classes = (
        result.boxes.cls
        .detach()
        .cpu()
        .numpy()
        .astype(int)
    )

    confidences = (
        result.boxes.conf
        .detach()
        .cpu()
        .numpy()
    )

    if keypoints.ndim != 3 or keypoints.shape[1:] != (8, 2):
        raise RuntimeError(
            f"期望(N,8,2)，实际得到{keypoints.shape}"
        )

    if len(keypoints) != len(classes):
        raise RuntimeError(
            "检测框与关键点数量不一致。"
        )

    detection_count = 0

    for i, points in enumerate(keypoints):

        class_id = int(classes[i])
        confidence = float(confidences[i])

        color = CLASS_COLORS.get(
            class_id,
            (150, 162, 176)
        )

        # 绘制3D立方体
        success = draw_cuboid(
            image,
            points,
            color,
            thickness=2,
            point_radius=4,
            fill=fill,
            show_indices=show_indices
        )

        if not success:
            continue

        # 绘制类别和置信度
        draw_label(
            image,
            points,
            class_id,
            confidence,
            result.names
        )

        detection_count += 1

    return image, detection_count


# =========================================================
# 10. 收集测试图片
# =========================================================

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".webp"
}


def collect_images(source):

    source = Path(source)

    if source.is_file():

        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError("仅支持图片文件。")

        return [source]

    if not source.is_dir():
        raise FileNotFoundError(source)

    return sorted(
        p for p in source.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )


# =========================================================
# 11. 主程序
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description="YOLO11 3D八角点可视化"
    )

    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS
    )

    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.7
    )

    parser.add_argument(
        "--device",
        default="0"
    )

    parser.add_argument(
        "--max-det",
        type=int,
        default=500
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0
    )

    parser.add_argument(
        "--show-indices",
        action="store_true"
    )

    parser.add_argument(
        "--no-fill",
        action="store_true"
    )

    args = parser.parse_args()

    weights = Path(args.weights)

    if not weights.is_file():
        raise FileNotFoundError(
            f"模型权重不存在：{weights}"
        )

    # 加载模型
    print("Loading YOLO11 Pose model...")

    model = YOLO(str(weights))

    print("Model classes:", model.names)

    # 收集测试图片
    images = collect_images(args.source)

    if args.limit > 0:
        images = images[:args.limit]

    if not images:
        raise RuntimeError("没有找到测试图片。")

    output_dir = Path(args.output)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    source_dir = Path(args.source)

    total_detections = 0
    processed = 0

    print("Total images:", len(images))

    # 逐张推理
    for index, image_path in enumerate(images, 1):

        results = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            max_det=args.max_det,
            verbose=False
        )

        if not results:
            continue

        result = results[0]

        # 生成3D可视化图
        vis, count = visualize_result(
            result,
            show_indices=args.show_indices,
            fill=not args.no_fill
        )

        # 输出路径
        if source_dir.is_dir():

            relative = image_path.relative_to(source_dir)

            save_path = (
                output_dir / relative
            ).with_suffix(".jpg")

        else:

            save_path = (
                output_dir /
                f"{image_path.stem}_3d.jpg"
            )

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        cv2.imwrite(
            str(save_path),
            vis
        )

        total_detections += count
        processed += 1

        print(
            f"[{index}/{len(images)}] "
            f"{image_path.name} "
            f"Detections={count}"
        )

    print("\n========== Finished ==========")

    print("Processed:", processed)

    print("Total detections:", total_detections)

    print("Output:", output_dir)


if __name__ == "__main__":
    main()
