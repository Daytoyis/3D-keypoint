import json
import csv
import numpy as np
import open3d as o3d
from pathlib import Path
from PIL import Image, ImageDraw
from ultralytics import YOLO

# ============================================================
# 1. 路径与参数
# ============================================================

ROOT = Path("/mnt/sdb/lys/yolo11/datasets/DAIR-V2X-I")
META = ROOT / "single-infrastructure-side"

IMG = ROOT / "single-infrastructure-side-image/000000.jpg"
PCD = ROOT / "single-infrastructure-side-velodyne/000000.pcd"
CALIB = META / "calib/virtuallidar_to_camera/000000.json"
INTR = META / "calib/camera_intrinsic/000000.json"
LABEL = META / "label/camera/000000.json"

WEIGHT = Path(
    "runs/pose/train-3/weights/best.pt"
)

OUT = Path("runs/dair_3d_demo")
OUT.mkdir(parents=True, exist_ok=True)

IMGSZ = 640
CONF = 0.05
CAR_CLASS_ID = 0

# 关键点附近搜索半径，单位：像素
SEARCH_RADII = [12, 20, 30, 45, 60]

# 关键点附近最多使用多少个 LiDAR 点估计深度
MAX_LOCAL_POINTS = 7

# 立方体边连接关系
EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]


# ============================================================
# 2. 工具函数
# ============================================================

def box_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2-x1) * max(0, y2-y1)

    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])

    return inter / max(area_a + area_b - inter, 1e-9)


def expand_box(box, ratio=0.12):
    x1, y1, x2, y2 = box
    w = x2-x1
    h = y2-y1

    return np.array([
        x1-ratio*w,
        y1-ratio*h,
        x2+ratio*w,
        y2+ratio*h
    ])


def box_mask(u, v, box):
    return (
        (u >= box[0]) & (u <= box[2])
        & (v >= box[1]) & (v <= box[3])
    )


def lift_to_3d(u, v, depth, K, R, t):
    """
    Pixel + camera Z depth -> Camera XYZ -> LiDAR XYZ.

    注意：R 不一定是严格正交矩阵，
    因此使用 solve(R, ...) 而不是 R.T。
    """

    pixel = np.array([u, v, 1.0])

    ray = np.linalg.inv(K) @ pixel
    cam_xyz = ray * depth

    lidar_xyz = np.linalg.solve(
        R,
        cam_xyz - t
    )

    return cam_xyz, lidar_xyz


# ============================================================
# 3. 读取相机标定
# ============================================================

print("\n========== 1. LOAD CALIBRATION ==========", flush=True)

cal = json.loads(CALIB.read_text())
intr = json.loads(INTR.read_text())
labels = json.loads(LABEL.read_text())

R = np.asarray(
    cal["rotation"], dtype=np.float64
)

t = np.asarray(
    cal["translation"], dtype=np.float64
).reshape(3)

K = np.asarray(
    intr["cam_K"], dtype=np.float64
).reshape(3, 3)

print("Rotation determinant:", np.linalg.det(R))
print("Rotation inverse condition:", np.linalg.cond(R))
print("Camera intrinsic:\n", K)


# ============================================================
# 4. 读取并投影 LiDAR 点云
# ============================================================

print("\n========== 2. LOAD POINT CLOUD ==========", flush=True)

pcd = o3d.io.read_point_cloud(str(PCD))

xyz = np.asarray(
    pcd.points, dtype=np.float64
)

print("Original points:", len(xyz), flush=True)

# Virtual LiDAR -> Camera
cam = xyz @ R.T + t

valid = (
    np.isfinite(cam).all(axis=1)
    & (cam[:, 2] > 0.1)
)

xyz = xyz[valid]
cam = cam[valid]

# Camera -> Image
projected = cam @ K.T

u = projected[:, 0] / projected[:, 2]
v = projected[:, 1] / projected[:, 2]

depth = cam[:, 2]

img = Image.open(IMG).convert("RGB")
W, H = img.size

inside = (
    (u >= 0) & (u < W)
    & (v >= 0) & (v < H)
)

xyz = xyz[inside]
cam = cam[inside]

u = u[inside]
v = v[inside]
depth = depth[inside]

print("Points inside image:", len(xyz), flush=True)


# ============================================================
# 5. 自动选择点云最多的 Car
# ============================================================

print("\n========== 3. SELECT TARGET CAR ==========", flush=True)

gt_cars = []

for idx, obj in enumerate(labels):

    if str(obj.get("type", "")).lower() != "car":
        continue

    bb = obj["2d_box"]

    box = np.array([
        float(bb["xmin"]),
        float(bb["ymin"]),
        float(bb["xmax"]),
        float(bb["ymax"])
    ])

    mask = box_mask(u, v, box)

    count = int(mask.sum())

    gt_cars.append({
        "index": idx,
        "box": box,
        "count": count,
        "object": obj
    })

if not gt_cars:
    raise RuntimeError("No Car objects found in DAIR labels.")

target = max(
    gt_cars,
    key=lambda x: x["count"]
)

gt_box = target["box"]

print("Target label index:", target["index"])
print("Target class: Car")
print("GT box:", gt_box)
print("LiDAR points inside GT box:", target["count"])


# ============================================================
# 6. YOLO11n-Pose 推理
# ============================================================

print("\n========== 4. YOLO POSE INFERENCE ==========", flush=True)

model = YOLO(str(WEIGHT))

results = model.predict(
    source=str(IMG),
    imgsz=IMGSZ,
    conf=CONF,
    iou=0.7,
    max_det=500,
    device=0,
    verbose=False
)

result = results[0]

if result.boxes is None or len(result.boxes) == 0:
    raise RuntimeError("YOLO detected no objects.")

if result.keypoints is None:
    raise RuntimeError("YOLO keypoints are unavailable.")

pred_boxes = result.boxes.xyxy.cpu().numpy()
pred_cls = result.boxes.cls.cpu().numpy().astype(int)
pred_conf = result.boxes.conf.cpu().numpy()

pred_kpts = result.keypoints.xy.cpu().numpy()

if pred_kpts.shape[1:] != (8, 2):
    raise RuntimeError(
        f"Expected (N,8,2), got {pred_kpts.shape}"
    )

# 匹配目标 Car
car_indices = np.where(
    pred_cls == CAR_CLASS_ID
)[0]

if len(car_indices) == 0:
    raise RuntimeError("YOLO did not detect any Car.")

matches = [
    (i, box_iou(pred_boxes[i], gt_box))
    for i in car_indices
]

det_idx, match_iou = max(
    matches,
    key=lambda x: x[1]
)

pred_box = pred_boxes[det_idx]
keypoints = pred_kpts[det_idx]

print("Matched detection:", det_idx)
print("Detection confidence:", pred_conf[det_idx])
print("Detection IoU with GT:", match_iou)
print("Predicted box:", pred_box)
print("Predicted keypoints:\n", keypoints)

if match_iou < 0.2:
    raise RuntimeError(
        "Detection IoU is too low. "
        "Check the prediction before reconstruction."
    )


# ============================================================
# 7. 提取预测框内 LiDAR 候选点
# ============================================================

print("\n========== 5. EXTRACT OBJECT CANDIDATES ==========", flush=True)

crop_box = expand_box(pred_box, ratio=0.12)

crop_mask = box_mask(
    u, v, crop_box
)

crop_indices = np.where(crop_mask)[0]

candidate_xyz = xyz[crop_indices]
candidate_cam = cam[crop_indices]

candidate_uv = np.column_stack([
    u[crop_indices],
    v[crop_indices]
])

candidate_depth = depth[crop_indices]

print("Expanded predicted box:", crop_box)
print("Candidate points:", len(candidate_xyz), flush=True)

# 保存候选点云
if len(candidate_xyz):
    candidate_pcd = o3d.geometry.PointCloud()
    candidate_pcd.points = o3d.utility.Vector3dVector(
        candidate_xyz
    )

    candidate_pcd.paint_uniform_color(
        [0.65, 0.65, 0.65]
    )

    o3d.io.write_point_cloud(
        str(OUT / "candidate_points.ply"),
        candidate_pcd
    )


# ============================================================
# 8. 八个关键点的局部深度估计
# ============================================================

print("\n========== 6. KEYPOINT DEPTH MATCHING ==========", flush=True)

keypoint_rows = []
keypoints_3d = []
valid_flags = []

for i, (kx, ky) in enumerate(keypoints):

    if (
        not np.isfinite([kx, ky]).all()
        or len(crop_indices) == 0
    ):
        keypoint_rows.append({
            "keypoint": i,
            "u": float(kx),
            "v": float(ky),
            "status": "missing"
        })

        valid_flags.append(False)
        continue

    distances = np.linalg.norm(
        candidate_uv - np.array([kx, ky]),
        axis=1
    )

    selected = None
    used_radius = None

    # 从小到大扩大搜索范围
    for radius in SEARCH_RADII:

        ids = np.where(
            distances <= radius
        )[0]

        if len(ids) >= 3:
            selected = ids
            used_radius = radius
            break

    # 如果没有找到至少3个点，保留最多5个最近点
    if selected is None:

        ids = np.argsort(distances)[:5]

        if len(ids) == 0 or distances[ids[0]] > SEARCH_RADII[-1]:
            keypoint_rows.append({
                "keypoint": i,
                "u": float(kx),
                "v": float(ky),
                "status": "missing"
            })

            valid_flags.append(False)
            continue

        selected = ids
        used_radius = SEARCH_RADII[-1]

    # 选取最近的若干点
    selected = selected[
        np.argsort(distances[selected])
    ][:MAX_LOCAL_POINTS]

    local_depths = candidate_depth[selected]
    local_distances = distances[selected]

    # 使用局部 LiDAR 深度中位数
    estimated_depth = float(
        np.median(local_depths)
    )

    cam_xyz, lidar_xyz = lift_to_3d(
        kx, ky, estimated_depth,
        K, R, t
    )

    count = len(selected)

    if count >= 5:
        status = "strong"
    elif count >= 3:
        status = "medium"
    else:
        status = "weak"

    keypoints_3d.append(lidar_xyz)
    valid_flags.append(True)

    row = {
        "keypoint": i,
        "u": float(kx),
        "v": float(ky),
        "depth_camera": estimated_depth,
        "local_points": count,
        "search_radius": used_radius,
        "nearest_pixel_distance": float(
            local_distances.min()
        ),
        "depth_std": float(
            np.std(local_depths)
        ),
        "camera_x": float(cam_xyz[0]),
        "camera_y": float(cam_xyz[1]),
        "camera_z": float(cam_xyz[2]),
        "lidar_x": float(lidar_xyz[0]),
        "lidar_y": float(lidar_xyz[1]),
        "lidar_z": float(lidar_xyz[2]),
        "status": status
    }

    keypoint_rows.append(row)

    print(
        f"KPT {i}: "
        f"depth={estimated_depth:.3f}m, "
        f"points={count}, "
        f"radius={used_radius}px, "
        f"status={status}"
    )


# ============================================================
# 9. 保存三维关键点 CSV
# ============================================================

print("\n========== 7. SAVE 3D KEYPOINTS ==========", flush=True)

csv_path = OUT / "keypoints_3d.csv"

fieldnames = sorted({
    key
    for row in keypoint_rows
    for key in row
})

with csv_path.open(
    "w", newline="", encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(keypoint_rows)


# ============================================================
# 10. 保存三维关键点 PLY 和边连接
# ============================================================

valid_points = np.asarray(
    keypoints_3d,
    dtype=np.float64
).reshape(-1, 3)

if len(valid_points):

    kp_pcd = o3d.geometry.PointCloud()

    kp_pcd.points = o3d.utility.Vector3dVector(
        valid_points
    )

    colors = np.tile(
        np.array([[1.0, 0.0, 0.0]]),
        (len(valid_points), 1)
    )

    kp_pcd.colors = o3d.utility.Vector3dVector(
        colors
    )

    o3d.io.write_point_cloud(
        str(OUT / "keypoints_3d.ply"),
        kp_pcd
    )

# 只有八个点全部有效才连接立方体边
if len(valid_points) == 8:

    line_set = o3d.geometry.LineSet()

    line_set.points = o3d.utility.Vector3dVector(
        valid_points
    )

    line_set.lines = o3d.utility.Vector2iVector(
        np.asarray(EDGES, dtype=np.int32)
    )

    line_set.colors = o3d.utility.Vector3dVector(
        np.tile(
            np.array([[0.0, 1.0, 0.0]]),
            (len(EDGES), 1)
        )
    )

    o3d.io.write_line_set(
        str(OUT / "cuboid_edges.ply"),
        line_set
    )


# ============================================================
# 11. 保存二维关键点匹配可视化
# ============================================================

print("\n========== 8. SAVE 2D VISUALIZATION ==========", flush=True)

canvas = img.copy()
draw = ImageDraw.Draw(canvas)

# GT box: 蓝色
draw.rectangle(
    list(gt_box),
    outline="blue",
    width=4
)

# Predicted box: 绿色
draw.rectangle(
    list(pred_box),
    outline="lime",
    width=4
)

for i, (kx, ky) in enumerate(keypoints):

    if not np.isfinite([kx, ky]).all():
        continue

    radius = 6

    draw.ellipse(
        [
            kx-radius,
            ky-radius,
            kx+radius,
            ky+radius
        ],
        fill="red",
        outline="white",
        width=2
    )

    draw.text(
        (kx+7, ky-10),
        str(i),
        fill="yellow"
    )

overlay_path = OUT / "keypoint_depth_matching.jpg"

canvas.save(overlay_path)


# ============================================================
# 12. 三维散点可视化
# ============================================================

print("\n========== 9. SAVE 3D VISUALIZATION ==========", flush=True)

if len(candidate_xyz) and len(valid_points):

    try:
        import matplotlib
        matplotlib.use("Agg")

        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(
            candidate_xyz[:, 0],
            candidate_xyz[:, 1],
            candidate_xyz[:, 2],
            s=2,
            alpha=0.18,
            label="LiDAR candidates"
        )

        ax.scatter(
            valid_points[:, 0],
            valid_points[:, 1],
            valid_points[:, 2],
            s=65,
            c="red",
            label="Lifted keypoints"
        )

        # 仅当八个点都有效时绘制完整立方体
        if len(valid_points) == 8:

            for a, b in EDGES:

                p1 = valid_points[a]
                p2 = valid_points[b]

                ax.plot(
                    [p1[0], p2[0]],
                    [p1[1], p2[1]],
                    [p1[2], p2[2]],
                    "g-",
                    linewidth=1.5
                )

        ax.set_xlabel("LiDAR X (m)")
        ax.set_ylabel("LiDAR Y (m)")
        ax.set_zlabel("LiDAR Z (m)")

        ax.set_title("Single Car 3D Keypoint Reconstruction")
        ax.legend()

        plt.tight_layout()

        fig.savefig(
            OUT / "reconstruction_3d.png",
            dpi=180
        )

        plt.close(fig)

    except Exception as exc:
        print("3D plot failed:", exc)


# ============================================================
# 13. 保存实验信息
# ============================================================

summary = {
    "image": str(IMG),
    "weight": str(WEIGHT),
    "target_gt_label_index": int(target["index"]),
    "target_class": "Car",
    "gt_box": gt_box.tolist(),
    "pred_box": pred_box.tolist(),
    "detection_confidence": float(pred_conf[det_idx]),
    "matching_iou": float(match_iou),
    "gt_box_point_count": int(target["count"]),
    "pred_box_candidate_point_count": int(len(candidate_xyz)),
    "valid_3d_keypoints": int(len(valid_points)),
    "total_3d_keypoints": 8,
    "rotation_determinant": float(np.linalg.det(R)),
    "keypoints": keypoint_rows,
    "warning": (
        "Local LiDAR depths are preliminary surface-depth estimates. "
        "They are not guaranteed to be the exact geometric corner depths. "
        "No official 3D labels were used for reconstruction."
    )
}

with (OUT / "summary.json").open(
    "w", encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2
    )

print("\n========== FINISHED ==========", flush=True)
print("Target label index:", target["index"])
print("Detection IoU:", match_iou)
print("Candidate points:", len(candidate_xyz))
print("Valid 3D keypoints:", len(valid_points), "/ 8")

print("\nOutput directory:", OUT.resolve())

for p in sorted(OUT.iterdir()):
    print(p.name, p.stat().st_size, "bytes")
