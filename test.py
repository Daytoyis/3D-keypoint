from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

# ===============================
# 1. 路径配置
# ===============================
weights = "runs/pose/train/weights/best.pt"
image_path = "/mnt/sdb/lys/yolo11/assets/000041.png"

output_dir = Path("runs/single_image_prediction")
output_dir.mkdir(parents=True, exist_ok=True)

# ===============================
# 2. 加载模型
# ===============================
model = YOLO(weights)

print("模型任务:", model.task)
print("类别:", model.names)
print("关键点配置:", model.model.yaml.get("kpt_shape"))

# ===============================
# 3. 单张图片预测
# ===============================
results = model.predict(
    source=image_path,
    imgsz=640,
    conf=0.15,
    device=0,
    save=False,
    verbose=False
)

r = results[0]

img = cv2.imread(image_path)

# ===============================
# 4. 获取预测结果
# ===============================
if r.boxes is None or r.keypoints is None:
    print("没有检测到目标或关键点")
else:
    boxes = r.boxes.xyxy.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    keypoints = r.keypoints.xy.cpu().numpy()

    kp_conf = (
        r.keypoints.conf.cpu().numpy()
        if r.keypoints.conf is not None else None
    )

    # 八角点连接关系
    connections = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7)
    ]

    for i, (box, cls, score, kpts) in enumerate(
        zip(boxes, classes, scores, keypoints)
    ):
        x1,y1,x2,y2 = map(int, box)

        cls_name = model.names[int(cls)]

        # 绘制检测框
        cv2.rectangle(
            img, (x1,y1), (x2,y2),
            (0,255,0), 2
        )

        cv2.putText(
            img, f"{cls_name} {score:.2f}",
            (x1, max(y1-8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0,255,0), 2
        )

        # 绘制八角点连接线
        for a,b in connections:
            pa = tuple(np.round(kpts[a]).astype(int))
            pb = tuple(np.round(kpts[b]).astype(int))
            cv2.line(img, pa, pb, (255,100,0), 2)

        # 绘制八个带编号的关键点
        for j, (x,y) in enumerate(kpts):
            x,y = int(x),int(y)

            cv2.circle(img, (x,y), 5, (0,0,255), -1)

            cv2.putText(
                img, str(j),
                (x+5,y-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0,0,255), 2
            )

        print(f"\n目标 {i+1}: {cls_name}")
        print(f"检测置信度: {score:.4f}")

        for j, (x,y) in enumerate(kpts):
            if kp_conf is not None:
                print(
                    f"Keypoint {j}: "
                    f"x={x:.2f}, y={y:.2f}, "
                    f"conf={kp_conf[i][j]:.4f}"
                )
            else:
                print(
                    f"Keypoint {j}: "
                    f"x={x:.2f}, y={y:.2f}"
                )

# ===============================
# 5. 保存结果
# ===============================
output_path = output_dir / (
    Path(image_path).stem + "_8keypoints.jpg"
)

cv2.imwrite(str(output_path), img)

print("\n预测完成！")
print("结果图片:", output_path.resolve())