import sys
import gc
import json
from pathlib import Path

sys.path.insert(0, "/mnt/sdb/lys/yolo11/ultralytics_src")

import torch
from ultralytics import YOLO

ROOT = Path("/mnt/sdb/lys/yolo11")
DATA = ROOT / "datasets/YOLO11_3D_Pose_3Source"

MODELS = {
    "VPCL": ROOT / "runs/pose/train-2/weights/best.pt",
    "Baseline": ROOT / "runs/pose/train-3/weights/best.pt",
}

OUT = ROOT / "runs/test_compare"

# 自动检查测试集每张图片的标注数量
label_files = list((DATA / "labels/test").glob("*.txt"))

if not label_files:
    raise RuntimeError("没有找到测试集标签")

max_gt = max(
    sum(1 for line in f.open() if line.strip())
    for f in label_files
)

max_det = max(300, max_gt + 50)

print("=" * 60)
print("TEST SET EVALUATION")
print("Test images:", len(label_files))
print("Maximum GT objects per image:", max_gt)
print("max_det:", max_det)
print("=" * 60)

for name, weight in MODELS.items():

    if not weight.exists():
        raise FileNotFoundError(weight)

    print(f"\nEvaluating: {name}")
    print("Weights:", weight)

    model = YOLO(str(weight))

    metrics = model.val(
        data=str(DATA / "data.yaml"),
        split="test",

        imgsz=640,
        batch=16,
        device=0,
        workers=8,

        conf=0.001,
        iou=0.7,
        max_det=max_det,

        augment=False,
        plots=True,
        verbose=True,

        project=str(OUT),
        name=name,
        exist_ok=True,
    )

    # 保存总体指标
    results = getattr(metrics, "results_dict", {})

    summary = {}
    for k, v in results.items():
        try:
            summary[k] = float(v)
        except (TypeError, ValueError):
            summary[k] = str(v)

    save_dir = Path(metrics.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\n{name} evaluation finished.")
    print(summary)

    del model
    del metrics
    gc.collect()
    torch.cuda.empty_cache()

print("\nALL TEST EVALUATIONS COMPLETED.")
