"""YOLO11 3D Cuboid Keypoint Detection Experiments."""

from __future__ import annotations

# 数据增强超参数
AUGMENTATION: dict[str, float | str] = {
    "hsv_h": 0.01315,
    "hsv_s": 0.35348,
    "hsv_v": 0.19383,
    "degrees": 0.00012,
    "translate": 0.27484,
    "scale": 0.95,
    "shear": 0.00136,
    "perspective": 0.00074,
    "flipud": 0.00653,
    "fliplr": 0.30393,
    "bgr": 0.0,
    "mosaic": 0.99182,
    "mixup": 0.42713,
    "cutmix": 0.00082,
    "copy_paste": 0.40413,
    "copy_paste_mode": "flip",
    "auto_augment": "randaugment",
    "erasing": 0.4,
}


# 模型规模
SCALES_ALL: list[str] = ["n", "s", "m", "l", "x"]
SCALES_V9: list[str] = ["t", "s", "m", "c", "e"]


# 3. 不同模型规模的 Batch 配置
def _make_entries(
    arch: str,
    variant: str,
    scales: list[str],
    imgsz: int,
    batch_per_device: int,
    epochs: int,
    *,
    pretrained: bool,
) -> list[dict]:
    """Return a list of experiment dicts for *arch* across *scales*."""
    return [
        {
            "model_cfg": f"{arch}{s}-pose{variant}.yaml",
            "weights": f"{arch}{s}.pt" if pretrained else None,
            "imgsz": imgsz,
            "batch_per_device": batch_per_device,
            "epochs": epochs,
        }
        for s in scales
    ]


def pretrained(
    arch: str, variant: str, scales: list[str], imgsz: int, batch_per_device: int, epochs: int,
) -> list[dict]:
    """Experiment entries that fine-tune from COCO-pretrained weights."""
    return _make_entries(arch, variant, scales, imgsz, batch_per_device, epochs, pretrained=True)


def scratch(
    arch: str, variant: str, scales: list[str], imgsz: int, batch_per_device: int, epochs: int,
) -> list[dict]:
    """Experiment entries that train from scratch (random init)."""
    return _make_entries(arch, variant, scales, imgsz, batch_per_device, epochs, pretrained=False)


# ---------------------------------------------------------------------------
# Full experiment list — order matches the paper's ablation tables
# ---------------------------------------------------------------------------
# 训练实验列表
EXPERIMENTS = [
    # 实验1：标准 YOLO11n-Pose
    {
        "model_cfg": "cfg/models/yolo11-pose.yaml",
        "weights": "yolo11n.pt",
        "imgsz": 640,
        "batch_per_device": 8,
        "epochs": 100,
    },

    # 实验2：UrbanOmniDetect YOLO11n-Pose-P2
    {
        "model_cfg": "cfg/models/yolo11-pose-p2.yaml",
        "weights": "yolo11n.pt",
        "imgsz": 640,
        "batch_per_device": 8,
        "epochs": 100,
    },
]
EXPERIMENTS: list[dict] = [
    # ── High-resolution pretrained ────────────────────────────────────────
    #{"model_cfg": "yolo11n-pose-p2.yaml", "weights": "yolo11n.pt", "imgsz": 3840, "batch_per_device": 4,  "epochs": 100},
    #{"model_cfg": "yolo11n-pose-p2.yaml", "weights": "yolo11n.pt", "imgsz": 1920, "batch_per_device": 8,  "epochs": 100},
    #{"model_cfg": "yolo11x-pose-p1.yaml", "weights": "yolo11x.pt", "imgsz": 640,  "batch_per_device": 24, "epochs": 200},
    #{"model_cfg": "yolo11x-pose-p2.yaml", "weights": "yolo11x.pt", "imgsz": 1280, "batch_per_device": 8,  "epochs": 100},

    # ── YOLO11 P2 pretrained (n/s/m/l) ───────────────────────────────────
    #*pretrained("yolo11", "-p2", ["n", "s", "m", "l"], 640, 32, 100),
    *pretrained("yolo11", "-p2", ["n"], 640, 16, 100)

  # ── YOLO11x P2 pretrained (additional 640 resolution experiment) ────
    #{"model_cfg": "yolo11x-pose-p2.yaml", "weights": "yolo11x.pt", "imgsz": 640, "batch_per_device": 8, "epochs": 100},

    # ── YOLO11 P6 pretrained (single scale) ─────────────────
    #{"model_cfg": "yolo11x-pose-p6.yaml", "weights": "yolo11x.pt", "imgsz": 640, "batch_per_device": 32, "epochs": 100},
    
    # ── YOLO11 pretrained (all scales, base head) ────────────────────────
    #*pretrained("yolo11", "", SCALES_ALL, 640, 32, 100),

    # ── YOLO11 base head from scratch (n/s/m/l/x) ─────────────────────────
    #*scratch("yolo11", "", SCALES_ALL, 640, 32, 100),
]
