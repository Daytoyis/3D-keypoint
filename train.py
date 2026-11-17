
from __future__ import annotations

import argparse
import logging
import sys
import time
import re
from pathlib import Path

from ultralytics import YOLO

from cfg.experiments import AUGMENTATION, EXPERIMENTS

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
#设置训练环境
ROOT = Path(__file__).resolve().parent
#训练数据集配置文件
DEFAULT_DATASET = (
    "/mnt/sdb/lys/yolo11/datasets/"    
    "YOLO11_3D_Pose_Full/data.yaml"
    )
DEFAULT_DEVICES = [0]
# 训练轮数
DEFAULT_PATIENCE = 1000

log = logging.getLogger("yolo11.train")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
#寻找模型配置文件
def _resolve_model_cfg(name: str) -> Path:
    direct = ROOT / name
    if direct.is_file():
        return direct

    under_cfg = ROOT / "cfg" / "models" / name
    if under_cfg.is_file():
        return under_cfg

    # Strip the scale letter (e.g. yolo11n-pose-p2.yaml -> yolo11-pose-p2.yaml)
    # because cfg/models/ stores the generic template.
    return under_cfg

    


def _validate_dataset(path: str) -> None:
    """Abort early if the dataset YAML is missing."""
    resolved = Path(path) if Path(path).is_absolute() else ROOT / path
    if not resolved.is_file():
        log.error("Dataset config not found: %s", resolved)
        sys.exit(1)


def _validate_experiments(experiments: list[dict]) -> None:
    """Warn about model configs that cannot be found locally."""
    missing = []
    for exp in experiments:
        cfg_path = _resolve_model_cfg(exp["model_cfg"])
        if not cfg_path.is_file():
            missing.append(exp["model_cfg"])
    if missing:
        unique = sorted(set(missing))
        log.warning(
            "%d model config(s) not found locally (Ultralytics may still "
            "resolve them at runtime): %s",
            len(unique),
            ", ".join(unique),
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
# 启动训练
def run_experiment(
    model_cfg: str,
    weights: str | None,
    imgsz: int,
    batch: int,
    epochs: int,
    data: str,
    devices: list[int],
    patience: int = DEFAULT_PATIENCE,
) -> None:
    """Instantiate a YOLO model and launch a single training run."""
    cfg_path = _resolve_model_cfg(model_cfg)

    model = YOLO(cfg_path)

    if weights is not None:
        model.load(weights)
   

    model.train(
        data=data,
        imgsz=imgsz,
        device=devices,
        resume=False,
        batch=batch,
        epochs=epochs,
        patience=patience,
        **AUGMENTATION,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
# 命令行参数
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO11 3D Pose Estimation — launch training experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Experiment selection
    sel = parser.add_argument_group("experiment selection")
    sel.add_argument(
        "--filter", type=str, default=None,
        help="Only run experiments whose model_cfg contains this substring",
    )
    sel.add_argument(
        "--start-from", type=int, default=1, metavar="N",
        help="Skip experiments before index N (1-based, default: 1)",
    )

    # Hardware / paths
    hw = parser.add_argument_group("hardware & paths")
    hw.add_argument(
        "--devices", type=int, nargs="+", default=None, metavar="ID",
        help="GPU device IDs (default: 0-7)",
    )
    hw.add_argument(
        "--data", type=str, default=None, metavar="YAML",
        help=f"Dataset config path (default: {DEFAULT_DATASET})",
    )

    # Training overrides
    tr = parser.add_argument_group("training overrides")
    tr.add_argument(
        "--epochs", type=int, default=None,
        help="Override epoch count for every experiment",
    )
    tr.add_argument(
        "--patience", type=int, default=DEFAULT_PATIENCE,
        help=f"Early-stopping patience (default: {DEFAULT_PATIENCE})",
    )

    # Behaviour
    beh = parser.add_argument_group("behaviour")
    beh.add_argument(
        "--dry-run", action="store_true",
        help="Print the experiment plan without training",
    )
    beh.add_argument(
        "--stop-on-error", action="store_true",
        help="Abort the run on the first failed experiment",
    )
    beh.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG-level logging",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _format_duration(seconds: float) -> str:
    h, remainder = divmod(int(seconds), 3600)
    m, s = divmod(remainder, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"

# 训练主函数
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    devices = args.devices if args.devices is not None else DEFAULT_DEVICES
    data = args.data if args.data is not None else DEFAULT_DATASET
    num_devices = len(devices)

    _validate_dataset(data)

    # --- Filter & slice ---------------------------------------------------
    experiments = list(EXPERIMENTS)
    if args.filter:
        experiments = [e for e in experiments if args.filter in e["model_cfg"]]

    if not experiments:
        log.error("No experiments match the given --filter.")
        sys.exit(1)

    if args.start_from > 1:
        experiments = experiments[args.start_from - 1:]
        log.info("Skipping to experiment #%d", args.start_from)

    _validate_experiments(experiments)

    # --- Run --------------------------------------------------------------
    total = len(experiments)
    log.info("Queued %d experiment(s) on devices %s", total, devices)

    failed: list[tuple[int, str, str]] = []
    wall_start = time.monotonic()

    for i, exp in enumerate(experiments, 1):
        model_cfg = exp["model_cfg"]
        weights = exp["weights"]
        imgsz = exp["imgsz"]
        epochs = args.epochs if args.epochs is not None else exp["epochs"]
        batch = num_devices * exp["batch_per_device"]
        tag = f"pretrained={weights}" if weights else "from-scratch"

        log.info(
            "[%d/%d] %s  imgsz=%d  batch=%d  epochs=%d  (%s)",
            i, total, model_cfg, imgsz, batch, epochs, tag,
        )

        if args.dry_run:
            continue

        t0 = time.monotonic()
        try:
            run_experiment(
                model_cfg, weights, imgsz, batch, epochs, data, devices,
                patience=args.patience,
            )
            elapsed = _format_duration(time.monotonic() - t0)
            log.info("[%d/%d] Finished in %s", i, total, elapsed)
        except Exception:
            elapsed = _format_duration(time.monotonic() - t0)
            log.exception("[%d/%d] FAILED after %s", i, total, elapsed)
            failed.append((i, model_cfg, tag))
            if args.stop_on_error:
                log.error("Aborting (--stop-on-error).")
                sys.exit(1)

    # --- Summary ----------------------------------------------------------
    wall_elapsed = _format_duration(time.monotonic() - wall_start)

    if args.dry_run:
        log.info("Dry run complete — %d experiment(s) listed.", total)
        return

    if failed:
        log.warning(
            "%d / %d experiment(s) failed (wall time %s):",
            len(failed), total, wall_elapsed,
        )
        for idx, cfg, tag in failed:
            log.warning("  #%d  %s  (%s)", idx, cfg, tag)
        sys.exit(1)

    log.info("All %d experiment(s) completed successfully (wall time %s).", total, wall_elapsed)


if __name__ == "__main__":
    main()
