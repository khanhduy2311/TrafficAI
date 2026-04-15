import argparse
import os
from pathlib import Path


PRESET_CHOICES = (
    "base_exact",
    "balanced",
    "stable",
    "speed",
    "strict_tiny",
    "optimized_legacy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified YOLO trainer for this project.")
    parser.add_argument("--data", default="../../configs/data.yaml", help="Dataset YAML path.")
    parser.add_argument("--model", default="yolo11s.pt", help="Starting checkpoint.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument(
        "--preset",
        choices=PRESET_CHOICES,
        default="base_exact",
        help="Training preset. Use base_exact for the cleanest baseline comparison.",
    )
    parser.add_argument("--imgsz", type=int, default=960, help="Training image size.")
    parser.add_argument("--batch", type=int, default=None, help="Batch size override.")
    parser.add_argument("--device", default="0", help="CUDA device id or cpu.")
    parser.add_argument("--workers", type=int, default=None, help="Dataloader workers override.")
    parser.add_argument("--project", default="../../final_runs", help="Output project folder.")
    parser.add_argument("--name", default=None, help="Run name override.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience.")
    parser.add_argument("--freeze", type=int, default=0, help="Number of layers to freeze.")
    parser.add_argument("--cache", choices=("false", "ram", "disk"), default="false", help="Dataset cache mode.")
    parser.add_argument("--optimizer", default="auto", help="Ultralytics optimizer mode.")
    parser.add_argument("--multi-scale", action="store_true", help="Enable multi-scale training.")
    parser.add_argument("--save-period", type=int, default=None, help="Checkpoint save period override.")
    return parser.parse_args()


def preset_defaults() -> dict:
    return {
        "base_exact": {
            "batch": 16,
            "workers": 0,
            "save_period": 10,
            "name": "yolo11s_base_datanew_balanced",
            "train": {
                "pretrained": True,
                "optimizer": "auto",
                "verbose": True,
                "seed": 0,
                "deterministic": True,
                "cos_lr": True,
                "close_mosaic": 10,
                "amp": True,
                "lr0": 0.01,
                "lrf": 0.01,
                "momentum": 0.937,
                "weight_decay": 0.0005,
                "warmup_epochs": 3.0,
                "warmup_momentum": 0.8,
                "warmup_bias_lr": 0.1,
                "box": 7.5,
                "cls": 0.5,
                "dfl": 1.5,
                "mosaic": 0.5,
                "degrees": 5.0,
                "shear": 2.0,
                "translate": 0.1,
                "scale": 0.5,
                "perspective": 0.0,
                "mixup": 0.0,
                "cutmix": 0.0,
                "copy_paste": 0.0,
                "fliplr": 0.5,
                "flipud": 0.0,
                "hsv_h": 0.015,
                "hsv_s": 0.7,
                "hsv_v": 0.4,
                "auto_augment": "randaugment",
                "erasing": 0.4,
                "overlap_mask": True,
                "mask_ratio": 4,
            },
        },
        "balanced": {
            "batch": 16,
            "workers": 1,
            "save_period": 10,
            "name": "yolo11s_balanced_datanew",
            "train": {
                "pretrained": True,
                "optimizer": "auto",
                "verbose": True,
                "cos_lr": True,
                "amp": True,
                "lr0": 0.01,
                "lrf": 0.01,
                "close_mosaic": 10,
                "box": 7.5,
                "cls": 0.5,
                "dfl": 1.5,
                "mosaic": 0.5,
                "degrees": 5.0,
                "shear": 2.0,
                "translate": 0.1,
                "scale": 0.5,
                "perspective": 0.0,
                "mixup": 0.0,
                "copy_paste": 0.0,
                "fliplr": 0.5,
                "flipud": 0.0,
                "hsv_h": 0.015,
                "hsv_s": 0.7,
                "hsv_v": 0.4,
            },
        },
        "stable": {
            "batch": 16,
            "workers": 1,
            "save_period": 10,
            "name": "yolo11s_stable_datanew",
            "train": {
                "pretrained": True,
                "optimizer": "auto",
                "verbose": True,
                "cos_lr": True,
                "amp": True,
                "lr0": 0.01,
                "lrf": 0.01,
                "close_mosaic": 12,
                "box": 7.5,
                "cls": 0.5,
                "dfl": 1.5,
                "mosaic": 0.3,
                "degrees": 3.0,
                "shear": 1.0,
                "translate": 0.08,
                "scale": 0.4,
                "perspective": 0.0,
                "mixup": 0.0,
                "copy_paste": 0.0,
                "fliplr": 0.5,
                "flipud": 0.0,
                "hsv_h": 0.015,
                "hsv_s": 0.7,
                "hsv_v": 0.4,
            },
        },
        "speed": {
            "batch": 16,
            "workers": 1,
            "save_period": 10,
            "name": "yolo11s_speed_datanew",
            "train": {
                "pretrained": True,
                "optimizer": "auto",
                "verbose": True,
                "cos_lr": True,
                "amp": True,
                "lr0": 0.01,
                "lrf": 0.01,
                "close_mosaic": 8,
                "box": 7.5,
                "cls": 0.5,
                "dfl": 1.5,
                "mosaic": 0.4,
                "degrees": 4.0,
                "shear": 1.5,
                "translate": 0.1,
                "scale": 0.45,
                "perspective": 0.0,
                "mixup": 0.0,
                "copy_paste": 0.0,
                "fliplr": 0.5,
                "flipud": 0.0,
                "hsv_h": 0.015,
                "hsv_s": 0.7,
                "hsv_v": 0.4,
            },
        },
        "strict_tiny": {
            "batch": 16,
            "workers": 1,
            "save_period": 10,
            "name": "yolo11s_strict_tiny_datanew",
            "train": {
                "pretrained": True,
                "optimizer": "auto",
                "verbose": True,
                "cos_lr": True,
                "amp": True,
                "lr0": 0.01,
                "lrf": 0.01,
                "close_mosaic": 10,
                "box": 7.5,
                "cls": 1.5,
                "dfl": 1.5,
                "mosaic": 1.0,
                "degrees": 15.0,
                "shear": 4.0,
                "translate": 0.2,
                "scale": 0.7,
                "perspective": 0.0002,
                "mixup": 0.15,
                "copy_paste": 0.3,
                "fliplr": 0.5,
                "flipud": 0.0,
                "hsv_h": 0.015,
                "hsv_s": 0.7,
                "hsv_v": 0.4,
            },
        },
        "optimized_legacy": {
            "batch": 8,
            "workers": 1,
            "save_period": 5,
            "name": "yolo11s_optimized_legacy_datanew",
            "train": {
                "pretrained": True,
                "optimizer": "auto",
                "verbose": True,
                "cos_lr": True,
                "amp": True,
                "lr0": 0.01,
                "lrf": 0.1,
                "box": 7.5,
                "cls": 1.5,
                "dfl": 1.5,
                "mosaic": 1.0,
                "degrees": 5.0,
                "shear": 2.0,
                "translate": 0.1,
                "scale": 0.3,
                "perspective": 0.0002,
                "mixup": 0.15,
                "copy_paste": 0.3,
                "fliplr": 0.5,
                "flipud": 0.3,
            },
        },
    }


def build_train_kwargs(args: argparse.Namespace) -> tuple[dict, str]:
    presets = preset_defaults()
    preset = presets[args.preset]
    workers_override = args.workers if args.workers is not None else preset["workers"]
    workers = min(workers_override, os.cpu_count() or workers_override)
    cache_mode = {"false": False, "ram": True, "disk": "disk"}[args.cache]
    batch = args.batch if args.batch is not None else preset["batch"]
    save_period = args.save_period if args.save_period is not None else preset["save_period"]
    run_name = args.name or preset["name"]

    train_kwargs = {
        "data": str(Path(args.data).resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": batch,
        "device": args.device,
        "workers": workers,
        "project": args.project,
        "name": run_name,
        "patience": args.patience,
        "freeze": args.freeze,
        "cache": cache_mode,
        "multi_scale": args.multi_scale,
        "plots": True,
        "save": True,
        "save_period": save_period,
        "rect": False,
        "single_cls": False,
        **preset["train"],
    }

    if args.optimizer != "auto":
        train_kwargs["optimizer"] = args.optimizer

    return train_kwargs, run_name


def write_metrics_log(args: argparse.Namespace, run_name: str, results) -> None:
    outputs_dir = Path("outputs/logs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    best_weights = Path(args.project) / run_name / "weights" / "best.pt"
    p = results.box.mp
    r = results.box.mr
    f1 = 2 * (p * r) / (p + r + 1e-16)
    log_file = outputs_dir / f"{run_name}_best_metrics.txt"
    with log_file.open("w", encoding="utf-8") as f:
        f.write(f"model: {args.model}\n")
        f.write(f"data: {args.data}\n")
        f.write(f"preset: {args.preset}\n")
        f.write(f"epochs_run: {args.epochs}\n")
        f.write(f"precision: {p:.5f}\n")
        f.write(f"recall: {r:.5f}\n")
        f.write(f"f1_score: {f1:.5f}\n")
        f.write(f"mAP50: {results.box.map50:.5f}\n")
        f.write(f"mAP50_95: {results.box.map:.5f}\n")
        f.write(f"best_pt: {best_weights.resolve()}\n")
    print(f"\nSaved metrics log: {log_file}")


def main() -> None:
    args = parse_args()
    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing dependency 'ultralytics'. Install with: pip install -r requirements.txt") from exc

    train_kwargs, run_name = build_train_kwargs(args)
    model = YOLO(args.model)

    print(f"Starting train: preset={args.preset}, model={args.model}, data={args.data}")
    print(f"Run name: {run_name}, batch={train_kwargs['batch']}, imgsz={args.imgsz}")
    results = model.train(**train_kwargs)

    try:
        write_metrics_log(args, run_name, results)
    except Exception as exc:
        print(f"\nCould not write metrics log automatically: {exc}")
        print(f"Training artifacts are still available at: {args.project}/{run_name}")


if __name__ == "__main__":
    main()
