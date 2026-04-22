import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO11s with settings matched to the original yolo11n_base run."
    )
    parser.add_argument("--data", default="D:/Detect/Data_new/data.yaml", help="Dataset YAML path.")
    parser.add_argument("--model", default="yolo11s.pt", help="Model weights.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=960, help="Image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", default="0", help="CUDA device id or cpu.")
    parser.add_argument("--workers", type=int, default=1, help="Number of dataloader workers.")
    parser.add_argument("--project", default="D:/Detect/final_runs", help="Output folder.")
    parser.add_argument("--name", default="yolo11s_base_datanew", help="Run name.")
    parser.add_argument("--temp-dir", default="D:/temp", help="Temporary directory for Windows/Python cache files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    temp_dir = Path(args.temp_dir).resolve()
    temp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = str(temp_dir)
    os.environ["TMP"] = str(temp_dir)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Missing 'ultralytics'. Install with: pip install ultralytics")
        return

    model = YOLO(args.model)

    train_kwargs = {
        "data": str(Path(args.data).resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "patience": 20,
        "save_period": 10,
        "cache": False,
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
        "plots": True,
        "save": True,
    }

    print("--- RUNNING TRAIN WITH YOLO11N_BASE-MATCHED SETTINGS ---")
    print(f"Target: {args.name}, Epochs: {args.epochs}, Batch: {args.batch}")
    print(f"Data: {args.data}")
    print(f"Model: {args.model}")
    print(f"Temp dir: {temp_dir}")
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
