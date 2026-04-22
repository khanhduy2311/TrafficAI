import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


SCRIPT_ROOT = Path(__file__).resolve().parent
TRAIN_ROOT = SCRIPT_ROOT.parents[1]
CONFIG_DIR = TRAIN_ROOT / "configs"


def resolve_data_config() -> str:
    for candidate in (
        CONFIG_DIR / "dataset_balanced.yaml",
        CONFIG_DIR / "dataset.yaml",
        CONFIG_DIR / "dataset.example.yaml",
    ):
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError("No dataset config found in training/configs")


def measure_latency(model, img_size=640, num_warmup=10, num_runs=100, device="cpu"):
    dummy = torch.zeros(1, 3, img_size, img_size)
    use_cuda = device != "cpu" and torch.cuda.is_available()
    if use_cuda:
        dummy = dummy.cuda()

    for _ in range(num_warmup):
        model(dummy, verbose=False)

    times = []
    if use_cuda:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        for _ in range(num_runs):
            starter.record()
            model(dummy, verbose=False)
            ender.record()
            torch.cuda.synchronize()
            times.append(starter.elapsed_time(ender))
    else:
        for _ in range(num_runs):
            start_time = time.perf_counter()
            model(dummy, verbose=False)
            times.append((time.perf_counter() - start_time) * 1000)

    latency_ms = float(np.mean(times))
    return round(latency_ms, 2), round(1000.0 / latency_ms, 2)


def find_weights(root: str):
    weight_paths = sorted(Path(root).rglob("*.pt"))
    if not weight_paths:
        print(f"No .pt files found in {root}")
    return weight_paths


def eval_single(weight_path: Path, device: str, args):
    print(f"\n{'=' * 60}")
    print(f"Model : {weight_path}")
    print(f"Device: {device}")
    print(f"{'=' * 60}")

    model = YOLO(str(weight_path))
    results = model.val(
        data=args.data,
        split="test",
        imgsz=args.imgsz,
        device=device,
        conf=args.conf,
        iou=args.iou,
        verbose=False,
    )

    box = results.box
    precision = float(box.mp)
    recall = float(box.mr)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    map50 = float(box.map50)
    map50_95 = float(box.map)

    latency_ms, fps = measure_latency(
        model,
        img_size=args.imgsz,
        num_warmup=args.warmup,
        num_runs=args.runs,
        device=device,
    )

    label = f"{weight_path.parent.parent.name} / {weight_path.stem}"
    return {
        "run": weight_path.parent.parent.name,
        "weight": weight_path.stem,
        "label": label,
        "Precision(%)": round(precision * 100, 2),
        "Recall(%)": round(recall * 100, 2),
        "F1(%)": round(f1 * 100, 2),
        "mAP@0.5(%)": round(map50 * 100, 2),
        "mAP@0.5:0.95(%)": round(map50_95 * 100, 2),
        "Latency(ms)": latency_ms,
        "FPS": fps,
    }


def print_table(rows):
    columns = [
        ("label", "Model", 25),
        ("Precision(%)", "Precision(%)", 12),
        ("Recall(%)", "Recall(%)", 10),
        ("F1(%)", "F1(%)", 8),
        ("mAP@0.5(%)", "mAP@0.5", 10),
        ("mAP@0.5:0.95(%)", "mAP@.5:.95", 12),
        ("Latency(ms)", "Latency(ms)", 13),
        ("FPS", "FPS", 8),
    ]
    header = " ".join(f"{title:>{width}}" for _, title, width in columns)
    separator = "-" * len(header)
    print(f"\n{separator}\n{header}\n{separator}")
    for row in rows:
        print(" ".join(f"{str(row[key]):>{width}}" for key, _, width in columns))
    print(separator)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO checkpoints on the test split")
    parser.add_argument("--device", default="0", help="GPU id such as 0, or cpu")
    parser.add_argument("--data", default=resolve_data_config(), help="Path to dataset yaml")
    parser.add_argument(
        "--weights",
        default=str(TRAIN_ROOT / "result_train_fisheye"),
        help="Directory containing trained checkpoints",
    )
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument(
        "--output",
        default="eval_results.csv",
        help="CSV file used to save evaluation results",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\nScanning weights from: {args.weights}")
    print(f"Dataset config      : {args.data}")
    print(f"Device              : {args.device}\n")

    weight_paths = find_weights(args.weights)
    if not weight_paths:
        return

    all_results = []
    for weight_path in weight_paths:
        try:
            result = eval_single(weight_path, args.device, args)
            all_results.append(result)
            print(
                f"\nDone {result['label']}: "
                f"mAP50={result['mAP@0.5(%)']:.2f}% "
                f"mAP50-95={result['mAP@0.5:0.95(%)']:.2f}% "
                f"Latency={result['Latency(ms)']}ms FPS={result['FPS']}"
            )
        except Exception as exc:
            print(f"Failed to evaluate {weight_path}: {exc}")

    if not all_results:
        return

    print_table(all_results)

    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nSaved evaluation results to {args.output}")


if __name__ == "__main__":
    main()
