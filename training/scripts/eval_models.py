# eval_models.py
import os
import time
import torch
import numpy as np
import csv
import argparse
from pathlib import Path
from ultralytics import YOLO

# ===================== CẤU HÌNH MẶC ĐỊNH =====================
DATA_YAML      = "/AIClub_NAS/core_baotg/thuyntn/Datasets/CV/data_fisheye/dataset.yaml"
WEIGHTS_ROOT   = "/AIClub_NAS/core_baotg/thuyntn/PlugIR_Workspace/training_scripts/runs/result_train_fisheye/run_yolo11_final/weights"   # Tự scan *.pt trong subfolders
IMG_SIZE       = 960
CONF_THRES     = 0.25
IOU_THRES      = 0.5
NUM_WARMUP     = 10
NUM_LATENCY    = 100
OUTPUT_CSV     = "eval_results_yolo11_final.csv"

# ===================== ĐO LATENCY =====================
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
        ender   = torch.cuda.Event(enable_timing=True)
        for _ in range(num_runs):
            starter.record()
            model(dummy, verbose=False)
            ender.record()
            torch.cuda.synchronize()
            times.append(starter.elapsed_time(ender))
    else:
        for _ in range(num_runs):
            t0 = time.perf_counter()
            model(dummy, verbose=False)
            times.append((time.perf_counter() - t0) * 1000)

    latency_ms = float(np.mean(times))
    return round(latency_ms, 2), round(1000.0 / latency_ms, 2)


# ===================== TÌM TẤT CẢ FILE .PT =====================
def find_weights(root: str):
    """
    Scan toàn bộ subfolders, trả về list path .pt.
    Ưu tiên best.pt, sau đó last.pt, bỏ qua các file khác nếu muốn.
    """
    root_path = Path(root)
    all_pts = sorted(root_path.rglob("*.pt"))
    if not all_pts:
        print(f"[ERROR] Không tìm thấy file .pt nào trong '{root}'")
    for p in all_pts:
        print(f"  Tìm thấy: {p}")
    return all_pts


# ===================== EVAL 1 MODEL =====================
def eval_single(weight_path: Path, device: str):
    print(f"\n{'='*65}")
    print(f"  Model : {weight_path}")
    print(f"  Device: {device}")
    print(f"{'='*65}")

    model = YOLO(str(weight_path))

    # -- Validation trên tập test (dùng split="test" theo test.txt trong yaml) --
    results = model.val(
        data    = DATA_YAML,
        split   = "test",          # <-- dùng test.txt
        imgsz   = IMG_SIZE,
        device  = device,
        conf    = CONF_THRES,
        iou     = IOU_THRES,
        verbose = False,
    )

    box        = results.box
    precision  = float(box.mp)
    recall     = float(box.mr)
    f1         = 2 * precision * recall / (precision + recall + 1e-9)
    map50      = float(box.map50)
    map50_95   = float(box.map)

    latency_ms, fps = measure_latency(
        model, IMG_SIZE, NUM_WARMUP, NUM_LATENCY, device
    )

    # Lấy tên dễ đọc: "run_rtdetr / best"
    label = f"{weight_path.parent.parent.name} / {weight_path.stem}"

    return {
        "run"           : weight_path.parent.parent.name,
        "weight"        : weight_path.stem,
        "label"         : label,
        "Precision(%)"  : round(precision  * 100, 2),
        "Recall(%)"     : round(recall     * 100, 2),
        "F1(%)"         : round(f1         * 100, 2),
        "mAP@0.5(%)"    : round(map50      * 100, 2),
        "mAP@0.5:0.95(%)": round(map50_95  * 100, 2),
        "Latency(ms)"   : latency_ms,
        "FPS"           : fps,
    }


# ===================== IN BẢNG =====================
def print_table(rows):
    cols = [
        ("label",            "Model",            25),
        ("Precision(%)",     "Precision(%)",     12),
        ("Recall(%)",        "Recall(%)",        10),
        ("F1(%)",            "F1(%)",            8),
        ("mAP@0.5(%)",       "mAP@0.5",          10),
        ("mAP@0.5:0.95(%)",  "mAP@.5:.95",       12),
        ("Latency(ms)",      "Latency(ms)",      13),
        ("FPS",              "FPS",              8),
    ]
    header = " ".join(f"{h:>{w}}" for _, h, w in cols)
    sep    = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for r in rows:
        line = " ".join(
            f"{str(r[k]):>{w}}" for k, _, w in cols
        )
        print(line)
    print(sep)


# ===================== MAIN =====================
def main():
    # ← Khai báo global TRƯỚC TIÊN, trước mọi thứ khác
    global DATA_YAML, IMG_SIZE, CONF_THRES, IOU_THRES

    parser = argparse.ArgumentParser(description="Eval YOLO models")
    parser.add_argument("--device",   default="0",
                        help="GPU id (e.g. 0, 1) hoặc 'cpu'")
    parser.add_argument("--data",     default=DATA_YAML,
                        help="Đường dẫn dataset.yaml")
    parser.add_argument("--weights",  default=WEIGHTS_ROOT,
                        help="Thư mục gốc chứa các runs")
    parser.add_argument("--imgsz",    type=int, default=IMG_SIZE)
    parser.add_argument("--conf",     type=float, default=CONF_THRES)
    parser.add_argument("--iou",      type=float, default=IOU_THRES)
    args = parser.parse_args()

    DATA_YAML  = args.data
    IMG_SIZE   = args.imgsz
    CONF_THRES = args.conf
    IOU_THRES  = args.iou

    print(f"\n📂 Scan weights từ : {args.weights}")
    print(f"📄 Dataset yaml    : {DATA_YAML}")
    print(f"🖥️  Device          : {args.device}\n")

    pt_files = find_weights(args.weights)
    if not pt_files:
        return

    all_results = []
    for pt in pt_files:
        try:
            r = eval_single(pt, args.device)
            all_results.append(r)
            print(f"\n✅ {r['label']}: "
                  f"mAP50={r['mAP@0.5(%)']:.2f}%  "
                  f"mAP50-95={r['mAP@0.5:0.95(%)']:.2f}%  "
                  f"Latency={r['Latency(ms)']}ms  FPS={r['FPS']}")
        except Exception as e:
            print(f"[WARN] Lỗi khi eval {pt}: {e}")

    if not all_results:
        return

    print_table(all_results)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n💾 Đã lưu kết quả → '{OUTPUT_CSV}'")

if __name__ == "__main__":
    main()