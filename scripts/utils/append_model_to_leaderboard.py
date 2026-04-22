import csv
from ultralytics import YOLO
import os

def append_to_leaderboard():
    model_name = "yolo11s_base_legacy"
    weights_path = "D:/Detect/final_runs/yolo11s_base_legacy/weights/best.pt"
    data_path = "D:/Detect/configs/dataset_fisheye.yaml"
    
    global_csv = "outputs/results/leaderboard_all.csv"
    per_class_csv = "outputs/results/leaderboard_per_class.csv"

    print(f"🚀 Đang đánh giá riêng model: {model_name}...")
    
    if not os.path.exists(weights_path):
        print(f"❌ Không tìm thấy weights tại {weights_path}")
        return

    model = YOLO(weights_path)
    # Chạy validation trên tập Test
    stats = model.val(data=data_path, split='test', workers=0, verbose=False)

    # 1. Trich xuat Global Metrics
    p_global = stats.results_dict['metrics/precision(B)']
    r_global = stats.results_dict['metrics/recall(B)']
    f1_global = 2 * (p_global * r_global) / (p_global + r_global + 1e-16)
    map50_global = stats.results_dict['metrics/mAP50(B)']
    map50_95_global = stats.results_dict['metrics/mAP50-95(B)']

    global_row = {
        "Model": model_name,
        "Precision": round(float(p_global), 5),
        "Recall": round(float(r_global), 5),
        "F1-Score": round(float(f1_global), 5),
        "mAP50": round(float(map50_global), 5),
        "mAP50-95": round(float(map50_95_global), 5),
        "Dataset": os.path.basename(data_path)
    }

    # 2. Trich xuat Per-Class Metrics
    per_class_rows = []
    ap_class_index = stats.ap_class_index
    for i, cls_id in enumerate(ap_class_index):
        cls_name = model.names[cls_id]
        p = float(stats.box.p[i])
        r = float(stats.box.r[i])
        f1 = 2 * (p * r) / (p + r + 1e-16)
        map50 = float(stats.box.ap50[i])
        map50_95 = float(stats.box.maps[cls_id])
        
        per_class_rows.append({
            "Class": cls_name,
            "Model": model_name,
            "Precision": round(p, 5),
            "Recall": round(r, 5),
            "F1-Score": round(f1, 5),
            "mAP50": round(map50, 5),
            "mAP50-95": round(map50_95, 5),
            "Dataset": os.path.basename(data_path)
        })

    # 3. Ghi vao global leaderboard (Append)
    if os.path.exists(global_csv):
        # Kiem tra xem da co model nay chua de tranh trung lap
        with open(global_csv, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            if model_name in content:
                print(f"⚠️ Model {model_name} da co trong {global_csv}. Dang bo qua...")
            else:
                with open(global_csv, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=global_row.keys())
                    writer.writerow(global_row)
                print(f"✅ Da them {model_name} vao {global_csv}")
    else:
        print(f"❌ Khong tim thay {global_csv}")

    # 4. Ghi vao per-class leaderboard (Append)
    if os.path.exists(per_class_csv):
        with open(per_class_csv, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            if f"{model_name}" in content:
                 print(f"⚠️ Model {model_name} da co trong {per_class_csv}. Dang bo qua...")
            else:
                with open(per_class_csv, 'a', newline='', encoding='utf-8') as f:
                    keys = ["Class", "Model", "Precision", "Recall", "F1-Score", "mAP50", "mAP50-95", "Dataset"]
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writerows(per_class_rows)
                print(f"✅ Da them {len(per_class_rows)} dong vao {per_class_csv}")
    else:
        print(f"❌ Khong tim thay {per_class_csv}")

if __name__ == "__main__":
    append_to_leaderboard()
