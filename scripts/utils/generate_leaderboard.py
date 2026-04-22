import csv
from ultralytics import YOLO
import os

def generate_leaderboard():
    # Danh sách model và data tương ứng (Đã sửa lỗi đường dẫn và cấu hình cho Windows)
    models_to_eval = [
        {"name": "yolo11n_base", "weights": "final_runs/yolo11n_base_50e_log/weights/best.pt", "data": "D:/Detect/configs/dataset_fisheye.yaml"},
        {"name": "yolo11n_final", "weights": "final_runs/yolo11n_final_50e_balanced2/weights/best.pt", "data": "D:/Detect/configs/dataset_balanced.yaml"},
        {"name": "yolo11n_opt_datanew", "weights": "final_runs/yolo11n_opt_datanew/weights/best.pt", "data": "D:/Detect/Data_new/data.yaml"},
        {"name": "yolo11n_opt_data_new2", "weights": "final_runs/yolo11n_opt_data_new2/weights/best.pt", "data": "D:/Detect/Data_new2/data.yaml"},
        {"name": "yolo11s_base_reproduce", "weights": "final_runs/yolo11s_base_reproduce/weights/best.pt", "data": "D:/Detect/Data_new/data.yaml"},
        {"name": "yolo11s_optimized", "weights": "final_runs/yolo11s_optimized_50e/weights/best.pt", "data": "D:/Detect/configs/dataset_balanced.yaml"},
        {"name": "yolo11s_opt_datanew", "weights": "final_runs/yolo11s_opt_datanew/weights/best.pt", "data": "D:/Detect/Data_new/data.yaml"},
        {"name": "yolo11s_base_legacy", "weights": "final_runs/yolo11s_base_legacy/weights/best.pt", "data": "D:/Detect/configs/dataset_fisheye.yaml"},
    ]

    results_list = []
    output_csv = "outputs/results/leaderboard_all.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    print(f"🚀 Bắt đầu đánh giá {len(models_to_eval)} models trên tập Test tương ứng (workers=0 for stability)...")

    for item in models_to_eval:
        m_name = item["name"]
        w_path = item["weights"]
        d_path = item["data"]

        if not os.path.exists(w_path):
            print(f"⚠️ Skip {m_name}: Không tìm thấy file weights {w_path}")
            continue
        
        if not os.path.exists(d_path):
            print(f"⚠️ Skip {m_name}: Không tìm thấy file config data {d_path}")
            continue

        print(f"\n--- Đang đánh giá: {m_name} ---")
        try:
            model = YOLO(w_path)
            # Chạy Validation trên tập Test 
            # Dùng workers=0 để tránh lỗi Shared Memory trên Windows
            stats = model.val(data=d_path, split='test', workers=0, verbose=False)
            
            # Trích xuất 5 thông số chính
            p = stats.box.mp      # Precision
            r = stats.box.mr      # Recall
            f1 = 2 * (p * r) / (p + r + 1e-16) # F1 Score
            map50 = stats.box.map50 # mAP50
            map50_95 = stats.box.map # mAP50-95
            
            results_list.append({
                "Model": m_name,
                "Dataset": os.path.basename(d_path),
                "Precision": round(float(p), 5),
                "Recall": round(float(r), 5),
                "F1-Score": round(float(f1), 5),
                "mAP50": round(float(map50), 5),
                "mAP50-95": round(float(map50_95), 5)
            })
            print(f"✅ Xong {m_name}: mAP50={map50:.5f}")
        except Exception as e:
            print(f"❌ Lỗi khi đánh giá {m_name}: {e}")

    # Sắp xếp theo mAP50 giảm dần
    results_list.sort(key=lambda x: x["mAP50"], reverse=True)

    # Ghi vào file CSV
    if results_list:
        keys = ["Model", "Dataset", "Precision", "Recall", "F1-Score", "mAP50", "mAP50-95"]
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results_list)
        
        print(f"\n✨ Đã lưu Leaderboard vào: {output_csv}")
        print("\n--- BẢNG TỔNG HỢP KẾT QUẢ ---")
        header = f"{'Model':<25} | {'P':<8} | {'R':<8} | {'F1':<8} | {'mAP50':<8} | {'mAP50-95':<8}"
        print(header)
        print("-" * len(header))
        for row in results_list:
            print(f"{row['Model']:<25} | {row['Precision']:<8} | {row['Recall']:<8} | {row['F1-Score']:<8} | {row['mAP50']:<8} | {row['mAP50-95']:<8}")
    else:
        print("Không có kết quả nào để lưu.")

if __name__ == "__main__":
    generate_leaderboard()
