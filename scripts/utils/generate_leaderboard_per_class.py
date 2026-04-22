import csv
from ultralytics import YOLO
import os

def generate_leaderboard_per_class():
    # Danh sách model và data tương ứng
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

    per_class_results = []
    output_csv = "outputs/results/leaderboard_per_class.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    print(f"🚀 Bắt đầu đánh giá hiệu năng CỰC KỲ CHI TIẾT (5 thông số) của {len(models_to_eval)} models...")

    for item in models_to_eval:
        m_name = item["name"]
        w_path = item["weights"]
        d_path = item["data"]

        if not os.path.exists(w_path) or not os.path.exists(d_path):
            print(f"⚠️ Skip {m_name}: Không tìm thấy weights/config.")
            continue
        
        print(f"\n--- Đang đánh giá: {m_name} ---")
        try:
            model = YOLO(w_path)
            stats = model.val(data=d_path, split='test', workers=0, verbose=False)
            
            # ap_class_index chua danh sach Class ID co trong du lieu test
            ap_class_index = stats.ap_class_index
            
            # stats.box.maps: mang chua mAP50-95 cua moi lop (index theo id)
            # stats.box.ap50: mang chua mAP50 cua moi lop
            # stats.box.p/r: mang precision/recall moi lop
            for i, cls_id in enumerate(ap_class_index):
                cls_name = model.names[cls_id]
                p = float(stats.box.p[i])
                r = float(stats.box.r[i])
                f1 = 2 * (p * r) / (p + r + 1e-16)
                map50 = float(stats.box.ap50[i])
                map50_95 = float(stats.box.maps[cls_id]) # Dung maps vi no map truc tiep theo ID
                
                per_class_results.append({
                    "Class": cls_name,
                    "Model": m_name,
                    "Precision": round(p, 5),
                    "Recall": round(r, 5),
                    "F1-Score": round(f1, 5),
                    "mAP50": round(map50, 5),
                    "mAP50-95": round(map50_95, 5),
                    "Dataset": os.path.basename(d_path)
                })
            
            print(f"✅ Xong {m_name}")
        except Exception as e:
            print(f"❌ Lỗi khi đánh giá {m_name}: {e}")

    # Sắp xếp (Yellow -> Red -> Green -> Off)
    def sort_key(x):
        priority = {"yellow": 0, "red": 1, "green": 2, "off": 3}
        return (priority.get(x["Class"].lower(), 99), -x["mAP50"])

    per_class_results.sort(key=sort_key)

    # Ghi vào file CSV
    if per_class_results:
        keys = ["Class", "Model", "Precision", "Recall", "F1-Score", "mAP50", "mAP50-95", "Dataset"]
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(per_class_results)
        
        print(f"\n✨ Đã lưu Leaderboard toàn diện (5 thông số) vào: {output_csv}")
        
        # In bang tong hop chi tiet
        current_class = ""
        for row in per_class_results:
            if row["Class"] != current_class:
                current_class = row["Class"]
                print(f"\n--- CLASS: {current_class.upper()} ---")
                print(f"{'Model':<25} | {'P':<8} | {'R':<8} | {'F1':<8} | {'m50':<8} | {'m50-95':<8}")
                print("-" * 80)
            print(f"{row['Model']:<25} | {row['Precision']:<8} | {row['Recall']:<8} | {row['F1-Score']:<8} | {row['mAP50']:<8} | {row['mAP50-95']:<8}")
    else:
        print("Không có kết quả nào.")

if __name__ == "__main__":
    generate_leaderboard_per_class()
