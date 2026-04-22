from ultralytics import YOLO
import os

def check_model_per_class(name, model_path, data_path, split='test'):
    print(f"\n{'='*40}")
    print(f"=== ĐÁNH GIÁ MÔ HÌNH: {name} ===")
    print(f"{'='*40}")
    if not os.path.exists(model_path):
        print(f"Error: Model path not found: {model_path}")
        return
    model = YOLO(model_path)
    # Run validation
    results = model.val(data=data_path, split=split, verbose=True, imgsz=960, plots=False, workers=0)
    
    # In ra Metrics tong quan va tung class
    print(f"\n[Kết quả Tổng quan - Mọi classes]")
    p = results.box.mp
    r = results.box.mr
    map50 = results.box.map50
    map95 = results.box.map
    print(f"Precision: {p:.5f} | Recall: {r:.5f} | mAP@50: {map50:.5f} | mAP@50-95: {map95:.5f}")
    
    print(f"\n[Kết quả Từng Class (Precision / Recall / mAP50)]")
    class_names = results.names
    for idx in range(len(class_names)):
        cls_name = class_names[idx]
        cls_p = results.box.p[idx]
        cls_r = results.box.r[idx]
        cls_map = results.box.ap50[idx]
        print(f" - {cls_name.ljust(8)}: P = {cls_p:.5f} | R = {cls_r:.5f} | mAP50 = {cls_map:.5f}")

if __name__ == "__main__":
    # Su dung cung tap test chung de so sanh cong bang nhat
    data_config = r"D:\Detect\configs\dataset_balanced.yaml"
    
    models_to_check = {
        "11n_base": r"D:\Detect\final_runs\yolo11n_base_50e_log\weights\best.pt",
        "11n_final": r"D:\Detect\final_runs\yolo11n_final_50e_balanced2\weights\best.pt",
        "11s_opt": r"D:\Detect\final_runs\yolo11s_opt_datanew\weights\best.pt",
        "11s_optimized": r"D:\Detect\final_runs\yolo11s_optimized_50e\weights\best.pt"
    }
    
    for name, path in models_to_check.items():
        check_model_per_class(name, path, data_config, split='test')
