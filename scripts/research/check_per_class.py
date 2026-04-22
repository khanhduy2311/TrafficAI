from ultralytics import YOLO
import os

def check_metrics(name, model_path, data_path):
    print(f"\n{'='*20} Checking {name} {'='*20}")
    if not os.path.exists(model_path):
        print(f"Error: Model path not found: {model_path}")
        return
    model = YOLO(model_path)
    # Run validation and let YOLO print the table
    model.val(data=data_path, split='val', verbose=True, imgsz=960, plots=False, workers=0)

if __name__ == "__main__":
    base_model = r"D:\Detect\final_runs\yolo11n_base_50e_log\weights\best.pt"
    base_data = r"D:\Detect\configs\dataset_fisheye.yaml"
    check_metrics("yolo11n_base", base_model, base_data)
