import os
from pathlib import Path
from ultralytics import YOLO

# 1. Tải mô hình biến thể YOLO26n (Theo file setup trong workspace)
model = YOLO("../yolo26n.pt") 

# 2. Định nghĩa thư mục lưu
project_dir = "../result_train_fisheye"
run_name = "run_yolo26"
os.makedirs(project_dir, exist_ok=True)

# 3. Tiến hành Training
model.train(
    data="/kaggle/input/datasets/nguynthnhthy/data-root/data_fisheye/dataset.yaml",
    epochs=100,
    imgsz=960,             
    batch=16,              # Bị OOM (Out of Memory) thì hãy hạ xuống 8
    device=4,
    project=project_dir,
    name=run_name,
    patience=30,
    workers=8,

    # --- Các thông số Augmented Ảnh quan trọng ---
    mosaic=1.0,  
    mixup=0.0,   
    degrees=10.0,
    scale=0.5,   
    fliplr=0.0,  
)
