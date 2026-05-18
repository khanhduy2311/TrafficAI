import os
from pathlib import Path
from ultralytics import YOLO

model = YOLO("../yolo26n.pt") 

project_dir = "../result_train_fisheye"
run_name = "run_yolo26"
os.makedirs(project_dir, exist_ok=True)

model.train(
    data="/kaggle/input/datasets/nguynthnhthy/data-root/data_fisheye/dataset.yaml",
    epochs=100,
    imgsz=960,             
    batch=16,             
    device=4,
    project=project_dir,
    name=run_name,
    patience=30,
    workers=8,

    mosaic=1.0,  
    mixup=0.0,   
    degrees=10.0,
    scale=0.5,   
    fliplr=0.0,  
)
