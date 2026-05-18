import os
from pathlib import Path
from ultralytics import YOLO


model = YOLO("../rtdetr-l.pt")  

project_dir = "../result_train_fisheye"
run_name = "run_rtdetr"
os.makedirs(project_dir, exist_ok=True)

model.train(
    data="/kaggle/input/datasets/nguynthnhthy/data-root/data_fisheye/dataset.yaml",
    epochs=100,
    imgsz=960,             # Kích thước ảnh đầu vào. Ảnh quá lớn VRAM sẽ rất tốn.
    batch=8,               # Giảm batch size vì ảnh 960 yêu cầu RAM GPU lớn. Bị văng lỗi OOM thì hạ xuống 4
    device=3,
    project=project_dir,
    name=run_name,
    patience=30,
    workers=8,

    # --- Các thông số Augmented Ảnh quan trọng ---
    # Fisheye bản chất có góc méo, việc lạm dụng augmentations lật/xoay có thể phá vỡ đặc trưng gốc. 
    # Nhưng một số Augmentation mặc định của YOLO rất hữu ích:
    mosaic=1.0,  # Ghép 4 ảnh lại với nhau, giúp model học các vật thể bé xíu rất tốt (1.0 = Bật 100%)
    mixup=0.1,   # Trộn mờ 2 ảnh với nhau, tăng cường độ khó (để 0.1 ~ 10%)
    degrees=10.0,# Chọn góc xoay Random (để nhẹ nhàng +-10 độ vì biển báo hiếm khi lộn ngược)
    scale=0.5,   # Zoom-in, Zoom-out vật thể
    fliplr=0.0,  # Fisheye giao thông trên đường, nếu lật ngang có thể dải phân cách sai hướng (tắt: 0.0)
)
