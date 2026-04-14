from ultralytics import YOLO

model = YOLO("yolo11s.pt")  # 🔥 đổi sang s (tốt hơn cho data nhỏ)

model.train(
    data="/kaggle/working/data.yaml",
    epochs=100,
    imgsz=960,
    batch=8,

    # augment
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10,
    translate=0.1,
    scale=0.5,
    shear=2.0,
    mosaic=1.0,

    # 🔥 thêm mấy cái này
    patience=20,     # tự dừng nếu không improve
    save_period=1,   # lưu mỗi epoch (rất quan trọng)
    name="train_yolo11"
)