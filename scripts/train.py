from ultralytics import YOLO
import sys

def train_optimized():
    # --- KIỂM TRA THAM SỐ DÒNG LỆNH ---
    # Nếu chạy 'python scripts/train_optimized.py --resume' -> RESUME_MODE sẽ là True
    RESUME_MODE = "--resume" in sys.argv
    # ----------------------------------

    project_name = "runs_fisheye"
    run_name = "yolo11s_optimized_50e"
    
    if RESUME_MODE:
        # Nếu Resume, trỏ thẳng vào file last.pt trong thư mục weights
        model_path = f"{project_name}/{run_name}/weights/last.pt"
        print(f"🚀 Đang chuẩn bị RESUME từ: {model_path}")
        model = YOLO(model_path)
    else:
        # Nếu Train mới, dùng weights gốc của YOLO11n
        model_path = "yolo11n.pt"
        print(f"🌟 Đang chuẩn bị TRAIN MỚI với: {model_path}")
        model = YOLO(model_path)

    # 2. Cau hinh duong dan
    dataset_path = "D:/Detect/Data_new2/data.yaml"
    project_name = "D:/Detect/final_runs"
    run_name = "yolo11n_opt_data_new2"
    # 3. Bat dau huan luyen voi bo tham so chuyen sau
    results = model.train(
        # Co ban
        data=dataset_path,
        epochs=50,
        batch=8,              # Ha xuong 8 de tranh tran RAM card hinh (OOM) luc Validation
        imgsz=960,
        device=0,
        project=project_name,
        name=run_name,
        resume=RESUME_MODE,  # Kích hoạt chế độ resume của YOLO
        
        # Hyperparameters (Toi uu theo yeu cau cua ban)
        cos_lr=True,
        lr0=0.01,
        lrf=0.1,
        patience=20,
        
        # Loss weights (Giam sat chat che classification)
        box=7.5,
        cls=1.5,
        dfl=1.5,
        
        # Augmentation mạnh (Danh cho du lieu hiem va camera fisheye)
        mosaic=1.0,           # Toi da hoa mosaic
        mixup=0.15,           # Tron anh de tang tinh da dang
        copy_paste=0.3,       # "Dan" den giao thong sang cac bo canh khac
        degrees=5.0,          # Xoay nhe
        perspective=0.0002,   # Bien dang fisheye nhe
        translate=0.1,
        scale=0.3,
        shear=2.0,
        flipud=0.3,           # Lat doc nhe do dac thu fisheye
        fliplr=0.5,           # Lat ngang
        
        # Misc
        save=True,
        save_period=5,
        cache=False,
        workers=1,           # Bat buoc de 0 de chong treo may (deadlock) tren Windows
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        verbose=True,
        plots=True
    )

    print(f"Huan luyen hoan tat! Ket qua luu tai: {project_name}/{run_name}")

if __name__ == "__main__":
    train_optimized()
