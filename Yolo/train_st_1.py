from ultralytics import YOLO
if __name__ == "__main__":
    model = YOLO("yolo11m.pt")
    model.train(device=0, data="data.yaml", epochs=150, batch=4, imgsz=960, save_period=5, close_mosaic = 15, cfg="hyps.yaml", optimizer="Adam", workers=40, resume=False, val=True, name="11m_stg_1")