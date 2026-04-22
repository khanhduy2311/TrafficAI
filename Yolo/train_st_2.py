from ultralytics import YOLO
if __name__ == "__main__":
    model = YOLO("Yolo/runs/detect/11m_stg_1/weights/best.pt")
    model.train(device=0, data="data_stage_2.yaml", epochs=10, batch=4, imgsz=960, save_period=5, close_mosaic = 15, cfg="hyps.yaml", optimizer="Adam", workers=40, resume=False, val=False, name="11m_stg_2")