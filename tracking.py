"""
Code tracking phuong tien giao thong tren duong
- Su dung YOLO tracking (ByteTrack)
- Model da train tren 5 class: Bus, Bike, Car, Pedestrian, Truck
- Nguon video: webcam hoac video tu file
- Cach su dung: python tracking.py  
"""
import cv2
from ultralytics import YOLO

# ================= CONFIG =================
MODEL_PATH = "best.pt"   # model bạn đã train
VIDEO_PATH = 0           # 0 = webcam | hoặc "video.mp4"

CLASS_NAMES = ["Bus", "Bike", "Car", "Pedestrian", "Truck"]

# ================= LOAD MODEL =================
model = YOLO(MODEL_PATH)

# ================= VIDEO =================
cap = cv2.VideoCapture(VIDEO_PATH)

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO tracking (ByteTrack mặc định)
    results = model.track(
        frame,
        persist=True,   # giữ ID giữa các frame
        conf=0.3,
        iou=0.5
    )

    # lấy kết quả
    boxes = results[0].boxes

    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            track_id = int(box.id[0]) if box.id is not None else -1
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = f"{CLASS_NAMES[cls_id]} ID:{track_id} {conf:.2f}"

            # vẽ bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Traffic Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()