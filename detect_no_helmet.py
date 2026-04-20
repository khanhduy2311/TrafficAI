"""
He thong phat hien nguoi khong doi mu bao hiem
- Su dung YOLO tracking (ByteTrack)
- Model da train tren 2 class: helmet, no_helmet
- Nguon video: webcam hoac video tu file
- Cach su dung: python detect_no_helmet.py
"""
import cv2
import os
from ultralytics import YOLO
from datetime import datetime

# ================= CONFIG =================
MODEL_PATH = "runs_yolo11/helmet_violation/weights/best.pt"
VIDEO_PATH = 0  # webcam hoặc video.mp4

CLASS_NAMES = ["helmet", "no_helmet"]  # chỉnh đúng theo model của bạn

SAVE_DIR = "evidence"
os.makedirs(SAVE_DIR, exist_ok=True)

# ================= LOAD MODEL =================
model = YOLO(MODEL_PATH)

# ================= VIDEO =================
cap = cv2.VideoCapture(VIDEO_PATH)

# lưu ID đã vi phạm (tránh lưu nhiều lần)
violated_ids = set()

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        conf=0.4,
        iou=0.5
    )

    boxes = results[0].boxes

    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            track_id = int(box.id[0]) if box.id is not None else -1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = f"{CLASS_NAMES[cls_id]} ID:{track_id} {conf:.2f}"

            # ================= DRAW =================
            color = (0, 255, 0)
            if CLASS_NAMES[cls_id] == "no_helmet":
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # ================= VIOLATION =================
            if CLASS_NAMES[cls_id] == "no_helmet":
                if track_id not in violated_ids:
                    violated_ids.add(track_id)

                    # crop ảnh bằng chứng
                    crop = frame[y1:y2, x1:x2]

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{SAVE_DIR}/violation_ID{track_id}_{timestamp}.jpg"

                    cv2.imwrite(filename, crop)

                    print(f"[VIOLATION] Saved: {filename}")

    cv2.imshow("Helmet Violation Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()