"""
He thong phat hien nguoi khong doi mu bao hiem
- Su dung YOLO tracking (ByteTrack)
- Model da train tren 2 class: helmet, no_helmet
- Ap dung Algorithm 4: Temporal Confidence Window
- Nguon video: webcam hoac video tu file
- Cach su dung: python detect_no_helmet.py
"""
import cv2
import os
from collections import defaultdict, deque
from ultralytics import YOLO
from datetime import datetime

# ================= CONFIG =================
MODEL_PATH = "runs_yolo11/helmet_violation/weights/best.pt"
VIDEO_PATH = 0  # webcam hoặc video.mp4

CLASS_NAMES = ["helmet", "no_helmet"]  # chỉnh đúng theo model của bạn

SAVE_DIR = "evidence"
os.makedirs(SAVE_DIR, exist_ok=True)

# ================= ALGORITHM 4 PARAMETERS =================
# Temporal Confidence Window (TCW)
WINDOW_SIZE = 10          # N: số frame để tính trung bình (mặc định 10)
CONFIDENCE_THRESHOLD = 0.5  # τ: ngưỡng để xác định vi phạm (mặc định 0.5)
GRACE_PERIOD = 50         # Số frame tránh lưu lại vi phạm của cùng ID

# ================= LOAD MODEL =================
model = YOLO(MODEL_PATH)

# ================= VIDEO =================
cap = cv2.VideoCapture(VIDEO_PATH)

# ================= STATE MANAGEMENT =================
# B_ID: bộ đếm lưu confidence score cho mỗi track_id
# Key: track_id, Value: deque của confidence scores (max len = WINDOW_SIZE)
confidence_buffer = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))

# Lưu frame cuối cùng vi phạm để tránh ghi duplicate
last_violation_frame = {}

# Lưu vị trí bbox cuối cùng để vẽ lên frame
last_bbox = {}

# Frame counter
frame_count = 0

def calculate_violation_score(track_id):
    """
    Tính điểm vi phạm trung bình từ buffer.
    Theo Algorithm 4: S = (1/N) * Σ B_ID[j]
    """
    buffer = confidence_buffer[track_id]
    if len(buffer) == 0:
        return 0.0
    return sum(buffer) / len(buffer)

def is_violation_threshold_met(track_id):
    """
    Kiểm tra điều kiện vi phạm:
    - Độ dài buffer >= WINDOW_SIZE (N khung hình)
    - Điểm vi phạm trung bình > CONFIDENCE_THRESHOLD (τ)
    """
    buffer = confidence_buffer[track_id]
    if len(buffer) < WINDOW_SIZE:
        return False
    
    avg_score = calculate_violation_score(track_id)
    return avg_score > CONFIDENCE_THRESHOLD

def save_evidence(frame, x1, y1, x2, y2, track_id, frame_number, avg_confidence):
    """Lưu ảnh bằng chứng với thông tin chi tiết."""
    h, w = frame.shape[:2]
    pad = 20
    
    # Crop với padding
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(w, x2 + pad)
    cy2 = min(h, y2 + pad)
    crop = frame[cy1:cy2, cx1:cx2]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{SAVE_DIR}/helmet_violation_ID{track_id}_F{frame_number}_S{avg_confidence:.2f}_{timestamp}.jpg"
    
    cv2.imwrite(filename, crop)
    return filename

def cleanup_old_tracks(current_ids, frame_number):
    """Xóa state của track_id không còn xuất hiện (memory management)."""
    all_tracked_ids = set(confidence_buffer.keys()) | set(last_bbox.keys())
    
    for track_id in all_tracked_ids:
        if track_id not in current_ids:
            # Check grace period
            last_frame = last_violation_frame.get(track_id, -999999)
            if frame_number - last_frame > GRACE_PERIOD:
                confidence_buffer.pop(track_id, None)
                last_bbox.pop(track_id, None)
                last_violation_frame.pop(track_id, None)

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    original_frame = frame.copy()
    
    results = model.track(
        frame,
        persist=True,
        conf=0.4,
        iou=0.5
    )

    boxes = results[0].boxes
    current_ids = set()
    violations_detected = []

    # ================= BƯỚC 1: CẬP NHẬT BUFFER (Algorithm 4, dòng 2-9) =================
    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else -1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            current_ids.add(track_id)
            last_bbox[track_id] = (x1, y1, x2, y2)
            
            # Dòng 4: Dự đoán lớp c ∈ {helmet, no_helmet}
            cls_name = CLASS_NAMES[cls_id].lower()
            
            # Dòng 5-9: if c == no_helmet then B_ID.append(P_i) else B_ID.append(0)
            if "no_helmet" in cls_name:
                confidence_buffer[track_id].append(conf)
            else:
                # helmet hoặc class khác → thêm 0
                confidence_buffer[track_id].append(0.0)

    # ================= BƯỚC 2: KIỂM TRA VI PHẠM (Algorithm 4, dòng 10-16) =================
    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else -1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            cls_name = CLASS_NAMES[cls_id].lower()
            
            # Dòng 10: if length(B_ID) >= N then
            if is_violation_threshold_met(track_id):
                avg_confidence = calculate_violation_score(track_id)
                
                # Kiểm tra grace period
                last_frame = last_violation_frame.get(track_id, -999999)
                if frame_count - last_frame > GRACE_PERIOD:
                    # Dòng 11-15: Cảnh báo vi phạm
                    last_violation_frame[track_id] = frame_count
                    
                    evidence_file = save_evidence(
                        original_frame, x1, y1, x2, y2, 
                        track_id, frame_count, avg_confidence
                    )
                    
                    violations_detected.append({
                        'track_id': track_id,
                        'confidence': avg_confidence,
                        'frame': frame_count,
                        'evidence': evidence_file
                    })
                    
                    print(f"[VIOLATION] ID:{track_id} | AvgConf:{avg_confidence:.3f} | "
                          f"Frame:{frame_count} | Saved: {evidence_file}")

    # ================= BƯỚC 3: VẼ BBOX LÊN FRAME =================
    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else -1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            cls_name = CLASS_NAMES[cls_id]
            
            # Chọn màu
            color = (0, 255, 0)  # Xanh lá = helmet
            if "no_helmet" in cls_name.lower():
                color = (0, 0, 255)  # Đỏ = no_helmet
            
            # Hiển thị buffer length và average confidence
            buffer = confidence_buffer[track_id]
            if len(buffer) > 0:
                avg_conf = calculate_violation_score(track_id)
                label = f"{cls_name} ID:{track_id} Conf:{conf:.2f} Avg:{avg_conf:.2f} [{len(buffer)}/{WINDOW_SIZE}]"
            else:
                label = f"{cls_name} ID:{track_id} {conf:.2f}"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # ================= BƯỚC 4: VẼ THÔNG TIN VIOLATION =================
    if violations_detected:
        cv2.putText(frame, 
                    f"[!] VIOLATION DETECTED: {len(violations_detected)} ID(s)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.0, (0, 0, 255), 2)
        
        for i, vio in enumerate(violations_detected):
            text = f"  - ID {vio['track_id']}: {vio['confidence']:.3f}"
            cv2.putText(frame, text, (10, 60 + i*30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
    
    # ================= BƯỚC 5: CLEANUP & DISPLAY =================
    cleanup_old_tracks(current_ids, frame_count)
    
    # Hiển thị thông số
    info_text = f"Frame: {frame_count} | Window: {WINDOW_SIZE} | Threshold: {CONFIDENCE_THRESHOLD}"
    cv2.putText(frame, info_text, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("Helmet Violation Detection (Algorithm 4 - TCW)", frame)

    # ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n[INFO] Xử lý xong {frame_count} frames")
print(f"[INFO] Total violations saved in '{SAVE_DIR}' directory")