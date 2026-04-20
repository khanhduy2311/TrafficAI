import argparse
import cv2
import json
import hashlib
import numpy as np
import math
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────
# 1. ARGUMENT PARSER
# ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Hệ thống phát hiện vi phạm tốc độ (Fisheye Aware)")
    parser.add_argument("--source", required=True, help="Đường dẫn video đầu vào")
    parser.add_argument("--vehicle-weights", default="models/yolo11m.pt", help="Model tracking xe")
    parser.add_argument("--sign-weights", default="models/speed_sign.pt", help="Model nhận diện biển báo")
    parser.add_argument("--output", default="outputs/results/speed_result.mp4", help="Video kết quả")
    parser.add_argument("--evidence-dir", default="outputs/violations", help="Thư mục lưu bằng chứng")
    parser.add_argument("--log-file", default="outputs/violations/speed_log.json", help="File log vi phạm")

    # Thông số Fisheye & Tốc độ
    parser.add_argument("--limit", type=int, default=50, help="Giới hạn tốc độ mặc định (km/h)")
    parser.add_argument("--threshold", type=float, default=1.05, help="Ngưỡng vi phạm (VD: 1.05 = quá 5%)")
    parser.add_argument("--fov", type=float, default=185.0, help="Góc nhìn ngang của Fisheye (độ)")
    parser.add_argument("--real-radius", type=float, default=25.0, help="Bán kính thực tế tại rìa Safe Zone (mét)")
    parser.add_argument("--safe-ratio", type=float, default=0.8, help="Tỉ lệ bán kính Safe Zone (0.0-1.0)")
    
    parser.add_argument("--conf", type=float, default=0.3, help="Nguồng confidence phát hiện xe")
    parser.add_argument("--no-view", action="store_true", help="Không hiển thị cửa sổ live")
    return parser.parse_args()

# ──────────────────────────────────────────────────────────
# 2. CORE LOGIC: FISHEYE & SPEED
# ──────────────────────────────────────────────────────────

class SpeedEngine:
    def __init__(self, w, h, fov_deg, real_radius_m, safe_ratio):
        self.cx, self.cy = w / 2.0, h / 2.0
        self.R_px = min(self.cx, self.cy)
        self.safe_limit_px = self.R_px * safe_ratio
        
        # Equidistant model: r = f * theta
        fov_rad = math.radians(fov_deg) / 2.0
        self.f_px = self.R_px / fov_rad
        self.meters_per_radian = real_radius_m / fov_rad
        
        self.track_history = {} # track_id -> {'pos': deque, 'times': deque, 'speeds': deque}

    def pixel_to_real(self, px, py):
        dx, dy = px - self.cx, py - self.cy
        r_px = math.sqrt(dx**2 + dy**2)
        if r_px < 1e-6: return 0.0, 0.0
        
        theta = r_px / self.f_px
        real_r = self.meters_per_radian * theta
        phi = math.atan2(dy, dx)
        return real_r * math.cos(phi), real_r * math.sin(phi)

    def update_and_get_speed(self, track_id, pt, timestamp):
        # Kiểm tra Safe Zone
        dist_from_center = math.sqrt((pt[0]-self.cx)**2 + (pt[1]-self.cy)**2)
        if dist_from_center > self.safe_limit_px:
            return -1 # Ngoài vùng đo

        if track_id not in self.track_history:
            self.track_history[track_id] = {
                'pos': deque(maxlen=5), 'times': deque(maxlen=5), 'speeds': deque(maxlen=7)
            }
        
        hist = self.track_history[track_id]
        real_pos = self.pixel_to_real(pt[0], pt[1])
        
        speed_kmh = 0
        if len(hist['pos']) > 0:
            dt = timestamp - hist['times'][-1]
            if 0 < dt < 1.0:
                prev_pos = hist['pos'][-1]
                dist = math.sqrt((real_pos[0]-prev_pos[0])**2 + (real_pos[1]-prev_pos[1])**2)
                
                # Teleport check (loại bỏ nhiễu nhảy tâm)
                if dist / dt < 40.0: # < 144km/h
                    current_speed = (dist / dt) * 3.6
                    hist['speeds'].append(current_speed)
                    speed_kmh = np.median(list(hist['speeds']))

        hist['pos'].append(real_pos)
        hist['times'].append(timestamp)
        return int(speed_kmh)

# ──────────────────────────────────────────────────────────
# 3. UTILS
# ──────────────────────────────────────────────────────────

def sha256_of_file(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()

def detect_speed_limit(model, cap, default_limit):
    """Scan 10 frame đầu để tìm biển báo tốc độ"""
    print("[INFO] Đang quét biển báo tốc độ...")
    found_limits = []
    for _ in range(10):
        ret, frame = cap.read()
        if not ret: break
        results = model.predict(frame, verbose=False)[0]
        for box in results.boxes:
            cls_name = model.names[int(box.cls[0])]
            nums = [int(s) for s in cls_name.split('_') if s.isdigit()]
            if nums: found_limits.append(nums[0])
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if found_limits:
        final_limit = max(set(found_limits), key=found_limits.count)
        print(f"[INFO] Phát hiện biển báo: {final_limit} km/h")
        return final_limit
    return default_limit

# ──────────────────────────────────────────────────────────
# 4. MAIN PIPELINE
# ──────────────────────────────────────────────────────────

def main():
    args = parse_args()
    evid_dir = Path(args.evidence_dir)
    evid_dir.mkdir(parents=True, exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Load Models
    vehicle_model = YOLO(args.vehicle_weights)
    sign_model = YOLO(args.sign_weights)
    
    cap = cv2.VideoCapture(args.source)
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    
    # Auto detect speed limit
    current_limit = detect_speed_limit(sign_model, cap, args.limit)
    
    engine = SpeedEngine(w, h, args.fov, args.real_radius, args.safe_ratio)
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    violation_log = []
    cooldown_dict = {} # track_id -> last_violation_time
    
    print(f"[START] Xử lý: {args.source} | Giới hạn: {current_limit} km/h")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        curr_time = frame_count / fps
        display_frame = frame.copy()

        # Vẽ Safe Zone
        cv2.circle(display_frame, (int(w/2), int(h/2)), int(engine.safe_limit_px), (255, 255, 255), 2)
        
        # Tracking xe
        results = vehicle_model.track(frame, persist=True, verbose=False, conf=args.conf)[0]
        
        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            ids = results.boxes.id.int().cpu().numpy()
            clss = results.boxes.cls.int().cpu().numpy()

            for box, tid, cls_id in zip(boxes, ids, clss):
                tid = int(tid)
                x1, y1, x2, y2 = map(int, box)
                anchor = ((x1 + x2) // 2, y2) # Điểm chân xe
                
                speed = engine.update_and_get_speed(tid, anchor, curr_time)
                
                color = (0, 255, 0)
                label = f"ID:{tid} {speed}km/h"

                # Kiểm tra vi phạm
                if speed > current_limit * args.threshold:
                    color = (0, 0, 255)
                    label = f"!! OVER SPEED {speed}km/h !!"
                    
                    # Lưu bằng chứng nếu hết cooldown (5 giây)
                    last_vio = cooldown_dict.get(tid, -99)
                    if curr_time - last_vio > 5.0:
                        cooldown_dict[tid] = curr_time
                        ts = datetime.now().strftime("%H%M%S_%f")
                        
                        # Save images
                        crop = frame[max(0, y1-20):min(h, y2+20), max(0, x1-20):min(w, x2+20)]
                        crop_path = evid_dir / f"speed_id{tid}_{ts}_crop.jpg"
                        full_path = evid_dir / f"speed_id{tid}_{ts}_full.jpg"
                        cv2.imwrite(str(crop_path), crop)
                        
                        # Vẽ minh họa lên ảnh full
                        full_evid = frame.copy()
                        cv2.rectangle(full_evid, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(full_evid, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                        cv2.imwrite(str(full_path), full_evid)
                        
                        # Ghi log
                        entry = {
                            "track_id": tid,
                            "speed": speed,
                            "limit": current_limit,
                            "time_offset": round(curr_time, 2),
                            "crop_hash": sha256_of_file(crop_path),
                            "full_img": str(full_path)
                        }
                        violation_log.append(entry)
                        print(f" [VIOLATION] ID:{tid} chạy {speed} km/h (Limit: {current_limit})")

                if speed != -1:
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # HUD
        cv2.rectangle(display_frame, (10, 10), (280, 80), (0,0,0), -1)
        cv2.putText(display_frame, f"LIMIT: {current_limit} km/h", (20, 45), 0, 0.8, (0, 255, 255), 2)
        cv2.putText(display_frame, f"VIOLATIONS: {len(violation_log)}", (20, 70), 0, 0.6, (0, 0, 255), 2)

        out.write(display_frame)
        if not args.no_view:
            cv2.imshow("Speed Detector", cv2.resize(display_frame, (1280, 720)))
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    out.release()
    
    # Xuất JSON
    with open(args.log_file, "w", encoding="utf-8") as f:
        json.dump({"summary": {"total": len(violation_log)}, "data": violation_log}, f, indent=2)
    
    print(f"Hoàn tất. Kết quả lưu tại {args.output}")

if __name__ == "__main__":
    main()