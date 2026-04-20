"""
Traffic Speed Violation Detection System
=========================================[Phiên Bản Dứt Điểm - Đã Fix Kích Thước Ảnh & Lấy Mẫu]
"""

import argparse
import csv
import math
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.files import increment_path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_VEHICLE_MODEL = "models/vehicle.pt"
DEFAULT_SIGN_MODEL    = "models/sign.pt"

IMG_SIZE       = 960
CONF_THRESHOLD = 0.25
SIGN_DETECT_CONF = 0.10

SENSOR_FOV_DEG = 185.0
REAL_RADIUS_METERS = 25.0
VALID_RADIUS_RATIO = 0.85

SPEED_MEDIAN_WINDOW       = 10     
SPEED_STABLE_FRAMES       = 5
SPEED_STABLE_TOLERANCE_KMH = 3.0
MAX_PHYSICAL_KMH          = 150.0  
MAX_FRAME_GAP             = 5
UNLOCK_DELTA_KMH          = 8.0
WARMUP_FRAMES             = 6      

DEFAULT_SPEED_LIMIT  = 50
VIOLATION_THRESHOLD  = 1.0
MIN_SPEED_TO_CHECK   = 5
COOLDOWN_FRAMES      = 90

TARGET_CLASSES = {0: "Bus", 1: "Bike", 2: "Car", 3: "Pedestrian", 4: "Truck"}
CLASS_COLORS = {"Bus": (255, 0, 0), "Bike": (0, 255, 255), "Car": (0, 255, 0), "Pedestrian": (0, 0, 255), "Truck": (255, 0, 255)}

EVIDENCE_DIR       = "violations/"
VIOLATION_CSV_PATH = "output/violation_report.csv"
TRACKER_CONFIG     = "bytetrack.yaml"

VALID_SPEED_LIMITS = {20, 30, 40, 50, 60, 70, 80, 90, 100, 120}

# ─────────────────────────────────────────────────────────────────────────────
# CLASSES
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ViolationEvent:
    track_id:    int
    class_name:  str
    speed_kmh:   int
    limit_kmh:   int
    bbox:        tuple
    frame_idx:   int
    timestamp:   float = field(default_factory=time.time)
    image_path:  Optional[str] = None

class FisheyeSpeedEstimator:
    def __init__(self, frame_w: int, frame_h: int, fps: float, K=None, D=None):
        self.fps = fps
        self.cx, self.cy = frame_w / 2.0, frame_h / 2.0
        self.R_px = min(self.cx, self.cy)
        self.valid_R_px = self.R_px * VALID_RADIUS_RATIO

        self.K, self.D = K, D
        self.use_calibration = (K is not None and D is not None)

        fov_half_rad = math.radians(SENSOR_FOV_DEG / 2.0)
        self.f_px = self.R_px / fov_half_rad
        self.valid_theta = fov_half_rad * VALID_RADIUS_RATIO
        
        safe_theta = min(self.valid_theta, math.radians(75.0))
        tan_v = math.tan(safe_theta)
        self.camera_height_m = REAL_RADIUS_METERS / tan_v if tan_v > 1e-6 else 8.0
        self._tracks: dict = {}

    def estimate_speed(self, track_id: int, bbox: tuple, frame_idx: int) -> int:
        x1, y1, x2, y2 = bbox
        x, y = (x1 + x2) / 2.0, float(y2)

        if math.hypot(x - self.cx, y - self.cy) > self.valid_R_px:
            return -1

        if track_id not in self._tracks:
            self._tracks[track_id] = {
                "pos_history": deque(maxlen=10), "speeds": deque(maxlen=SPEED_MEDIAN_WINDOW),
                "stable_count": 0, "locked_speed": None, "zone_frames": 0, "ema_speed": 0.0
            }

        track = self._tracks[track_id]
        track["zone_frames"] += 1
        pos_history = track["pos_history"]

        if pos_history:
            prev_x, prev_y, prev_idx = pos_history[-1]
            frame_gap = frame_idx - prev_idx

            if frame_gap > MAX_FRAME_GAP:
                self._reset_track(track, x, y, frame_idx)
                return 0

            dt = frame_gap / self.fps if self.fps > 0 else 0.0
            if dt > 0:
                dist = self._real_distance(prev_x, prev_y, x, y)
                inst_speed_kmh = (dist / dt) * 3.6

                if inst_speed_kmh > MAX_PHYSICAL_KMH:
                    inst_speed_kmh = track["speeds"][-1] if track["speeds"] else 0.0

                pos_history.append((x, y, frame_idx))
                if track["zone_frames"] <= WARMUP_FRAMES:
                    return 0

                track["speeds"].append(inst_speed_kmh)
                median_speed = float(np.median(track["speeds"]))
                
                if track["ema_speed"] == 0.0:
                    track["ema_speed"] = median_speed
                else:
                    track["ema_speed"] = 0.7 * track["ema_speed"] + 0.3 * median_speed

                smoothed = track["ema_speed"]
                self._update_lock(track, smoothed)

                if track["locked_speed"] is not None: return int(round(track["locked_speed"]))
                return int(round(smoothed))

        pos_history.append((x, y, frame_idx))
        return 0

    def remove_track(self, track_id: int): self._tracks.pop(track_id, None)

    def _reset_track(self, track, x, y, frame_idx):
        track["pos_history"].clear()
        track["speeds"].clear()
        track.update({"stable_count": 0, "locked_speed": None, "ema_speed": 0.0, "zone_frames": 1})
        track["pos_history"].append((x, y, frame_idx))

    def _update_lock(self, track, smoothed: float):
        locked = track["locked_speed"]
        if locked is not None:
            if abs(smoothed - locked) <= UNLOCK_DELTA_KMH:
                track["locked_speed"] = 0.85 * locked + 0.15 * smoothed
            else:
                track["locked_speed"], track["stable_count"] = None, 0
            return
        speeds = list(track["speeds"])
        if len(speeds) < 3: return
        recent = speeds[-min(len(speeds), SPEED_STABLE_FRAMES):]
        if max(recent) - min(recent) <= SPEED_STABLE_TOLERANCE_KMH: track["stable_count"] += 1
        else: track["stable_count"] = 0
        if track["stable_count"] >= SPEED_STABLE_FRAMES - 1: track["locked_speed"] = smoothed

    def _real_distance(self, x1, y1, x2, y2) -> float:
        if self.use_calibration: return self._calibrated_dist(x1, y1, x2, y2)
        gx1, gy1 = self._to_ground(x1, y1)
        gx2, gy2 = self._to_ground(x2, y2)
        return math.hypot(gx2 - gx1, gy2 - gy1)

    def _to_ground(self, px, py):
        dx, dy = px - self.cx, py - self.cy
        r_px = math.hypot(dx, dy)
        if r_px < 1e-6: return 0.0, 0.0
        
        # Tính tỷ lệ khoảng cách từ tâm ra rìa (0.0 đến 1.0)
        ratio = r_px / self.R_px
        
        # Bù trừ độ nén: Càng ra rìa, hệ số nhân càng lớn (hack bù độ méo)
        # Bạn có thể tinh chỉnh số 1.2 này. Nếu rìa vẫn bị giảm tốc độ, tăng lên 1.3 hoặc 1.4
        distortion_correction = 1.0 + (0.25 * (ratio ** 2)) 
        
        theta = min(r_px / self.f_px, math.radians(80.0)) # Nới lỏng giới hạn lên 80 độ
        ground_r = self.camera_height_m * math.tan(theta) * distortion_correction
        
        phi = math.atan2(dy, dx)
        return ground_r * math.cos(phi), ground_r * math.sin(phi)

class ViolationDetector:
    def __init__(self, cooldown_frames: int = COOLDOWN_FRAMES):
        self.cooldown_frames, self._last = cooldown_frames, {}

    def check(self, track_id: int, class_name: str, speed_kmh: int, limit_kmh: int, bbox: tuple, frame_idx: int) -> Optional[ViolationEvent]:
        if speed_kmh < MIN_SPEED_TO_CHECK or speed_kmh <= limit_kmh * VIOLATION_THRESHOLD: return None
        if frame_idx - self._last.get(track_id, -self.cooldown_frames - 1) < self.cooldown_frames: return None
        self._last[track_id] = frame_idx
        return ViolationEvent(track_id, class_name, speed_kmh, limit_kmh, bbox, frame_idx)

class Annotator:
    COLOR_VIOLATION, COLOR_OUTSIDE = (0, 0, 255), (150, 150, 150)
    def __init__(self, frame_w: int, frame_h: int):
        self.cx, self.cy = frame_w // 2, frame_h // 2
        self.r_max = min(self.cx, self.cy)
        self.r_valid = int(self.r_max * VALID_RADIUS_RATIO)

    def draw_zones(self, frame: np.ndarray):
        cv2.circle(frame, (self.cx, self.cy), self.r_max, (255, 255, 255), 2)
        overlay = frame.copy()
        cv2.circle(overlay, (self.cx, self.cy), self.r_valid, (0, 50, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.circle(frame, (self.cx, self.cy), self.r_valid, (0, 255, 0), 2)

    def draw_vehicle(self, frame: np.ndarray, track_id: int, class_name: str, bbox: tuple, speed_kmh: int, is_violation: bool):
        x1, y1, x2, y2 = bbox
        color = self.COLOR_VIOLATION if is_violation else CLASS_COLORS.get(class_name, (255, 255, 255))
        thick = 3 if is_violation else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        cv2.circle(frame, (int((x1 + x2)/2), int(y2)), 5, (0, 255, 255), -1)
        label = f"ID:{track_id} {class_name}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        if speed_kmh == -1: cv2.putText(frame, "-- km/h", (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_OUTSIDE, 2)
        elif speed_kmh > 0:
            txt = f"!!! {speed_kmh} km/h !!!" if is_violation else f"{speed_kmh} km/h"
            cv2.putText(frame, txt, (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)

    def draw_hud(self, frame: np.ndarray, limit_kmh: int):
        h, w = frame.shape[:2]
        x, y = w - 150, 20
        cv2.rectangle(frame, (x - 5, y - 5), (x + 130, y + 55), (0, 0, 0), -1)
        cv2.rectangle(frame, (x - 5, y - 5), (x + 130, y + 55), (255, 255, 255), 2)
        cv2.putText(frame, "SPEED LIMIT", (x, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(frame, f"{limit_kmh} km/h", (x, y + 48), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 220, 255), 3)

    def draw_violation_flash(self, frame: np.ndarray):
        overlay, h, w = frame.copy(), frame.shape[0], frame.shape[1]
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.putText(frame, "VI PHAM TOC DO!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

class EvidenceSaver:
    def __init__(self, evidence_dir: str = EVIDENCE_DIR):
        self.evidence_dir = evidence_dir
        os.makedirs(evidence_dir, exist_ok=True)
    def save(self, frame: np.ndarray, event: ViolationEvent) -> str:
        x1, y1, x2, y2 = event.bbox
        h, w = frame.shape[:2]
        pad = 20
        crop = frame[max(0, y1-pad):min(h-1, y2+pad), max(0, x1-pad):min(w-1, x2+pad)].copy()
        ts = datetime.fromtimestamp(event.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(crop, f"ID:{event.track_id} {event.class_name}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(crop, f"Speed:{event.speed_kmh} Limit:{event.limit_kmh}", (5, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        cv2.putText(crop, ts, (5, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        fpath = os.path.join(self.evidence_dir, f"ID{event.track_id}_{event.speed_kmh}kmh.jpg")
        cv2.imwrite(fpath, crop)
        return fpath

class ReportLogger:
    COLUMNS =["timestamp", "frame_idx", "track_id", "class_name", "speed_kmh", "limit_kmh", "image_path"]
    def __init__(self, csv_path: str = VIOLATION_CSV_PATH):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.COLUMNS).writeheader()
    def log(self, event: ViolationEvent):
        row = {"timestamp": datetime.fromtimestamp(event.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
               "frame_idx": event.frame_idx, "track_id": event.track_id, "class_name": event.class_name,
               "speed_kmh": event.speed_kmh, "limit_kmh": event.limit_kmh, "image_path": event.image_path or ""}
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.COLUMNS).writerow(row)

# ─────────────────────────────────────────────────────────────────────────────
# ĐÃ FIX IMG_SIZE VÀ FRAME QUÉT
# ─────────────────────────────────────────────────────────────────────────────
def _detect_speed_limit(cap: cv2.VideoCapture, sign_model: YOLO) -> int:
    frames_to_check =[0, 5, 10, 15, 20, 25, 30] # Nhảy bước để quét qua cả frame 20 giống code test của bạn
    best_conf = 0.0
    best_limit = None
    
    print(f"\n[SpeedLimit] Đang quét các frame đầu (với imgsz=960) để chốt biển báo...")

    for fno in frames_to_check:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        ret, frame = cap.read()
        if not ret: break

        # FIX CHÍ MẠNG: imgsz=960 (GIỐNG CODE TEST CỦA BẠN ĐỂ NHÌN RÕ BIỂN BÁO)
        results = sign_model.predict(frame, imgsz=960, conf=SIGN_DETECT_CONF, verbose=False)
        if not results or len(results) == 0 or results[0].boxes is None: continue
            
        boxes = results[0].boxes
        for i in range(len(boxes)):
            conf = float(boxes.conf[i])
            cls_id = int(boxes.cls[i])
            class_name = results[0].names.get(cls_id, "")
            
            limit_val = None
            for n in re.findall(r'\d+', class_name):
                if int(n) in VALID_SPEED_LIMITS:
                    limit_val = int(n)
                    break
            if not limit_val and cls_id in VALID_SPEED_LIMITS:
                limit_val = cls_id
                
            if limit_val and conf > best_conf:
                best_conf = conf
                best_limit = limit_val

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if best_limit:
        print(f"[SpeedLimit] => ĐÃ CHỐT: {best_limit} km/h (độ tin cậy tối đa: {best_conf:.3f})\n")
        return best_limit

    print(f"[SpeedLimit] => Không tìm thấy biển báo rõ ràng → Dùng default {DEFAULT_SPEED_LIMIT} km/h\n")
    return DEFAULT_SPEED_LIMIT

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def run(source: str, output_dir: str = "output", vehicle_model: str = DEFAULT_VEHICLE_MODEL, 
        sign_model: str = DEFAULT_SIGN_MODEL, speed_limit: Optional[int] = None, K=None, D=None) -> str:
    
    v_model = YOLO(vehicle_model)
    s_model = YOLO(sign_model)

    cap = cv2.VideoCapture(source)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if speed_limit is not None:
        current_limit = speed_limit
        print(f"[SpeedLimit] User Override: {current_limit} km/h")
    else:
        current_limit = _detect_speed_limit(cap, s_model)

    estimator = FisheyeSpeedEstimator(frame_w, frame_h, fps=fps, K=K, D=D)
    detector  = ViolationDetector()
    annotator = Annotator(frame_w, frame_h)

    run_dir   = increment_path(Path(output_dir) / "run", exist_ok=False)
    os.makedirs(run_dir, exist_ok=True)
    out_video = str(run_dir / "result.mp4")
    out_evid  = str(run_dir / "evidence")
    
    saver  = EvidenceSaver(evidence_dir=out_evid)
    logger = ReportLogger(csv_path=str(run_dir / "violations.csv"))
    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_w, frame_h))

    tracker_cfg = TRACKER_CONFIG if os.path.exists(TRACKER_CONFIG) else "bytetrack.yaml"

    frame_idx, total_viol = 0, 0
    prev_track_ids = set()

    print(f"Bắt đầu Tracking Xe... \nVideo có {total} frames | Output: {out_video}")
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1

            if frame_idx % 50 == 0:
                print(f"  Frame {frame_idx}/{total} ({frame_idx/total*100:.1f}%) | Vi phạm: {total_viol}")

            annotator.draw_zones(frame)
            results = v_model.track(frame, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, tracker=tracker_cfg, persist=True, verbose=False)

            has_viol, cur_track_ids = False, set()

            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
                bboxes    = results[0].boxes.xyxy.cpu().numpy()
                cls_ids   = results[0].boxes.cls.int().cpu().tolist()

                for i, tid in enumerate(track_ids):
                    cid = cls_ids[i]
                    if cid not in TARGET_CLASSES: continue

                    cur_track_ids.add(tid)
                    cls_name = TARGET_CLASSES[cid]
                    bbox     = tuple(map(int, bboxes[i]))

                    speed = estimator.estimate_speed(tid, bbox, frame_idx)

                    is_viol = False
                    if speed > 0:
                        event = detector.check(tid, cls_name, speed, current_limit, bbox, frame_idx)
                        if event:
                            is_viol, has_viol = True, True
                            total_viol += 1
                            event.image_path = saver.save(frame, event)
                            logger.log(event)

                    annotator.draw_vehicle(frame, tid, cls_name, bbox, speed, is_viol)

            for tid in prev_track_ids - cur_track_ids:
                estimator.remove_track(tid)
            prev_track_ids = cur_track_ids

            if has_viol: annotator.draw_violation_flash(frame)
            annotator.draw_hud(frame, current_limit)
            writer.write(frame)

    except KeyboardInterrupt: pass
    finally:
        cap.release()
        writer.release()
    return out_video


# NẾU BẠN CHẠY TRÊN KAGGLE THÌ SỬA ĐƯỜNG DẪN 3 FILE DƯỚI ĐÂY CHO KHỚP LÀ XONG
run(
    source="/kaggle/input/datasets/trngti/full-video/_videos/Town05.mp4", 
    vehicle_model="/kaggle/input/datasets/trngti/main-source/main_system/all_model/yolo11m-big-75-2stg.pt", 
    sign_model="/kaggle/input/datasets/trngti/weigh-balanced/new_weight/best_yolo11m_balance.pt"
)