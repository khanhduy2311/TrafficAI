"""
Traffic Speed Violation 
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

#CONFIG
DEFAULT_VEHICLE_MODEL = "models/vehicle.pt"
DEFAULT_SIGN_MODEL    = "models/sign.pt"

IMG_SIZE         = 960
CONF_THRESHOLD   = 0.25
SIGN_DETECT_CONF = 0.10

# Perspective speed estimator
CAMERA_HEIGHT_M      = 9.0
FOCAL_LENGTH_PX      = 500.0
GLOBAL_SPEED_SCALE   = 1.4

SPEED_MEDIAN_WINDOW        = 15
SPEED_STABLE_FRAMES        = 3
SPEED_STABLE_TOLERANCE_KMH = 8.0
MAX_PHYSICAL_KMH           = 120.0
MAX_FRAME_GAP              = 4
UNLOCK_DELTA_KMH           = 8.0
WARMUP_FRAMES              = 10

DEFAULT_SPEED_LIMIT  = 50
VIOLATION_THRESHOLD  = 1.0
MIN_SPEED_TO_CHECK   = 5
COOLDOWN_FRAMES      = 90

TARGET_CLASSES = {0: "Bus", 1: "Bike", 2: "Car", 3: "Pedestrian", 4: "Truck"}
CLASS_COLORS   = {
    "Bus": (255, 0, 0), "Bike": (0, 255, 255),
    "Car": (0, 255, 0), "Pedestrian": (0, 0, 255), "Truck": (255, 0, 255),
}

EVIDENCE_DIR       = "violations/"
VIOLATION_CSV_PATH = "output/violation_report.csv"
TRACKER_CONFIG     = "bytetrack.yaml"

#Classes for speed sign
VALID_SPEED_LIMITS = {20, 30, 40, 50, 60, 70, 80, 90, 100, 120}

#Set up 
@dataclass
class ViolationEvent:
    track_id:   int
    class_name: str
    speed_kmh:  int
    limit_kmh:  int
    bbox:       tuple
    frame_idx:  int
    timestamp:  float = field(default_factory=time.time)
    image_path: Optional[str] = None

#Perspective Projection (Pinhole Camera Model) & Violation Detection with Cooldown
class PerspectiveSpeedEstimator:

    def __init__(self, frame_w: int, frame_h: int, fps: float, video_name: str = "unknown"):
        self.fps     = fps
        self.frame_w = frame_w
        self.frame_h = frame_h
        self._tracks: dict = {}
        
        # Cấu hình vùng tham chiếu ảo
        self.y_start = int(frame_h * 0.35)
        self.y_end   = int(frame_h * 0.65)
        self.s_zone  = 25.0
        self.video_name = video_name
        self.eval_csv = "output/evaluation_log.csv"
        
        os.makedirs(os.path.dirname(self.eval_csv) or ".", exist_ok=True)
        if not os.path.exists(self.eval_csv):
            with open(self.eval_csv, "w", encoding="utf-8") as f:
                f.write("video_name,track_id,v_raw,v_ema,v_locked,v_zone,entry_frame,exit_frame\n")

    def estimate_speed(self, track_id: int, bbox: tuple, frame_idx: int) -> int:
        """Trả về tốc độ km/h >= 0. Trả -1 nếu bbox quá nhỏ (không tin cậy)."""
        x1, y1, x2, y2 = bbox
        cx_px = (x1 + x2) / 2.0
        cy_px = (y1 + y2) / 2.0
        bh_px = float(y2 - y1)

        if bh_px < 8:
            return -1

        if track_id not in self._tracks:
            self._tracks[track_id] = {
                "history":      deque(maxlen=12),   # (cx, cy, bh, frame_idx)
                "speeds":       deque(maxlen=SPEED_MEDIAN_WINDOW),
                "stable_count": 0,
                "locked_speed": None,
                "ema_speed":    0.0,
                "zone_frames":  0,
                "entry_frame":  None,
                "exit_frame":   None,
                "v_zone_logged": False
            }

        track = self._tracks[track_id]
        track["zone_frames"] += 1
        history = track["history"]

        if history:
            prev_cx, prev_cy, prev_bh, prev_idx = history[-1]
            frame_gap = frame_idx - prev_idx

            if frame_gap > MAX_FRAME_GAP:
                self._reset_track(track, cx_px, cy_px, bh_px, frame_idx)
                return 0

            dt = frame_gap / self.fps if self.fps > 0 else 0.0
            if dt > 0:
                avg_bh = (bh_px + prev_bh) / 2.0

                # Depth từ similar triangles: camera_h * focal / bbox_h
                depth_m = (CAMERA_HEIGHT_M * FOCAL_LENGTH_PX) / max(avg_bh, 1.0)

                # Chuyển di chuyển pixel → mét thực
                dpx    = math.hypot(cx_px - prev_cx, cy_px - prev_cy)
                dist_m = dpx * depth_m / FOCAL_LENGTH_PX * GLOBAL_SPEED_SCALE

                inst_kmh = (dist_m / dt) * 3.6
                if inst_kmh > MAX_PHYSICAL_KMH:
                    inst_kmh = track["speeds"][-1] if track["speeds"] else 0.0

                history.append((cx_px, cy_px, bh_px, frame_idx))

                if track["zone_frames"] <= WARMUP_FRAMES:
                    return 0

                track["speeds"].append(inst_kmh)
                median_speed = float(np.median(track["speeds"]))

                # EMA smoothing
                if track["ema_speed"] == 0.0:
                    track["ema_speed"] = median_speed
                else:
                    track["ema_speed"] = 0.75 * track["ema_speed"] + 0.25 * median_speed

                smoothed = track["ema_speed"]
                self._update_lock(track, smoothed)

                # --- Vùng tham chiếu ảo logic ---
                if not track["v_zone_logged"]:
                    if track["entry_frame"] is None and cy_px >= self.y_start and cy_px < self.y_end:
                        track["entry_frame"] = frame_idx
                    
                    if track["entry_frame"] is not None and track["exit_frame"] is None and cy_px >= self.y_end:
                        track["exit_frame"] = frame_idx
                        frames_passed = track["exit_frame"] - track["entry_frame"]
                        
                        # Chỉ tính V_zone nếu xe tốn ít nhất 5 frames để đi qua (chống chia cho số quá nhỏ gây nhiễu)
                        if frames_passed >= 5:
                            time_passed = frames_passed / self.fps
                            v_zone = (self.s_zone / time_passed) * 3.6
                            v_raw = inst_kmh
                            v_ema = smoothed
                            v_locked = track["locked_speed"] if track["locked_speed"] is not None else 0.0
                            
                            with open(self.eval_csv, "a", encoding="utf-8") as f:
                                f.write(f"{self.video_name},{track_id},{v_raw:.2f},{v_ema:.2f},{v_locked:.2f},{v_zone:.2f},{track['entry_frame']},{track['exit_frame']}\n")
                        track["v_zone_logged"] = True
                # --------------------------------

                if track["locked_speed"] is not None:
                    return int(round(track["locked_speed"]))
                return int(round(smoothed))

        history.append((cx_px, cy_px, bh_px, frame_idx))
        return 0

    def remove_track(self, track_id: int):
        self._tracks.pop(track_id, None)

    def _reset_track(self, track, cx, cy, bh, frame_idx):
        track["history"].clear()
        track["speeds"].clear()
        track.update({"stable_count": 0, "locked_speed": None, "ema_speed": 0.0, "zone_frames": 1, "entry_frame": None, "exit_frame": None, "v_zone_logged": False})
        track["history"].append((cx, cy, bh, frame_idx))

    def _update_lock(self, track, smoothed: float):
        """Khóa tốc độ khi đã ổn định đủ frame liên tiếp."""
        locked = track["locked_speed"]
        if locked is not None:
            if abs(smoothed - locked) <= UNLOCK_DELTA_KMH:
                track["locked_speed"] = 0.85 * locked + 0.15 * smoothed
            else:
                track["locked_speed"], track["stable_count"] = None, 0
            return
        speeds = list(track["speeds"])
        if len(speeds) < 3:
            return
        recent = speeds[-min(len(speeds), SPEED_STABLE_FRAMES):]
        if max(recent) - min(recent) <= SPEED_STABLE_TOLERANCE_KMH:
            track["stable_count"] += 1
        else:
            track["stable_count"] = 0
        if track["stable_count"] >= SPEED_STABLE_FRAMES - 1:
            track["locked_speed"] = smoothed


class ViolationDetector:
    """
    Phát hiện vi phạm tốc độ với cooldown per-track.
    speed_map dùng để annotate tốc độ trên frame.
    """

    def __init__(self, cooldown_frames: int = COOLDOWN_FRAMES):
        self.cooldown_frames = cooldown_frames
        self._last: dict = {}       # track_id → frame_idx lần vi phạm cuối
        self.speed_map: dict = {}   # track_id → speed_kmh để annotate

    def check(
        self,
        track_id: int,
        class_name: str,
        speed_kmh: int,
        limit_kmh: int,
        bbox: tuple,
        frame_idx: int,
    ) -> Optional[ViolationEvent]:
        #Update speed map
        if speed_kmh >= 0:
            self.speed_map[track_id] = speed_kmh

        if speed_kmh < MIN_SPEED_TO_CHECK or speed_kmh <= limit_kmh * VIOLATION_THRESHOLD:
            return None
        if frame_idx - self._last.get(track_id, -self.cooldown_frames - 1) < self.cooldown_frames:
            return None

        self._last[track_id] = frame_idx
        return ViolationEvent(track_id, class_name, speed_kmh, limit_kmh, bbox, frame_idx)


class Annotator:
    COLOR_VIOLATION = (0, 0, 255)

    def __init__(self, frame_w: int, frame_h: int):
        # Không cần valid radius nữa vì dùng perspective estimator
        pass

    def draw_zones(self, frame: np.ndarray):
        pass

    def draw_vehicle(
        self,
        frame: np.ndarray,
        track_id: int,
        class_name: str,
        bbox: tuple,
        speed_kmh: int,
        is_violation: bool,
    ):
        x1, y1, x2, y2 = bbox
        color = self.COLOR_VIOLATION if is_violation else CLASS_COLORS.get(class_name, (255, 255, 255))
        thick = 3 if is_violation else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        cv2.circle(frame, (int((x1 + x2) / 2), int(y2)), 5, (0, 255, 255), -1)

        label = f"ID:{track_id} {class_name}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        if speed_kmh > 0:
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
        overlay = frame.copy()
        h, w = frame.shape[:2]
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
        pad  = 20
        crop = frame[
            max(0, y1 - pad):min(h - 1, y2 + pad),
            max(0, x1 - pad):min(w - 1, x2 + pad),
        ].copy()
        ts = datetime.fromtimestamp(event.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(crop, f"ID:{event.track_id} {event.class_name}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(crop, f"Speed:{event.speed_kmh} Limit:{event.limit_kmh}", (5, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        cv2.putText(crop, ts, (5, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        fpath = os.path.join(self.evidence_dir, f"ID{event.track_id}_{event.speed_kmh}kmh.jpg")
        cv2.imwrite(fpath, crop)
        return fpath


class ReportLogger:
    COLUMNS = ["timestamp", "frame_idx", "track_id", "class_name", "speed_kmh", "limit_kmh", "image_path"]

    def __init__(self, csv_path: str = VIOLATION_CSV_PATH):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.COLUMNS).writeheader()

    def log(self, event: ViolationEvent):
        row = {
            "timestamp":  datetime.fromtimestamp(event.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
            "frame_idx":  event.frame_idx,
            "track_id":   event.track_id,
            "class_name": event.class_name,
            "speed_kmh":  event.speed_kmh,
            "limit_kmh":  event.limit_kmh,
            "image_path": event.image_path or "",
        }
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.COLUMNS).writerow(row)


#DETECT SPEED LIMIT FROM VIDEO
def _detect_speed_limit(cap: cv2.VideoCapture, sign_model: YOLO) -> int:
    """Quét các frame đầu để chốt giới hạn tốc độ từ biển báo."""
    frames_to_check = [0, 5, 10, 15, 20, 25, 30]
    best_conf, best_limit = 0.0, None

    print(f"\n[SpeedLimit] Đang quét các frame đầu để chốt biển báo...")

    for fno in frames_to_check:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        ret, frame = cap.read()
        if not ret:
            break

        results = sign_model.predict(frame, imgsz=960, conf=SIGN_DETECT_CONF, verbose=False)
        if not results or results[0].boxes is None:
            continue

        boxes = results[0].boxes
        for i in range(len(boxes)):
            conf     = float(boxes.conf[i])
            cls_id   = int(boxes.cls[i])
            cls_name = results[0].names.get(cls_id, "")

            # Parse số hợp lệ từ class name, fallback sang cls_id
            limit_val = None
            for n in re.findall(r'\d+', cls_name):
                if int(n) in VALID_SPEED_LIMITS:
                    limit_val = int(n)
                    break
            if limit_val is None and cls_id in VALID_SPEED_LIMITS:
                limit_val = cls_id

            if limit_val and conf > best_conf:
                best_conf  = conf
                best_limit = limit_val

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if best_limit:
        print(f"[SpeedLimit] => Chốt: {best_limit} km/h (conf: {best_conf:.3f})\n")
        return best_limit

    print(f"[SpeedLimit] => Không tìm thấy biển → dùng default {DEFAULT_SPEED_LIMIT} km/h\n")
    return DEFAULT_SPEED_LIMIT


#MAIN
def process_video(
    source: str,
    v_model: YOLO,
    s_model: YOLO,
    output_dir: str = "output",
    speed_limit: Optional[int] = None,
) -> str:
    cap     = cv2.VideoCapture(source)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if speed_limit is not None:
        current_limit = speed_limit
        print(f"[SpeedLimit] User override: {current_limit} km/h")
    else:
        current_limit = _detect_speed_limit(cap, s_model)

    video_name = os.path.basename(source)
    estimator = PerspectiveSpeedEstimator(frame_w, frame_h, fps=fps, video_name=video_name)
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

    print(f"Bắt đầu tracking... {total} frames | Output: {out_video}")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if frame_idx % 50 == 0:
                print(f"  Frame {frame_idx}/{total} ({frame_idx / total * 100:.1f}%) | Vi phạm: {total_viol}")

            annotator.draw_zones(frame)
            results = v_model.track(
                frame, imgsz=IMG_SIZE, conf=CONF_THRESHOLD,
                tracker=tracker_cfg, persist=True, verbose=False,
            )

            has_viol, cur_track_ids = False, set()

            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
                bboxes    = results[0].boxes.xyxy.cpu().numpy()
                cls_ids   = results[0].boxes.cls.int().cpu().tolist()

                for i, tid in enumerate(track_ids):
                    cid = cls_ids[i]
                    if cid not in TARGET_CLASSES:
                        continue

                    cls_name = TARGET_CLASSES[cid]
                    if cls_name == "Pedestrian":
                        continue

                    cur_track_ids.add(tid)
                    bbox  = tuple(map(int, bboxes[i]))
                    speed = estimator.estimate_speed(tid, bbox, frame_idx)

                    is_viol = False
                    if speed > 0:
                        event = detector.check(tid, cls_name, speed, current_limit, bbox, frame_idx)
                        if event:
                            is_viol, has_viol = True, True
                            total_viol       += 1
                            event.image_path  = saver.save(frame, event)
                            logger.log(event)

                    annotator.draw_vehicle(frame, tid, cls_name, bbox, speed, is_viol)

            for tid in prev_track_ids - cur_track_ids:
                estimator.remove_track(tid)
            prev_track_ids = cur_track_ids

            if has_viol:
                annotator.draw_violation_flash(frame)
            annotator.draw_hud(frame, current_limit)
            writer.write(frame)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        writer.release()

    return out_video


def run(
    source_dir: str,
    output_dir: str = "output",
    vehicle_model: str = DEFAULT_VEHICLE_MODEL,
    sign_model: str = DEFAULT_SIGN_MODEL,
    speed_limit: Optional[int] = None,
):
    v_model = YOLO(vehicle_model)
    s_model = YOLO(sign_model)
    
    if os.path.isdir(source_dir):
        videos = [os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.endswith(('.mp4', '.avi'))]
    else:
        videos = [source_dir]
        
    # Xóa log cũ trước khi chạy batch mới để tránh dính dữ liệu cũ
    eval_csv = "output/evaluation_log.csv"
    if os.path.exists(eval_csv):
        os.remove(eval_csv)
        
    print(f"[*] Found {len(videos)} videos to process in {source_dir}")
    for vid in videos:
        print(f"\n{'='*50}\n[*] Processing {vid}\n{'='*50}")
        process_video(vid, v_model, s_model, output_dir, speed_limit)

if __name__ == "__main__":
    run(
        source_dir="data/test_videos",
        vehicle_model="model/tracking.pt",
        sign_model="model/detect_speed_sign.pt",
    )