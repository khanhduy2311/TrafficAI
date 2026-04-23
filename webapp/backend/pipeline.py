"""
Detection Pipeline — Orchestrate multi-model inference in parallel
Xử lý frame: chạy các model song song → merge → check violations → annotate
"""

import cv2
import numpy as np
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .models_loader import models_manager
from .violation_checker import (
    RedLightChecker,
    NoHelmetChecker,
    WrongLaneChecker,
    SpeedLimitChecker,
    ViolationEvent,
)


# Màu cho từng class xe
VEHICLE_COLORS = {
    "Bus":        (255, 165, 0),    # Cam
    "Car":        (0, 255, 128),    # Xanh lá
    "Bike":       (255, 255, 0),    # Vàng
    "Truck":      (128, 0, 255),    # Tím
    "Pedestrian": (255, 200, 200),  # Hồng nhạt
}

VIOLATION_COLORS = {
    "red_light": (0, 0, 255),
    "no_helmet": (0, 100, 255),
    "wrong_lane": (255, 0, 100),
    "speed_limit": (100, 0, 255),
}

LIGHT_STATUS_DISPLAY = {
    "red":     ("DEN DO",   (0, 0, 255)),
    "yellow":  ("DEN VANG", (0, 165, 255)),
    "green":   ("DEN XANH", (0, 255, 0)),
    "off":     ("DEN TAT",  (150, 150, 150)),
    "unknown": ("---",      (200, 200, 200)),
}


@dataclass
class FrameResult:
    """Kết quả xử lý 1 frame."""
    frame_number: int
    annotated_frame: np.ndarray
    violations: list[ViolationEvent] = field(default_factory=list)
    light_status: str = "unknown"
    vehicle_count: int = 0
    fps: float = 0.0


class DetectionPipeline:
    """
    Pipeline xử lý video:
    1. Đọc frame từ source (video file hoặc webcam)
    2. Chạy multi-model song song (ThreadPoolExecutor)
    3. Check violations
    4. Annotate frame
    5. Trả kết quả
    """

    def __init__(self, max_workers: int = 3):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Video capture
        self.cap: Optional[cv2.VideoCapture] = None
        self.source_type: str = ""       # "webcam" | "video"
        self.video_path: str = ""
        self.session_id: str = ""

        # Frame info
        self.frame_width: int = 0
        self.frame_height: int = 0
        self.source_fps: int = 25
        self.total_frames: int = 0
        self.frame_count: int = 0

        # Violation checkers
        self.red_light_checker = RedLightChecker()
        self.no_helmet_checker = NoHelmetChecker()
        self.wrong_lane_checker = WrongLaneChecker()
        self.speed_limit_checker = SpeedLimitChecker()

        # Running state
        self.is_running: bool = False

    def start(self, source: str = "webcam", video_path: str = ""):
        """Khởi tạo pipeline với nguồn video."""
        self.session_id = str(uuid.uuid4())[:8]
        self.frame_count = 0
        self.source_type = source

        # Reset checkers
        self.red_light_checker.reset()
        self.no_helmet_checker.reset()
        self.wrong_lane_checker.reset()
        self.speed_limit_checker.reset()

        # Load models (nếu chưa load)
        models_manager.load_all()

        # Open video source
        if source == "webcam":
            self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.VideoCapture(video_path)
            self.video_path = video_path

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source} / {video_path}")

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.source_fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 25
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Set ROI cho red_light_checker
        self.red_light_checker.set_roi_from_frame_size(
            self.frame_width, self.frame_height
        )

        self.is_running = True
        print(f"[Pipeline] Started — {self.frame_width}x{self.frame_height} "
              f"@{self.source_fps}fps, session={self.session_id}")

    def stop(self):
        """Dừng pipeline."""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        print("[Pipeline] Stopped.")

    def process_frame(self) -> Optional[FrameResult]:
        """
        Đọc & xử lý 1 frame.
        Trả về FrameResult hoặc None nếu hết video.
        """
        if not self.is_running or self.cap is None:
            return None

        ret, frame = self.cap.read()
        if not ret:
            self.is_running = False
            return None

        self.frame_count += 1
        t_start = time.time()
        original_frame = frame.copy()

        # ═══════════════════════════════════════════════
        # BƯỚC 1: CHẠY CÁC MODEL SONG SONG
        # ═══════════════════════════════════════════════
        futures = {}

        # 1a. Vehicle tracker
        vehicle_model = models_manager.get("vehicle_tracker")
        if vehicle_model:
            futures["vehicle"] = self.executor.submit(
                self._run_vehicle_tracker, vehicle_model, original_frame
            )

        # 1b. Traffic light detector
        light_model = models_manager.get("traffic_light")
        if light_model:
            futures["light"] = self.executor.submit(
                self._run_light_detector, light_model, original_frame
            )

        # 1c. Helmet detector
        helmet_model = models_manager.get("helmet")
        if helmet_model:
            futures["helmet"] = self.executor.submit(
                self._run_helmet_detector, helmet_model, original_frame
            )

        # 1d. Lane sign detector (placeholder)
        lane_model = models_manager.get("lane_sign")
        if lane_model:
            futures["lane"] = self.executor.submit(
                self._run_lane_detector, lane_model, original_frame
            )

        # 1e. Speed sign detector (placeholder)
        speed_model = models_manager.get("speed_sign")
        if speed_model:
            futures["speed"] = self.executor.submit(
                self._run_speed_detector, speed_model, original_frame
            )

        # Chờ tất cả hoàn thành
        results = {}
        for name, future in futures.items():
            try:
                results[name] = future.result(timeout=5)
            except Exception as e:
                print(f"[Pipeline] Error in {name}: {e}")
                results[name] = None

        # ═══════════════════════════════════════════════
        # BƯỚC 2: CHECK VIOLATIONS
        # ═══════════════════════════════════════════════
        all_violations: list[ViolationEvent] = []
        vehicle_count = 0
        light_status = "unknown"

        # --- Traffic light ---
        if results.get("light") is not None and light_model:
            light_status = self.red_light_checker.update_light_status(
                results["light"], light_model, self.frame_height
            )

        # --- Vehicle tracking + Red light check ---
        v_boxes, v_ids, v_classes, v_confs = [], [], [], []
        if results.get("vehicle") is not None:
            vr = results["vehicle"]
            if vr.boxes.id is not None:
                v_boxes = vr.boxes.xyxy.cpu().numpy()
                v_ids = vr.boxes.id.int().cpu().numpy()
                v_classes = vr.boxes.cls.int().cpu().numpy()
                v_confs = vr.boxes.conf.cpu().numpy()
                vehicle_count = len(v_boxes)

                # Red light violations
                red_violations = self.red_light_checker.check_vehicles(
                    v_boxes, v_ids, v_classes, v_confs,
                    vehicle_model, self.frame_count, original_frame
                )
                all_violations.extend(red_violations)

        # --- Helmet check ---
        if results.get("helmet") is not None and helmet_model:
            helmet_violations = self.no_helmet_checker.check(
                results["helmet"], helmet_model,
                self.frame_count, original_frame
            )
            all_violations.extend(helmet_violations)

        # --- Lane check (placeholder) ---
        if results.get("lane") is not None and lane_model and len(v_boxes) > 0:
            lane_violations = self.wrong_lane_checker.check(
                results["lane"], v_boxes, v_ids, v_classes, v_confs,
                vehicle_model, self.frame_count, original_frame
            )
            all_violations.extend(lane_violations)

        # --- Speed check ---
        if results.get("speed") is not None and speed_model and len(v_boxes) > 0:
            speed_violations = self.speed_limit_checker.check(
                results["speed"], v_boxes, v_ids, v_classes, v_confs,
                vehicle_model, self.frame_count, original_frame,
                source_fps=self.source_fps,
                speed_sign_model=speed_model,
            )
            all_violations.extend(speed_violations)

        # ═══════════════════════════════════════════════
        # BƯỚC 3: ANNOTATE FRAME
        # ═══════════════════════════════════════════════
        annotated = frame.copy()
        self._annotate_frame(
            annotated, v_boxes, v_ids, v_classes, v_confs,
            vehicle_model, results, helmet_model, light_model, speed_model,
            light_status, vehicle_count, all_violations
        )

        elapsed = time.time() - t_start
        fps = 1.0 / elapsed if elapsed > 0 else 0

        return FrameResult(
            frame_number=self.frame_count,
            annotated_frame=annotated,
            violations=all_violations,
            light_status=light_status,
            vehicle_count=vehicle_count,
            fps=round(fps, 1),
        )

    # ──────────────────────────────────────────────
    # Model inference functions (run in threads)
    # ──────────────────────────────────────────────

    def _run_vehicle_tracker(self, model, frame):
        return model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            conf=0.4, verbose=False
        )[0]

    def _run_light_detector(self, model, frame):
        return model(frame, verbose=False)[0]

    def _run_helmet_detector(self, model, frame):
        return model.track(
            frame, persist=True, conf=0.4,
            iou=0.5, verbose=False
        )[0]

    def _run_lane_detector(self, model, frame):
        return model(frame, verbose=False)[0]

    def _run_speed_detector(self, model, frame):
        return model(frame, verbose=False)[0]

    # ──────────────────────────────────────────────
    # Annotation
    # ──────────────────────────────────────────────

    def _annotate_frame(
        self, frame, v_boxes, v_ids, v_classes, v_confs,
        vehicle_model, results, helmet_model, light_model, speed_model,
        light_status, vehicle_count, violations
    ):
        """Vẽ bounding boxes, zones, labels, stats lên frame."""

        # 1. Vẽ zones (red light & wrong lane & speed safe-zone)
        self.red_light_checker.draw_zones(frame)
        self.wrong_lane_checker.draw_zones(frame)
        self.speed_limit_checker.draw_safe_zone(frame)

        # 2. Vẽ vehicle boxes
        violated_ids      = self.red_light_checker.violators
        lane_violated_ids = self.wrong_lane_checker.violated_ids
        speed_violated_ids = self.speed_limit_checker.violated_ids
        speed_map          = self.speed_limit_checker.speed_map

        for box, tid, cls_id, conf in zip(v_boxes, v_ids, v_classes, v_confs):
            x1, y1, x2, y2 = map(int, box)
            tid = int(tid)
            cls_name = vehicle_model.names[int(cls_id)]

            if tid in violated_ids or tid in lane_violated_ids or tid in speed_violated_ids:
                color = (0, 0, 255)
                extra = " [VI PHAM]"
            elif tid in self.red_light_checker.vehicles_in_zone1:
                color = (255, 0, 255)
                extra = ""
            else:
                color = VEHICLE_COLORS.get(cls_name, (0, 255, 0))
                extra = ""

            lane_info = ""
            if tid in self.wrong_lane_checker.lane_map:
                lane_info = f" | {self.wrong_lane_checker.lane_map[tid]}"
                if tid in self.wrong_lane_checker.direction_map:
                    lane_info += f" -> {self.wrong_lane_checker.direction_map[tid]}"

            speed_info = ""
            if tid in speed_map:
                speed_info = f" {speed_map[tid]}km/h"

            label = f"{cls_name} ID:{tid}{lane_info}{speed_info}{extra}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)

            # Bottom center dot
            bc_x, bc_y = (x1 + x2) // 2, y2
            cv2.circle(frame, (bc_x, bc_y), 4, (255, 255, 0), -1)

        # 3. Vẽ helmet detection boxes
        if results.get("helmet") is not None and helmet_model:
            for box in results["helmet"].boxes:
                cls_id = int(box.cls[0])
                conf_h = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_name = helmet_model.names[cls_id]

                if "no" in cls_name.lower():
                    color = (0, 0, 255)
                elif "helmet" in cls_name.lower():
                    color = (0, 255, 0)
                else:
                    color = (255, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{cls_name} {conf_h:.2f}"
                cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # 4. Vẽ traffic light boxes
        if results.get("light") is not None and light_model:
            for box in results["light"].boxes:
                cls_id = int(box.cls[0])
                conf_l = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_name = light_model.names[cls_id]

                if "red" in cls_name.lower():
                    color = (0, 0, 255)
                elif "yellow" in cls_name.lower():
                    color = (0, 200, 255)
                elif "green" in cls_name.lower():
                    color = (0, 255, 0)
                else:
                    color = (150, 150, 150)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{cls_name} {conf_l:.2f}"
                cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # 4b. Vẽ biển báo tốc độ
        if results.get("speed") is not None and speed_model:
            SPEED_SIGN_COLOR = (0, 200, 255)   # Cam-cyan
            for box in results["speed"].boxes:
                cls_id = int(box.cls[0])
                conf_s = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_name = speed_model.names[cls_id]  # VD: "50", "60"...

                # Vẽ khung và label
                cv2.rectangle(frame, (x1, y1), (x2, y2), SPEED_SIGN_COLOR, 2)
                sign_label = f"LIMIT {cls_name} km/h  {conf_s:.0%}"
                cv2.putText(frame, sign_label, (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, SPEED_SIGN_COLOR, 2, cv2.LINE_AA)

                # Vẽ badge nền đặc trong góc trên phải biển
                badge_text = f" {cls_name} "
                (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                bx1, by1 = x2 - tw - 4, y1
                bx2, by2 = x2, y1 + th + 6
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), SPEED_SIGN_COLOR, -1)
                cv2.putText(frame, badge_text, (bx1, by2 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

        # 5. HUD overlay
        h, w = frame.shape[:2]
        disp_label, disp_color = LIGHT_STATUS_DISPLAY.get(
            light_status, ("---", (200, 200, 200))
        )
        cv2.putText(frame, f"Light: {disp_label}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, disp_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Vehicles: {vehicle_count}", (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        # Speed limit badge
        limit_val = self.speed_limit_checker.current_speed_limit
        if limit_val > 0:
            cv2.putText(frame, f"Limit: {limit_val} km/h", (15, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1, cv2.LINE_AA)

        total_vio = (
            len(self.red_light_checker.violators)
            + len(self.no_helmet_checker.violated_ids)
            + len(self.wrong_lane_checker.violated_ids)
            + len(self.speed_limit_checker.violated_ids)
        )
        cv2.putText(frame, f"Violations: {total_vio}",
                    (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Frame: {self.frame_count}",
                    (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (150, 150, 150), 1, cv2.LINE_AA)

    def encode_frame_jpeg(self, frame: np.ndarray, quality: int = 70) -> bytes:
        """Encode frame thành JPEG bytes để stream qua WebSocket."""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buffer = cv2.imencode(".jpg", frame, encode_param)
        return buffer.tobytes()
