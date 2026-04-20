"""
Violation Checker — Logic phát hiện vi phạm
Tích hợp logic từ detect_traffic_sign.py và detect_no_helmet.py
Chừa placeholder cho lane_driving và speed_limit.
"""

import cv2
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


EVIDENCE_DIR = Path(__file__).resolve().parent.parent / "evidence"


@dataclass
class ViolationEvent:
    """Một sự kiện vi phạm."""
    track_id: int
    vehicle_type: str
    violation_type: str          # "red_light" | "no_helmet" | "wrong_lane" | "speed_limit"
    confidence: float
    frame_number: int
    bbox: list                   # [x1, y1, x2, y2]
    evidence_path: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def to_dict(self):
        return {
            "track_id": self.track_id,
            "vehicle_type": self.vehicle_type,
            "violation_type": self.violation_type,
            "confidence": round(self.confidence, 3),
            "frame_number": self.frame_number,
            "bbox": self.bbox,
            "evidence_path": self.evidence_path,
            "timestamp": self.timestamp,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1. RED LIGHT VIOLATION CHECKER
#     (Ported from detect_traffic_sign.py)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RedLightChecker:
    """
    Phát hiện xe vượt đèn đỏ:
    - Theo dõi xe đi từ Zone 1 → Zone 2 khi đèn đỏ
    - Yêu cầu N frame liên tiếp trong Zone 2 (frame_threshold)
    - Cooldown timer tránh bắt trùng
    - Smoothing trạng thái đèn bằng voting window
    """

    def __init__(
        self,
        frame_threshold: int = 3,
        cooldown_frames: int = 75,      # ~3s ở 25fps
        smooth_window: int = 10,
        light_conf: float = 0.25,
        light_max_y_ratio: float = 0.5,
    ):
        self.frame_threshold = frame_threshold
        self.cooldown_frames = cooldown_frames
        self.light_conf = light_conf
        self.light_max_y_ratio = light_max_y_ratio

        # State
        self.vehicles_in_zone1: set = set()
        self.violators: set = set()
        self.zone2_frame_count: dict = defaultdict(int)
        self.last_violation_frame: dict = {}
        self.light_history: deque = deque(maxlen=smooth_window)
        self.current_light_status: str = "unknown"

        # [Fix #1]
        self.zone1_light_state: dict = {}

        # [Fix #2]
        self.raw_transition_count: dict = defaultdict(int)
        self.prev_raw_status: str = "unknown"
        self.transition_confirm = 2

        # [Fix #3]
        self.last_seen_frame: dict = {}
        self.grace_period_frames = 50  # ~2.0 * fps (assume 25fps)

        # ROI polygons — sẽ được set từ pipeline
        self.roi1_polygon: Optional[np.ndarray] = None
        self.roi2_polygon: Optional[np.ndarray] = None

    def set_roi_from_frame_size(self, width: int, height: int):
        """Tạo ROI mặc định từ kích thước frame (Cấu hình cho xe đi từ dưới lên)."""
        # Zone 1: Vùng chờ ngay trước vạch dừng qua cột đèn đỏ
        self.roi1_polygon = np.array([
            [0,     int(height * 0.20)],
            [width, int(height * 0.20)],
            [width, int(height * 0.45)],
            [0,     int(height * 0.45)],
        ], dtype=np.int32).reshape((-1, 1, 2))

        # Zone 2: Vùng vi phạm, qua vạch (ở ngay trụ đèn đỏ trên cùng)
        self.roi2_polygon = np.array([
            [0,     int(height * 0.05)],
            [width, int(height * 0.05)],
            [width, int(height * 0.20)],
            [0,     int(height * 0.20)],
        ], dtype=np.int32).reshape((-1, 1, 2))

    def _build_light_class_map(self, model) -> dict:
        """Xây dựng mapping class_id → light color."""
        mapping = {"red": [], "green": [], "yellow": [], "off": []}
        for cls_id, cls_name in model.names.items():
            name = str(cls_name).lower()
            for key in mapping:
                if key in name:
                    mapping[key].append(cls_id)
        return mapping

    def update_light_status(self, light_results, light_model, frame_height: int):
        """
        Cập nhật trạng thái đèn từ kết quả detect.
        Trả về trạng thái đèn hiện tại (đã smoothing).
        """
        light_class_map = self._build_light_class_map(light_model)
        light_y_threshold = int(frame_height * self.light_max_y_ratio)
        raw_status = "unknown"

        for box in light_results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            _, y1, _, y2 = map(int, box.xyxy[0])
            center_y = (y1 + y2) // 2

            if center_y > light_y_threshold:
                continue
            if conf < self.light_conf:
                continue

            if cls_id in light_class_map["red"]:
                raw_status = "red"
            elif cls_id in light_class_map["yellow"]:
                if raw_status != "red":
                    raw_status = "yellow"
            elif cls_id in light_class_map["green"]:
                if raw_status not in ("red", "yellow"):
                    raw_status = "green"

        # [Fix #2] Smoothing + Hard transition logic
        if raw_status != "unknown":
            current_smoothed = (
                max(set(self.light_history), key=list(self.light_history).count)
                if self.light_history else "unknown"
            )
            if raw_status != current_smoothed:
                if raw_status == self.prev_raw_status:
                    self.raw_transition_count[raw_status] += 1
                else:
                    self.raw_transition_count.clear()
                    self.raw_transition_count[raw_status] = 1

                if self.raw_transition_count[raw_status] >= self.transition_confirm:
                    self.light_history.clear()
                    self.raw_transition_count.clear()
            else:
                self.raw_transition_count.clear()

        self.prev_raw_status = raw_status
        self.light_history.append(raw_status)
        
        # Voting: trạng thái xuất hiện nhiều nhất
        if self.light_history:
            self.current_light_status = max(set(self.light_history), key=list(self.light_history).count)
        
        return self.current_light_status

    def check_vehicles(
        self,
        vehicle_boxes,
        vehicle_track_ids,
        vehicle_classes,
        vehicle_confs,
        vehicle_model,
        frame_number: int,
        original_frame: np.ndarray,
    ) -> list[ViolationEvent]:
        """
        Kiểm tra xe vượt đèn đỏ:
        - Xe từ Zone1 đi vào Zone2 khi đèn đỏ → vi phạm
        """
        if self.roi1_polygon is None:
            return []

        violations = []
        current_ids = set()

        for box, track_id, cls_id, conf in zip(
            vehicle_boxes, vehicle_track_ids, vehicle_classes, vehicle_confs
        ):
            x1, y1, x2, y2 = map(int, box)
            track_id = int(track_id)
            current_ids.add(track_id)
            
            # [Fix #3] Cập nhật frame cuối cùng nhìn thấy xe
            self.last_seen_frame[track_id] = frame_number

            # Bottom-center
            bc_x = (x1 + x2) // 2
            bc_y = y2

            inside_roi1 = cv2.pointPolygonTest(
                self.roi1_polygon, (float(bc_x), float(bc_y)), False
            ) >= 0
            inside_roi2 = cv2.pointPolygonTest(
                self.roi2_polygon, (float(bc_x), float(bc_y)), False
            ) >= 0

            # [Fix #1] Ghi nhận đèn hiện tại mỗi khi ở zone1
            if inside_roi1:
                self.zone1_light_state[track_id] = self.current_light_status
                self.vehicles_in_zone1.add(track_id)
                self.zone2_frame_count[track_id] = 0

            if not inside_roi1 and not inside_roi2:
                if track_id in self.vehicles_in_zone1:
                    if self.current_light_status in ("green", "yellow", "off") and self.current_light_status != "unknown":
                        self.zone1_light_state[track_id] = self.current_light_status
                else:
                    if self.zone2_frame_count[track_id] == 0 and track_id not in self.violators:
                        self.zone1_light_state.pop(track_id, None)

            if inside_roi2:
                self.zone2_frame_count[track_id] += 1
            elif not inside_roi1:
                self.zone2_frame_count[track_id] = 0

            last_vio = self.last_violation_frame.get(track_id, -999999)
            cooldown_ok = (frame_number - last_vio) > self.cooldown_frames

            # [Fix #4] Quản lý violators
            if track_id in self.violators and cooldown_ok:
                self.violators.discard(track_id)

            consecutive = self.zone2_frame_count[track_id]
            crossed_on_red = self.zone1_light_state.get(track_id) == "red"

            if (
                crossed_on_red
                and inside_roi2
                and track_id in self.vehicles_in_zone1
                and consecutive >= self.frame_threshold
                and cooldown_ok
                and track_id not in self.violators
            ):
                self.violators.add(track_id)
                self.last_violation_frame[track_id] = frame_number
                
                # [Fix #5] Reset frame_count về âm
                self.zone2_frame_count[track_id] = -self.frame_threshold
                
                cls_name = vehicle_model.names[int(cls_id)]

                # Lưu evidence
                evidence_path = self._save_evidence(
                    original_frame, x1, y1, x2, y2, track_id, frame_number
                )

                violations.append(ViolationEvent(
                    track_id=track_id,
                    vehicle_type=cls_name,
                    violation_type="red_light",
                    confidence=float(conf),
                    frame_number=frame_number,
                    bbox=[x1, y1, x2, y2],
                    evidence_path=evidence_path,
                ))

        # [Fix #3] DON DEP STATE VOI GRACE PERIOD
        all_tracked_ids = set(self.vehicles_in_zone1) | set(self.zone1_light_state.keys())
        for tid in list(all_tracked_ids):
            if tid in current_ids:
                continue
                
            frames_absent = frame_number - self.last_seen_frame.get(tid, frame_number)
            if frames_absent > self.grace_period_frames:
                self.vehicles_in_zone1.discard(tid)
                self.zone1_light_state.pop(tid, None)
                self.zone2_frame_count.pop(tid, None)
                self.last_seen_frame.pop(tid, None)

                if frame_number - self.last_violation_frame.get(tid, -999999) > self.cooldown_frames * 2:
                    self.last_violation_frame.pop(tid, None)
                    self.violators.discard(tid)

        return violations

    def _save_evidence(self, frame, x1, y1, x2, y2, track_id, frame_number) -> str:
        """Crop và lưu ảnh bằng chứng."""
        EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        h, w = frame.shape[:2]
        pad = 20
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
        crop = frame[cy1:cy2, cx1:cx2]

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"redlight_id{track_id}_f{frame_number}_{ts}.jpg"
        path = EVIDENCE_DIR / filename
        cv2.imwrite(str(path), crop)
        return filename

    def draw_zones(self, frame: np.ndarray):
        """Vẽ Zone 1, Zone 2 lên frame."""
        if self.roi1_polygon is None:
            return
        is_red = self.current_light_status == "red"
        zone2_color = (0, 0, 255) if is_red else (0, 200, 0)

        _draw_zone_overlay(frame, self.roi1_polygon, (255, 200, 0), "Zone 1")
        _draw_zone_overlay(frame, self.roi2_polygon, zone2_color, "Zone 2")

    def reset(self):
        """Reset toàn bộ state."""
        self.vehicles_in_zone1.clear()
        self.violators.clear()
        self.zone2_frame_count.clear()
        self.last_violation_frame.clear()
        self.light_history.clear()
        self.zone1_light_state.clear()
        self.raw_transition_count.clear()
        self.last_seen_frame.clear()
        self.prev_raw_status = "unknown"
        self.current_light_status = "unknown"
        self.roi1_polygon = None
        self.roi2_polygon = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  2. NO HELMET VIOLATION CHECKER
#     (Ported from detect_no_helmet.py)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class NoHelmetChecker:
    """
    Phát hiện người đi xe máy không đội mũ bảo hiểm.
    Logic: model detect 3 class (Bike, helmet, no_helmet).
    Nếu phát hiện 'no helmet' → vi phạm.
    Dedup bằng track_id.
    """

    def __init__(self):
        self.violated_ids: set = set()

    def check(
        self,
        helmet_results,
        helmet_model,
        frame_number: int,
        original_frame: np.ndarray,
    ) -> list[ViolationEvent]:
        """Kiểm tra kết quả helmet detection."""
        violations = []
        boxes = helmet_results.boxes

        if boxes is None:
            return violations

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else -1
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cls_name = helmet_model.names[cls_id].lower()

            if "no" in cls_name and "helmet" in cls_name:
                if track_id not in self.violated_ids and track_id != -1:
                    self.violated_ids.add(track_id)

                    evidence_path = self._save_evidence(
                        original_frame, x1, y1, x2, y2, track_id, frame_number
                    )

                    violations.append(ViolationEvent(
                        track_id=track_id,
                        vehicle_type="Bike",
                        violation_type="no_helmet",
                        confidence=conf,
                        frame_number=frame_number,
                        bbox=[x1, y1, x2, y2],
                        evidence_path=evidence_path,
                    ))

        return violations

    def _save_evidence(self, frame, x1, y1, x2, y2, track_id, frame_number) -> str:
        EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        h, w = frame.shape[:2]
        pad = 20
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
        crop = frame[cy1:cy2, cx1:cx2]

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"helmet_id{track_id}_f{frame_number}_{ts}.jpg"
        path = EVIDENCE_DIR / filename
        cv2.imwrite(str(path), crop)
        return filename

    def reset(self):
        self.violated_ids.clear()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3. WRONG LANE VIOLATION CHECKER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _in_poly(cx, cy, poly):
    return cv2.pointPolygonTest(np.array(poly, np.int32), (float(cx), float(cy)), False) >= 0

def _box_in_poly_ratio(x1, y1, x2, y2, poly, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(poly, np.int32)], 1)

    box_mask = np.zeros((h, w), dtype=np.uint8)
    # Ràng buộc tọa độ
    y1, y2 = max(0, y1), min(h, y2)
    x1, x2 = max(0, x1), min(w, x2)
    box_mask[y1:y2, x1:x2] = 1

    overlap = np.logical_and(mask, box_mask).sum()
    box_area = (x2 - x1) * (y2 - y1)

    if box_area <= 0:
        return 0
    return overlap / box_area


class WrongLaneChecker:
    """Phát hiện xe chạy sai làn đường dựa trên Polygon định sẵn & Tracking."""

    def __init__(self):
        self.lane_left = [(19, 553), (51, 523), (243, 585), (93, 689), (48, 614)]
        self.lane_mid = [(100, 694), (314, 541), (666, 541), (644, 651), (595, 784), (370, 783), (203, 783)]
        self.lane_right = [(606, 785), (669, 546), (813, 522), (792, 635), (699, 748)]

        self.exit_right = [(647, 383), (759, 509), (841, 491), (843, 451), (737, 386)]
        self.exit_left = [(79, 497), (161, 352), (71, 353), (19, 467), (54, 462)]
        self.exit_straight = [(163, 338), (179, 363), (345, 355), (265, 325), (170, 336)]

        self.lane_map = {}
        self.direction_map = {}
        self.valid_ids = set()
        self.violated_ids = set()
        self.last_positions = {}
        self.sign_detected = False

    def check(
        self,
        lane_results,
        vehicle_boxes,
        vehicle_track_ids,
        vehicle_classes,
        vehicle_confs,
        vehicle_model,
        frame_number: int,
        original_frame: np.ndarray,
    ) -> list[ViolationEvent]:
        violations = []

        if lane_results is not None and len(lane_results.boxes) > 0:
            self.sign_detected = True

        if not self.sign_detected:
            return violations

        h, w = original_frame.shape[:2]
        current_frame_ids = set(int(t) for t in vehicle_track_ids)

        for box, track_id, cls_id, conf in zip(
            vehicle_boxes, vehicle_track_ids, vehicle_classes, vehicle_confs
        ):
            x1, y1, x2, y2 = map(int, box)
            track_id = int(track_id)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cls_name = vehicle_model.names[int(cls_id)].lower()

            if cls_name == "pedestrian":
                continue

            # ===== RECONNECT (fix + cùng loại xe) =====
            if track_id not in self.last_positions:
                for old_id, (old_cx, old_cy, old_frame, old_box, old_cls) in list(self.last_positions.items()):
                    if old_id not in current_frame_ids:
                        import math
                        dist = math.hypot(cx - old_cx, cy - old_cy)
                        frames_passed = frame_number - old_frame
                        dy = cy - old_cy
                        old_x1, old_y1, old_x2, old_y2 = old_box
                        old_w = old_x2 - old_x1
                        new_w = x2 - x1

                        if (
                            dist < 100
                            and frames_passed < 45
                            and dy < 0
                            and abs(old_w - new_w) < 40
                            and old_cls == cls_name
                        ):
                            if old_id in self.valid_ids:
                                self.valid_ids.add(track_id)
                            # Transfer lane_map so violation can be caught if ID changes in intersection
                            if old_id in self.lane_map:
                                self.lane_map[track_id] = self.lane_map[old_id]
                            break

            self.last_positions[track_id] = (cx, cy, frame_number, (x1, y1, x2, y2), cls_name)

            # Kiểm tra xe có nằm trong khu vực hợp lệ (làn đường ban đầu)
            if track_id not in self.valid_ids:
                r_left = _box_in_poly_ratio(x1, y1, x2, y2, self.lane_left, h, w)
                r_mid = _box_in_poly_ratio(x1, y1, x2, y2, self.lane_mid, h, w)
                r_right = _box_in_poly_ratio(x1, y1, x2, y2, self.lane_right, h, w)

                if max(r_left, r_mid, r_right) > 0.5:
                    self.valid_ids.add(track_id)

            if track_id not in self.valid_ids:
                continue

            # Lưu lại origin lane
            if track_id not in self.lane_map:
                ratios = {
                    "left": _box_in_poly_ratio(x1, y1, x2, y2, self.lane_left, h, w),
                    "straight": _box_in_poly_ratio(x1, y1, x2, y2, self.lane_mid, h, w),
                    "right": _box_in_poly_ratio(x1, y1, x2, y2, self.lane_right, h, w)
                }
                best_lane = max(ratios, key=ratios.get)
                if ratios[best_lane] > 0.5:
                    self.lane_map[track_id] = best_lane

            # Xét hướng đi vào exit rules
            if _in_poly(cx, cy, self.exit_straight):
                self.direction_map[track_id] = "straight"
            elif _in_poly(cx, cy, self.exit_left):
                self.direction_map[track_id] = "left"
            elif _in_poly(cx, cy, self.exit_right):
                self.direction_map[track_id] = "right"

            # Check vi phạm
            if track_id in self.lane_map and track_id in self.direction_map:
                if self.lane_map[track_id] != self.direction_map[track_id]:
                    if track_id not in self.violated_ids:
                        self.violated_ids.add(track_id)
                        
                        # Lưu ảnh bằng chứng
                        evidence_path = self._save_evidence(
                            original_frame, x1, y1, x2, y2, track_id, frame_number
                        )
                        
                        violations.append(ViolationEvent(
                            track_id=track_id,
                            vehicle_type=vehicle_model.names[int(cls_id)],
                            violation_type="wrong_lane",
                            confidence=float(conf),
                            frame_number=frame_number,
                            bbox=[x1, y1, x2, y2],
                            evidence_path=evidence_path,
                        ))

        return violations

    def _save_evidence(self, frame, x1, y1, x2, y2, track_id, frame_number) -> str:
        EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        h, w = frame.shape[:2]
        pad = 20
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
        crop = frame[cy1:cy2, cx1:cx2]

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"wronglane_id{track_id}_f{frame_number}_{ts}.jpg"
        path = EVIDENCE_DIR / filename
        cv2.imwrite(str(path), crop)
        return filename

    def reset(self):
        self.lane_map.clear()
        self.direction_map.clear()
        self.valid_ids.clear()
        self.violated_ids.clear()
        self.last_positions.clear()
        self.sign_detected = False

    def draw_zones(self, frame: np.ndarray):
        """Vẽ origin lanes + exit zones đẹp hơn với tâm label và hướng rõ ràng."""
        if not self.sign_detected:
            return

        # ---- Origin zones (nơi xe xuất phát) ----
        _draw_lane_zone(frame, self.lane_left,    (255, 180,  60), "<- LEFT",     alpha=0.18)
        _draw_lane_zone(frame, self.lane_mid,     ( 60, 220,  60), "^ STRAIGHT",  alpha=0.18)
        _draw_lane_zone(frame, self.lane_right,   ( 60, 160, 255), "-> RIGHT",    alpha=0.18)

        # ---- Exit zones (hướng đầu xe đi ra) ----
        _draw_lane_zone(frame, self.exit_left,     (200, 120,  20), "EXIT <-",  alpha=0.15, border_color=(255,200,100))
        _draw_lane_zone(frame, self.exit_straight, ( 20, 160,  20), "EXIT ^",   alpha=0.15, border_color=(120,255,120))
        _draw_lane_zone(frame, self.exit_right,    ( 20, 100, 200), "EXIT ->",  alpha=0.15, border_color=(120,180,255))



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  4. FISHEYE SPEED ESTIMATOR
#     Ported from detect_speed_limit.py (phiên bản mới nhất)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import math as _math
import re as _re

# ── Hằng số tốc độ (sync với detect_speed_limit.py) ──
_SPEED_MEDIAN_WINDOW        = 10
_SPEED_STABLE_FRAMES        = 5
_SPEED_STABLE_TOLERANCE_KMH = 3.0
_MAX_PHYSICAL_KMH           = 150.0
_MAX_FRAME_GAP              = 5
_UNLOCK_DELTA_KMH           = 8.0
_WARMUP_FRAMES              = 6
_VALID_SPEED_LIMITS         = {20, 30, 40, 50, 60, 70, 80, 90, 100, 120}
_VALID_RADIUS_RATIO         = 1.0
_SENSOR_FOV_DEG             = 185.0
_REAL_RADIUS_METERS         = 25.0
_MIN_SPEED_TO_CHECK         = 5
_VIOLATION_THRESHOLD        = 1.0   # >= limit*1.0 → vi phạm (không có biên dung sai)
_COOLDOWN_FRAMES            = 90


class FisheyeSpeedEstimator:
    """
    Tốc độ kế fisheye-aware — dùng mô hình camera height + distortion correction.
    Tích hợp EMA smoothing + lock speed để ổn định kết quả đầu ra.
    """

    def __init__(self, frame_w: int, frame_h: int, fps: float):
        self.fps = fps
        self.cx, self.cy = frame_w / 2.0, frame_h / 2.0
        self.R_px = min(self.cx, self.cy)
        self.valid_R_px = self.R_px * _VALID_RADIUS_RATIO

        fov_half_rad = _math.radians(_SENSOR_FOV_DEG / 2.0)
        self.f_px = self.R_px / fov_half_rad
        valid_theta = fov_half_rad * _VALID_RADIUS_RATIO
        safe_theta = min(valid_theta, _math.radians(75.0))
        tan_v = _math.tan(safe_theta)
        self.camera_height_m = _REAL_RADIUS_METERS / tan_v if tan_v > 1e-6 else 8.0

        self._tracks: dict = {}

    # ── Public API ──────────────────────────────────────────
    def estimate_speed(self, track_id: int, bbox: tuple, frame_idx: int) -> int:
        """
        Trả về tốc độ (km/h) >= 0, hoặc -1 nếu xe ngoài valid zone.
        Trả 0 trong warmup phase.
        """
        x1, y1, x2, y2 = bbox
        x, y = (x1 + x2) / 2.0, float(y2)

        if _math.hypot(x - self.cx, y - self.cy) > self.valid_R_px:
            return -1

        if track_id not in self._tracks:
            self._tracks[track_id] = {
                "pos_history": deque(maxlen=10),
                "speeds":      deque(maxlen=_SPEED_MEDIAN_WINDOW),
                "stable_count": 0,
                "locked_speed": None,
                "zone_frames":  0,
                "ema_speed":    0.0,
            }

        track = self._tracks[track_id]
        track["zone_frames"] += 1
        pos_history = track["pos_history"]

        if pos_history:
            prev_x, prev_y, prev_idx = pos_history[-1]
            frame_gap = frame_idx - prev_idx

            if frame_gap > _MAX_FRAME_GAP:
                self._reset_track(track, x, y, frame_idx)
                return 0

            dt = frame_gap / self.fps if self.fps > 0 else 0.0
            if dt > 0:
                dist = self._real_distance(prev_x, prev_y, x, y)
                inst_kmh = (dist / dt) * 3.6

                if inst_kmh > _MAX_PHYSICAL_KMH:
                    inst_kmh = track["speeds"][-1] if track["speeds"] else 0.0

                pos_history.append((x, y, frame_idx))

                if track["zone_frames"] <= _WARMUP_FRAMES:
                    return 0

                track["speeds"].append(inst_kmh)
                median_speed = float(np.median(track["speeds"]))

                if track["ema_speed"] == 0.0:
                    track["ema_speed"] = median_speed
                else:
                    track["ema_speed"] = 0.7 * track["ema_speed"] + 0.3 * median_speed

                smoothed = track["ema_speed"]
                self._update_lock(track, smoothed)

                if track["locked_speed"] is not None:
                    return int(round(track["locked_speed"]))
                return int(round(smoothed))

        pos_history.append((x, y, frame_idx))
        return 0

    def remove_track(self, track_id: int):
        self._tracks.pop(track_id, None)

    def reset(self):
        self._tracks.clear()

    # ── Private helpers ──────────────────────────────────────
    def _reset_track(self, track, x, y, frame_idx):
        track["pos_history"].clear()
        track["speeds"].clear()
        track.update({"stable_count": 0, "locked_speed": None,
                      "ema_speed": 0.0, "zone_frames": 1})
        track["pos_history"].append((x, y, frame_idx))

    def _update_lock(self, track, smoothed: float):
        locked = track["locked_speed"]
        if locked is not None:
            if abs(smoothed - locked) <= _UNLOCK_DELTA_KMH:
                track["locked_speed"] = 0.85 * locked + 0.15 * smoothed
            else:
                track["locked_speed"], track["stable_count"] = None, 0
            return
        speeds = list(track["speeds"])
        if len(speeds) < 3:
            return
        recent = speeds[-min(len(speeds), _SPEED_STABLE_FRAMES):]
        if max(recent) - min(recent) <= _SPEED_STABLE_TOLERANCE_KMH:
            track["stable_count"] += 1
        else:
            track["stable_count"] = 0
        if track["stable_count"] >= _SPEED_STABLE_FRAMES - 1:
            track["locked_speed"] = smoothed

    def _real_distance(self, x1, y1, x2, y2) -> float:
        gx1, gy1 = self._to_ground(x1, y1)
        gx2, gy2 = self._to_ground(x2, y2)
        return _math.hypot(gx2 - gx1, gy2 - gy1)

    def _to_ground(self, px, py):
        dx, dy = px - self.cx, py - self.cy
        r_px = _math.hypot(dx, dy)
        if r_px < 1e-6:
            return 0.0, 0.0
        ratio = r_px / self.R_px
        distortion_correction = 1.0 + (0.25 * (ratio ** 2))
        theta = min(r_px / self.f_px, _math.radians(80.0))
        ground_r = self.camera_height_m * _math.tan(theta) * distortion_correction
        phi = _math.atan2(dy, dx)
        return ground_r * _math.cos(phi), ground_r * _math.sin(phi)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  5. SPEED LIMIT VIOLATION CHECKER
#     Dùng FisheyeSpeedEstimator + model detect biển báo tốc độ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SpeedLimitChecker:
    """
    Phát hiện xe vượt quá tốc độ cho phép.

    - Chỉ bắt vi phạm sau khi model detect được biển báo tốc độ (sign_detected).
    - Tính tốc độ bằng FisheyeSpeedEstimator (EMA + lock speed + distortion).
    - Cooldown 90 frame / xe để tránh lưu bằng chứng trùng lặp.
    """

    def __init__(self):
        self.current_speed_limit: int = 0
        self.sign_detected: bool = False
        self.violated_ids: set = set()
        self.cooldown_dict: dict = {}       # track_id → last_violation_frame
        self.speed_map: dict = {}           # track_id → speed (km/h) để annotate
        self._estimator: Optional[FisheyeSpeedEstimator] = None
        self._prev_ids: set = set()

    # ── Sign parsing ─────────────────────────────────────────
    def _parse_speed_limit(self, sign_results, sign_model) -> Optional[int]:
        """
        Lấy giới hạn tốc độ cao nhất confidence từ kết quả model biển báo.
        Class name dạng "50", "speed_60", "limit_30" → lấy số hợp lệ.
        """
        if sign_results is None or sign_model is None:
            return None
        best_conf, best_limit = 0.0, None
        for box in sign_results.boxes:
            conf = float(box.conf[0])
            cls_name = sign_model.names[int(box.cls[0])]
            limit_val = None
            for n in _re.findall(r'\d+', cls_name):
                if int(n) in _VALID_SPEED_LIMITS:
                    limit_val = int(n)
                    break
            # Thử cls_id trực tiếp nếu class name không chứa số hợp lệ
            if limit_val is None:
                cls_id_val = int(box.cls[0])
                if cls_id_val in _VALID_SPEED_LIMITS:
                    limit_val = cls_id_val
            if limit_val and conf > best_conf:
                best_conf, best_limit = conf, limit_val
        return best_limit

    # ── Main check ───────────────────────────────────────────
    def check(
        self,
        speed_sign_results,
        vehicle_boxes,
        vehicle_track_ids,
        vehicle_classes,
        vehicle_confs,
        vehicle_model,
        frame_number: int,
        original_frame: np.ndarray,
        source_fps: int = 25,
        speed_sign_model=None,
    ) -> list[ViolationEvent]:
        violations = []

        # Khởi tạo estimator khi biết kích thước frame
        if self._estimator is None:
            h, w = original_frame.shape[:2]
            self._estimator = FisheyeSpeedEstimator(w, h, fps=float(source_fps))

        # Cập nhật giới hạn tốc độ từ biển báo
        new_limit = self._parse_speed_limit(speed_sign_results, speed_sign_model)
        if new_limit:
            self.current_speed_limit = new_limit
            self.sign_detected = True

        # Chưa thấy biển nào → không xử lý
        if not self.sign_detected or self.current_speed_limit <= 0:
            return violations

        limit = self.current_speed_limit
        cur_ids: set = set()

        for box, track_id, cls_id, conf in zip(
            vehicle_boxes, vehicle_track_ids, vehicle_classes, vehicle_confs
        ):
            x1, y1, x2, y2 = map(int, box)
            track_id = int(track_id)
            cls_name = vehicle_model.names[int(cls_id)]

            if cls_name.lower() == "pedestrian":
                continue

            cur_ids.add(track_id)
            bbox = (x1, y1, x2, y2)
            speed = self._estimator.estimate_speed(track_id, bbox, frame_number)

            # Lưu để annotate (kể cả speed == 0)
            if speed >= 0:
                self.speed_map[track_id] = speed

            # Kiểm tra vi phạm
            if speed >= _MIN_SPEED_TO_CHECK and speed > limit * _VIOLATION_THRESHOLD:
                last_vio_frame = self.cooldown_dict.get(track_id, -_COOLDOWN_FRAMES - 1)
                if frame_number - last_vio_frame >= _COOLDOWN_FRAMES:
                    self.cooldown_dict[track_id] = frame_number
                    self.violated_ids.add(track_id)

                    evidence_path = self._save_evidence(
                        original_frame, x1, y1, x2, y2, track_id, frame_number, speed
                    )
                    violations.append(ViolationEvent(
                        track_id=track_id,
                        vehicle_type=cls_name,
                        violation_type="speed_limit",
                        confidence=float(conf),
                        frame_number=frame_number,
                        bbox=[x1, y1, x2, y2],
                        evidence_path=evidence_path,
                    ))

        # Dọn track đã biến mất
        for gone in self._prev_ids - cur_ids:
            self._estimator.remove_track(gone)
        self._prev_ids = cur_ids

        return violations

    def _save_evidence(self, frame, x1, y1, x2, y2, track_id, frame_number, speed) -> str:
        EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        h, w = frame.shape[:2]
        pad = 20
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
        crop = frame[cy1:cy2, cx1:cx2].copy()

        # Ghi thông tin lên ảnh bằng chứng
        cv2.putText(crop, f"ID:{track_id} {speed}km/h",
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(crop, f"Limit:{self.current_speed_limit}km/h",
                    (5, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"speed_id{track_id}_f{frame_number}_{speed}kmh_{ts}.jpg"
        cv2.imwrite(str(EVIDENCE_DIR / filename), crop)
        return filename

    def draw_safe_zone(self, frame: np.ndarray):
        """Vẽ vòng tròn valid-zone khít rình lens fisheye."""
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        r = int(min(cx, cy) * _VALID_RADIUS_RATIO)
        cv2.circle(frame, (cx, cy), r, (180, 0, 255), 2, cv2.LINE_AA)

    def reset(self):
        self.violated_ids.clear()
        self.cooldown_dict.clear()
        self.speed_map.clear()
        self.current_speed_limit = 0
        self.sign_detected = False
        self._prev_ids.clear()
        if self._estimator:
            self._estimator.reset()



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  UTILITIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _poly_centroid(pts):
    """Tính tâm (centroid) của polygon."""
    arr = np.array(pts, dtype=np.float32)
    return int(arr[:, 0].mean()), int(arr[:, 1].mean())


def _draw_lane_zone(
    frame,
    pts,
    fill_color,
    label: str,
    arrow: str = None,
    alpha: float = 0.18,
    border_color=None,
):
    """
    Vẽ một polygon làn đường đẹp:
      - fill mờ trong suốt
      - viền dày rõ nét (LINE_AA)
      - label + mũi tên Unicode ở tâm polygon, có nền đen shadow
    """
    poly = np.array(pts, np.int32).reshape((-1, 1, 2))
    bcolor = border_color if border_color else fill_color

    # Fill mờ
    overlay = frame.copy()
    cv2.fillPoly(overlay, [poly], fill_color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Viền rõ nét
    cv2.polylines(frame, [poly], isClosed=True, color=bcolor, thickness=2, lineType=cv2.LINE_AA)

    # Label ở tâm
    cx, cy = _poly_centroid(pts)
    text = f"{arrow} {label}" if arrow else label
    font = cv2.FONT_HERSHEY_DUPLEX
    scale, thick = 0.52, 1

    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    # Nền đen phía sau chữ
    cv2.rectangle(
        frame,
        (cx - tw // 2 - 3, cy - th - 3),
        (cx + tw // 2 + 3, cy + 3),
        (0, 0, 0), -1,
    )
    cv2.putText(frame, text, (cx - tw // 2, cy),
                font, scale, bcolor, thick, cv2.LINE_AA)


def _draw_zone_overlay(frame, polygon, color, label, alpha=0.25):
    """Legacy helper — dùng bởi RedLightChecker."""
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], color)
    frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=2)
    if label:
        origin = tuple(polygon[0][0])
        cv2.putText(frame, label, origin,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
