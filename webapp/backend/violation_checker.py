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

        self.light_history.append(raw_status)
        # Voting: trạng thái xuất hiện nhiều nhất
        self.current_light_status = max(
            set(self.light_history), key=list(self.light_history).count
        )
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
        is_red = self.current_light_status == "red"
        current_ids = set()

        for box, track_id, cls_id, conf in zip(
            vehicle_boxes, vehicle_track_ids, vehicle_classes, vehicle_confs
        ):
            x1, y1, x2, y2 = map(int, box)
            track_id = int(track_id)
            current_ids.add(track_id)

            # Bottom-center
            bc_x = (x1 + x2) // 2
            bc_y = y2

            inside_roi1 = cv2.pointPolygonTest(
                self.roi1_polygon, (float(bc_x), float(bc_y)), False
            ) >= 0
            inside_roi2 = cv2.pointPolygonTest(
                self.roi2_polygon, (float(bc_x), float(bc_y)), False
            ) >= 0

            if inside_roi1:
                self.vehicles_in_zone1.add(track_id)

            if not inside_roi1 and not inside_roi2:
                self.vehicles_in_zone1.discard(track_id)
                self.zone2_frame_count[track_id] = 0

            if inside_roi2:
                self.zone2_frame_count[track_id] += 1
            else:
                self.zone2_frame_count[track_id] = 0

            consecutive = self.zone2_frame_count[track_id]
            last_vio = self.last_violation_frame.get(track_id, -999999)
            cooldown_ok = (frame_number - last_vio) > self.cooldown_frames

            if (
                is_red
                and inside_roi2
                and track_id in self.vehicles_in_zone1
                and consecutive >= self.frame_threshold
                and cooldown_ok
            ):
                self.violators.add(track_id)
                self.last_violation_frame[track_id] = frame_number
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

    def draw_zones(self, frame: np.ndarray):
        """Vẽ các polygon làn gốc (origin) và làn đích (exit) mờ mờ."""
        # Origin
        _draw_zone_overlay(frame, np.array(self.lane_left, np.int32).reshape((-1, 1, 2)), (255, 100, 100), "Origin:L", 0.2)
        _draw_zone_overlay(frame, np.array(self.lane_mid, np.int32).reshape((-1, 1, 2)), (100, 255, 100), "Origin:S", 0.2)
        _draw_zone_overlay(frame, np.array(self.lane_right, np.int32).reshape((-1, 1, 2)), (100, 100, 255), "Origin:R", 0.2)

        # Exit
        _draw_zone_overlay(frame, np.array(self.exit_left, np.int32).reshape((-1, 1, 2)), (200, 50, 50), "Exit:L", 0.2)
        _draw_zone_overlay(frame, np.array(self.exit_straight, np.int32).reshape((-1, 1, 2)), (50, 200, 50), "Exit:S", 0.2)
        _draw_zone_overlay(frame, np.array(self.exit_right, np.int32).reshape((-1, 1, 2)), (50, 50, 200), "Exit:R", 0.2)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  4. SPEED LIMIT VIOLATION CHECKER
#     (Placeholder — chèn logic detect_speed_limit.py sau)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SpeedLimitChecker:
    """
    Phát hiện xe vượt quá tốc độ cho phép.

    TODO: Tích hợp logic từ detect_speed_limit.py
    - Model detect biển giới hạn tốc độ: detect_speed_sign.pt
    - Kết hợp với tracking để tính tốc độ xe (pixel/frame → km/h)
    - So sánh với giới hạn từ biển báo

    Hiện tại: placeholder, trả về list rỗng.
    """

    def __init__(self):
        self.violated_ids: set = set()
        self.current_speed_limit: int = 0   # km/h, 0 = chưa detect

    def check(
        self,
        speed_sign_results,   # Kết quả từ model detect_speed_sign
        vehicle_boxes,
        vehicle_track_ids,
        vehicle_classes,
        vehicle_confs,
        vehicle_model,
        frame_number: int,
        original_frame: np.ndarray,
    ) -> list[ViolationEvent]:
        """
        TODO: Implement logic phát hiện vượt tốc độ
        Gợi ý:
        1. Detect biển báo tốc độ → lấy speed_limit
        2. Tính tốc độ xe từ trajectory (pixels/frame → estimate km/h)
        3. So sánh: nếu speed > speed_limit → vi phạm
        """
        violations = []
        # ── PLACEHOLDER: Chèn logic tại đây ──
        # ...
        return violations

    def reset(self):
        self.violated_ids.clear()
        self.current_speed_limit = 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  UTILITIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _draw_zone_overlay(frame, polygon, color, label, alpha=0.25):
    """Vẽ vùng polygon bán trong mờ + viền."""
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], color)
    frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=2)
    if label:
        origin = tuple(polygon[0][0])
        cv2.putText(frame, label, origin,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
