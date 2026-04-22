"""
Violation Checker — Logic phát hiện vi phạm
Tích hợp logic từ detect_traffic_sign.py và detect_no_helmet.py
Bao gồm WrongWayChecker (xe đi ngược chiều) tối ưu cho camera fisheye từ trên cao.
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
    violation_type: str          # "red_light" | "no_helmet" | "wrong_lane" | "wrong_way" | "speed_limit"
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

        # Chỉ vẽ zones khi đã detect được đèn giao thông
        self.traffic_light_ever_detected: bool = False

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

        # Đánh dấu đã detect được đèn ít nhất 1 lần
        if raw_status != "unknown":
            self.traffic_light_ever_detected = True

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
        """Vẽ Zone 1, Zone 2 lên frame — chỉ khi đã detect được đèn giao thông."""
        if self.roi1_polygon is None:
            return
        # Chỉ hiển thị zone khi đèn giao thông đã được phát hiện
        if not self.traffic_light_ever_detected:
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
        self.traffic_light_ever_detected = False
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
    y1, y2 = max(0, y1), min(h, y2)
    x1, x2 = max(0, x1), min(w, x2)
    box_mask[y1:y2, x1:x2] = 1

    overlap = np.logical_and(mask, box_mask).sum()
    box_area = (x2 - x1) * (y2 - y1)

    if box_area <= 0:
        return 0
    return overlap / box_area


# ──────────────────────────────────────────────────────────────
#  SceneFlowEstimator (dùng cho WrongWayChecker)
#  Tính hướng chính của luồng xe để làm reference
# ──────────────────────────────────────────────────────────────
class _SceneFlowEstimator:
    """
    Ước tính luồng di chuyển tổng thể của toàn bộ xe trong scene.
    Camera fisheye từ trên cao → xe di chuyển theo 2D vector (dx, dy).
    Scene flow = vector trung bình của các xe "đúng chiều" để làm tham chiếu.
    """

    def __init__(self, window: int = 30):
        # Lưu list vector (vx, vy) trung bình mỗi frame
        self._vx_buf: deque = deque(maxlen=window)
        self._vy_buf: deque = deque(maxlen=window)

    def update(self, vx_list: list, vy_list: list):
        """Nhận danh sách velocity từng xe trong frame, lấy median."""
        if len(vx_list) >= 2:
            self._vx_buf.append(float(np.median(vx_list)))
            self._vy_buf.append(float(np.median(vy_list)))

    def get_flow_vector(self) -> Optional[tuple]:
        """Trả về (vx, vy) scene flow hoặc None nếu chưa đủ data."""
        if len(self._vx_buf) < 5:
            return None
        return float(np.mean(self._vx_buf)), float(np.mean(self._vy_buf))

    def cosine_similarity(self, vx: float, vy: float) -> Optional[float]:
        """
        Tính cosine similarity giữa vector xe và scene flow.
        = 1.0  → cùng hướng hoàn toàn
        = 0.0  → vuông góc
        = -1.0 → ngược chiều hoàn toàn
        """
        flow = self.get_flow_vector()
        if flow is None:
            return None
        fx, fy = flow
        mag_v    = (vx**2 + vy**2) ** 0.5
        mag_flow = (fx**2 + fy**2) ** 0.5
        if mag_v < 1e-6 or mag_flow < 1e-6:
            return None
        return (vx * fx + vy * fy) / (mag_v * mag_flow)

    def reset(self):
        self._vx_buf.clear()
        self._vy_buf.clear()


# ──────────────────────────────────────────────────────────────
#  TrackMotion — lưu lịch sử vị trí & tính vector vận tốc 2D
# ──────────────────────────────────────────────────────────────
class _TrackMotion:
    """
    Lưu lịch sử tâm bbox theo frame.
    Tính vector vận tốc 2D (vx, vy) trung bình trong cửa sổ gần nhất.
    Dùng cho camera top-down / fisheye: cả dx và dy đều có ý nghĩa.
    """

    def __init__(self, history_len: int = 20):
        self._history: deque = deque(maxlen=history_len)  # (frame_idx, cx, cy)

    def update(self, frame_idx: int, cx: float, cy: float):
        self._history.append((frame_idx, cx, cy))

    def get_velocity(self, min_frames: int = 4) -> Optional[tuple]:
        """
        Trả về (vx, vy) pixel/frame trung bình.
        None nếu chưa đủ frame hoặc không di chuyển.
        """
        if len(self._history) < min_frames:
            return None
        pts = list(self._history)
        vx_list, vy_list = [], []
        for i in range(1, len(pts)):
            df = pts[i][0] - pts[i-1][0]
            if df <= 0:
                continue
            vx_list.append((pts[i][1] - pts[i-1][1]) / df)
            vy_list.append((pts[i][2] - pts[i-1][2]) / df)
        if not vx_list:
            return None
        return float(np.mean(vx_list)), float(np.mean(vy_list))

    def get_displacement(self) -> float:
        """Khoảng cách pixel từ điểm đầu đến điểm cuối."""
        if len(self._history) < 2:
            return 0.0
        import math
        p0, p1 = self._history[0], self._history[-1]
        return math.hypot(p1[1] - p0[1], p1[2] - p0[2])

    def speed_norm(self) -> float:
        """Tốc độ chuẩn hóa (pixel/frame)."""
        vel = self.get_velocity()
        if vel is None:
            return 0.0
        return (vel[0]**2 + vel[1]**2) ** 0.5

    def reset(self):
        self._history.clear()


# ──────────────────────────────────────────────────────────────
#  WrongWayChecker — phát hiện xe đi ngược chiều
#  Tối ưu cho camera fisheye từ trên cao (bird's-eye / top-down)
# ──────────────────────────────────────────────────────────────
class WrongWayChecker:
    """
    Phát hiện xe đi ngược chiều dựa trên HƯỚNG DI CHUYỂN 2D (vx, vy).

    Nguyên lý (top-down fisheye camera):
    ─────────────────────────────────────────────────────────────
    1. Mỗi xe có một track lịch sử tâm bbox → tính velocity vector (vx, vy).
    2. SceneFlowEstimator tính vector trung bình của TẤT CẢ xe đang chạy
       → đây là hướng "đúng chiều" tham chiếu.
    3. Cosine similarity giữa vector xe và scene flow:
         cos > threshold  → đúng chiều
         cos < -threshold → NGƯỢC CHIỀU → vi phạm
    4. Confirmation: phải có N frame liên tiếp cos âm mới bắt vi phạm
       (chống false positive khi xe rẽ / dừng / đổi hướng tạm thời).
    5. ROI: chỉ kiểm tra xe nằm trong vùng đường 1 chiều đã định nghĩa.

    Ưu điểm so với thuật toán cũ (chỉ dùng dy):
    ─────────────────────────────────────────────────────────────
    - Dùng vector 2D → hoạt động đúng với camera trên cao
    - Scene flow adaptive → tự học hướng đúng từ traffic thực tế
    - Không bị ảnh hưởng bởi góc nghiêng camera
    - Confirmation frames → ít false positive
    - Hỗ trợ nhiều vùng đường (multi-ROI) với từng hướng riêng
    """

    # Cấu hình mặc định
    DEFAULT_CFG = {
        # Ngưỡng cosine similarity: cos < -threshold → ngược chiều
        # Range: 0.0 (vuông góc) → 1.0 (đối diện hoàn toàn)
        # Khuyến nghị: 0.3 cho đường thẳng, 0.5 cho ngã tư
        'cosine_wrong_way_thresh': 0.30,

        # Số frame liên tiếp phải cos âm mới confirm vi phạm
        'confirm_frames': 5,

        # Giảm counter mỗi frame không vi phạm (hysteresis)
        'confirm_decay': 1,

        # Số frame tối thiểu để tính velocity
        'min_frames_to_judge': 5,

        # Tốc độ pixel/frame tối thiểu để xét (lọc xe đứng yên)
        'min_speed_px_per_frame': 1.5,

        # Khoảng cách dịch chuyển tối thiểu (pixel) trong toàn bộ history
        'min_displacement_px': 12,

        # Scene flow window (số frame tích lũy)
        'scene_flow_window': 30,

        # Tỉ lệ diện tích bbox tối đa so với frame (lọc bbox quá to)
        'max_bbox_area_pct': 0.60,

        # Cooldown frames giữa 2 lần bắt cùng 1 xe
        'cooldown_frames': 90,

        # Override per class (giảm ngưỡng cho Bike/Pedestrian)
        'class_overrides': {
            'Bike': {
                'min_speed_px_per_frame': 1.0,
                'min_displacement_px': 8,
                'confirm_frames': 4,
                'min_frames_to_judge': 4,
            },
            'Pedestrian': {
                'min_speed_px_per_frame': 0.8,
                'min_displacement_px': 6,
                'confirm_frames': 4,
                'min_frames_to_judge': 4,
                'cosine_wrong_way_thresh': 0.25,
            },
        },

        # Merge Bike ↔ Pedestrian track (giảm ID switch)
        'merge_bike_pedestrian': True,

        # Hiển thị vector debug lên frame
        'show_velocity_arrow': True,
    }

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = {**self.DEFAULT_CFG, **(cfg or {})}

        # Scene flow (adaptive hướng đúng chiều)
        self._scene_flow = _SceneFlowEstimator(
            window=self.cfg['scene_flow_window']
        )

        # Per-track state
        self._tracks: dict[int, dict] = {}
        # track_id → {
        #   'motion': _TrackMotion,
        #   'class_name': str,
        #   'consec_wrong': int,
        #   'confirmed': bool,
        #   'viol_frame': int | None,
        #   'last_cos': float | None,
        # }

        # Confirmed violators
        self.violated_ids: set = set()
        self._cooldown: dict = {}       # track_id → last_viol_frame

        # ROI zones: list of polygon pts (pixel)
        # Nếu rỗng → kiểm tra toàn frame
        self._roi_zones: list = []      # list of np.ndarray polygon

        # Mapping class alias (Bike ↔ Pedestrian merge)
        self._id_alias: dict = {}       # new_id → canonical_id

    # ── Public API ──────────────────────────────────────────

    def set_roi_zones(self, zones: list):
        """
        Định nghĩa các vùng đường cần kiểm tra ngược chiều.
        zones: list of polygon [(x,y), ...] pixel tuyệt đối.
        """
        self._roi_zones = [np.array(z, np.int32) for z in zones]

    def set_roi_from_frame_size(self, width: int, height: int):
        """
        Tự động tạo ROI mặc định: toàn bộ phần giữa frame
        (phù hợp với đường 1 chiều điển hình nhìn từ trên).
        Bạn nên override bằng set_roi_zones() với polygon thực tế.
        """
        self._roi_zones = [np.array([
            [int(width * 0.1), int(height * 0.1)],
            [int(width * 0.9), int(height * 0.1)],
            [int(width * 0.9), int(height * 0.9)],
            [int(width * 0.1), int(height * 0.9)],
        ], np.int32)]

    def check(
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
        Kiểm tra xe đi ngược chiều trong frame hiện tại.
        Trả về list ViolationEvent mới phát hiện.
        """
        h, w = original_frame.shape[:2]
        frame_area = h * w
        violations = []

        # ── Bước 1: Update track history ──────────────────
        vx_scene, vy_scene = [], []

        for box, track_id, cls_id, conf in zip(
            vehicle_boxes, vehicle_track_ids, vehicle_classes, vehicle_confs
        ):
            x1, y1, x2, y2 = map(int, box)
            track_id = int(track_id)
            cls_name = vehicle_model.names[int(cls_id)]

            # Lọc bbox quá lớn (noise)
            if (x2-x1) * (y2-y1) / frame_area > self.cfg['max_bbox_area_pct']:
                continue

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # Kiểm tra ROI
            if self._roi_zones and not self._in_any_roi(cx, cy):
                continue

            # Merge Bike/Pedestrian track alias
            canonical_id = self._resolve_id(track_id, cls_name)

            # Lấy hoặc tạo track state
            state = self._get_or_create(canonical_id, cls_name)
            state['motion'].update(frame_number, cx, cy)
            state['class_name'] = cls_name  # cập nhật class mới nhất

            # Đóng góp vào scene flow nếu xe đang di chuyển
            vel = state['motion'].get_velocity(
                min_frames=state['cfg']['min_frames_to_judge']
            )
            if vel is not None:
                spd = (vel[0]**2 + vel[1]**2) ** 0.5
                if spd >= state['cfg']['min_speed_px_per_frame']:
                    vx_scene.append(vel[0])
                    vy_scene.append(vel[1])

        # Cập nhật scene flow từ tất cả xe frame này
        self._scene_flow.update(vx_scene, vy_scene)

        # ── Bước 2: Evaluate từng xe ──────────────────────
        for box, track_id, cls_id, conf in zip(
            vehicle_boxes, vehicle_track_ids, vehicle_classes, vehicle_confs
        ):
            x1, y1, x2, y2 = map(int, box)
            track_id = int(track_id)
            cls_name = vehicle_model.names[int(cls_id)]

            if (x2-x1) * (y2-y1) / frame_area > self.cfg['max_bbox_area_pct']:
                continue

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            if self._roi_zones and not self._in_any_roi(cx, cy):
                continue

            canonical_id = self._resolve_id(track_id, cls_name)
            if canonical_id not in self._tracks:
                continue

            state = self._tracks[canonical_id]
            c = state['cfg']

            vel = state['motion'].get_velocity(min_frames=c['min_frames_to_judge'])
            disp = state['motion'].get_displacement()
            spd = state['motion'].speed_norm()

            is_wrong = False
            cos_val = None

            if (
                vel is not None
                and spd >= c['min_speed_px_per_frame']
                and disp >= c['min_displacement_px']
            ):
                cos_val = self._scene_flow.cosine_similarity(vel[0], vel[1])
                if cos_val is not None:
                    # cos âm và đủ lớn → ngược chiều
                    if cos_val < -c['cosine_wrong_way_thresh']:
                        is_wrong = True

            state['last_cos'] = cos_val

            # Confirmation logic (hysteresis)
            if is_wrong:
                state['consec_wrong'] += 1
            else:
                state['consec_wrong'] = max(0, state['consec_wrong'] - c.get('confirm_decay', 1))

            # Confirm vi phạm
            if (
                not state['confirmed']
                and state['consec_wrong'] >= c['confirm_frames']
            ):
                # Kiểm tra cooldown
                last_vio = self._cooldown.get(canonical_id, -99999)
                if frame_number - last_vio >= self.cfg['cooldown_frames']:
                    state['confirmed'] = True
                    state['viol_frame'] = frame_number
                    self._cooldown[canonical_id] = frame_number
                    self.violated_ids.add(canonical_id)

                    evidence_path = self._save_evidence(
                        original_frame, x1, y1, x2, y2,
                        canonical_id, frame_number, cos_val
                    )
                    violations.append(ViolationEvent(
                        track_id=canonical_id,
                        vehicle_type=cls_name,
                        violation_type="wrong_way",
                        confidence=float(conf),
                        frame_number=frame_number,
                        bbox=[x1, y1, x2, y2],
                        evidence_path=evidence_path,
                    ))

            # Nếu đã confirm nhưng xe đã quay đầu → reset confirm
            # (để bắt lại nếu quay đầu tiếp)
            if state['confirmed'] and state['consec_wrong'] == 0:
                state['confirmed'] = False

        return violations

    def is_wrong_way(self, track_id: int) -> bool:
        """Kiểm tra nhanh track_id có đang bị đánh dấu ngược chiều không."""
        canonical = self._id_alias.get(track_id, track_id)
        return canonical in self.violated_ids

    def get_debug_info(self, track_id: int) -> str:
        """Trả về string debug cho track."""
        canonical = self._id_alias.get(track_id, track_id)
        state = self._tracks.get(canonical)
        if state is None:
            return "no state"
        cos = state['last_cos']
        cos_s = f"{cos:+.3f}" if cos is not None else "N/A"
        flow = self._scene_flow.get_flow_vector()
        flow_s = f"({flow[0]:+.2f},{flow[1]:+.2f})" if flow else "warming"
        vel = state['motion'].get_velocity()
        vel_s = f"({vel[0]:+.2f},{vel[1]:+.2f})" if vel else "N/A"
        return (
            f"cos={cos_s} consec={state['consec_wrong']} "
            f"vel={vel_s} flow={flow_s}"
        )

    def draw_zones(self, frame: np.ndarray):
        """Vẽ các vùng ROI wrong-way lên frame."""
        for zone in self._roi_zones:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [zone], (0, 80, 200))
            cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
            cv2.polylines(frame, [zone], isClosed=True,
                          color=(0, 140, 255), thickness=2, lineType=cv2.LINE_AA)

            # Label ở tâm
            cx = int(zone[:, 0].mean())
            cy = int(zone[:, 1].mean())
            # Vẽ mũi tên hướng flow (nếu có)
            flow = self._scene_flow.get_flow_vector()
            if flow is not None:
                fx, fy = flow
                mag = (fx**2 + fy**2) ** 0.5
                if mag > 0:
                    scale = 40 / mag
                    ex, ey = int(cx + fx * scale), int(cy + fy * scale)
                    cv2.arrowedLine(frame, (cx, cy), (ex, ey),
                                    (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.3)

            cv2.putText(frame, "WAY ZONE", (cx - 60, cy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)

    def draw_velocity_arrows(self, frame: np.ndarray,
                              vehicle_boxes, vehicle_track_ids,
                              vehicle_classes, vehicle_model):
        """
        Vẽ mũi tên velocity lên từng xe trong frame.
        Màu: xanh lá = đúng chiều, đỏ = ngược chiều, vàng = chưa xác định.
        """
        if not self.cfg.get('show_velocity_arrow', True):
            return

        for box, track_id, cls_id in zip(
            vehicle_boxes, vehicle_track_ids, vehicle_classes
        ):
            track_id = int(track_id)
            canonical = self._id_alias.get(track_id, track_id)
            state = self._tracks.get(canonical)
            if state is None:
                continue

            vel = state['motion'].get_velocity()
            if vel is None:
                continue

            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            vx, vy = vel
            mag = (vx**2 + vy**2) ** 0.5
            if mag < 0.5:
                continue

            scale = 25.0 / mag
            ex, ey = int(cx + vx * scale), int(cy + vy * scale)

            cos_val = state['last_cos']
            c = state['cfg']
            if cos_val is None:
                color = (200, 200, 0)
            elif cos_val < -c['cosine_wrong_way_thresh']:
                color = (0, 0, 255)   # Đỏ = ngược chiều
            else:
                color = (0, 220, 60)  # Xanh = đúng chiều

            cv2.arrowedLine(frame, (cx, cy), (ex, ey),
                            color, 2, cv2.LINE_AA, tipLength=0.35)

    def reset(self):
        self._tracks.clear()
        self._scene_flow.reset()
        self.violated_ids.clear()
        self._cooldown.clear()
        self._id_alias.clear()

    # ── Private helpers ─────────────────────────────────────

    def _in_any_roi(self, cx: float, cy: float) -> bool:
        for zone in self._roi_zones:
            if cv2.pointPolygonTest(zone, (float(cx), float(cy)), False) >= 0:
                return True
        return False

    def _resolve_id(self, track_id: int, cls_name: str) -> int:
        """
        Nếu merge_bike_pedestrian=True và track mới là Bike/Pedestrian
        có một track cũ gần đó thuộc class đối diện → dùng id cũ.
        """
        if not self.cfg.get('merge_bike_pedestrian', True):
            return track_id
        if cls_name not in ('Bike', 'Pedestrian'):
            return track_id
        if track_id in self._id_alias:
            return self._id_alias[track_id]
        # Nếu track_id đã có state → giữ nguyên
        if track_id in self._tracks:
            return track_id
        return track_id

    def _get_or_create(self, track_id: int, cls_name: str) -> dict:
        if track_id not in self._tracks:
            c = self.cfg.get('class_overrides', {}).get(cls_name, {})
            merged_cfg = {**self.cfg, **c}
            self._tracks[track_id] = {
                'motion':       _TrackMotion(history_len=self.cfg.get('track_history_len', 20)),
                'class_name':   cls_name,
                'cfg':          merged_cfg,
                'consec_wrong': 0,
                'confirmed':    False,
                'viol_frame':   None,
                'last_cos':     None,
            }
        return self._tracks[track_id]

    def _save_evidence(
        self, frame, x1, y1, x2, y2,
        track_id, frame_number, cos_val
    ) -> str:
        EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        h, w = frame.shape[:2]
        pad = 25
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
        crop = frame[cy1:cy2, cx1:cx2].copy()

        # Overlay thông tin lên ảnh bằng chứng
        cos_s = f"{cos_val:+.3f}" if cos_val is not None else "N/A"
        cv2.putText(crop, f"ID:{track_id} WRONG WAY",
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        cv2.putText(crop, f"cos={cos_s}",
                    (5, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 200, 255), 1)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"wrongway_id{track_id}_f{frame_number}_{ts}.jpg"
        cv2.imwrite(str(EVIDENCE_DIR / filename), crop)
        return filename


class WrongLaneChecker:
    """
    Phát hiện xe chạy sai làn đường tại ngã tư (dựa trên Polygon định sẵn & Tracking).
    Tích hợp WrongWayChecker (đi ngược chiều) vào cùng nhóm.
    """

    def __init__(self):
        # ── Lane & exit polygons ──
        self.lane_left  = [(19, 553), (51, 523), (243, 585), (93, 689), (48, 614)]
        self.lane_mid   = [(100, 694), (314, 541), (666, 541), (644, 651), (595, 784), (370, 783), (203, 783)]
        self.lane_right = [(606, 785), (669, 546), (813, 522), (792, 635), (699, 748)]

        self.exit_right    = [(647, 383), (759, 509), (841, 491), (843, 451), (737, 386)]
        self.exit_left     = [(79, 497), (161, 352), (71, 353), (19, 467), (54, 462)]
        self.exit_straight = [(163, 338), (179, 363), (345, 355), (265, 325), (170, 336)]

        # ── Wrong lane state ──
        self.lane_map:     dict = {}
        self.direction_map: dict = {}
        self.valid_ids:    set = set()
        self.violated_ids: set = set()
        self.last_positions: dict = {}
        self.sign_detected: bool = False

        # ── Wrong Way Checker (tích hợp) ──
        self.wrong_way_checker = WrongWayChecker()

    # ── Setter ROI cho WrongWayChecker ──────────────────────

    def set_wrong_way_roi(self, zones: list):
        """
        Định nghĩa vùng cần kiểm tra ngược chiều.
        zones: list of polygon [(x,y), ...]
        """
        self.wrong_way_checker.set_roi_zones(zones)

    def set_wrong_way_roi_from_frame(self, width: int, height: int):
        """Tự động tạo ROI wrong-way từ kích thước frame."""
        self.wrong_way_checker.set_roi_from_frame_size(width, height)

    # ── Main check (gộp wrong_lane + wrong_way) ─────────────

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

        # ── A. Kiểm tra sai làn tại ngã tư ─────────────────
        wrong_lane_viols = self._check_wrong_lane(
            lane_results, vehicle_boxes, vehicle_track_ids,
            vehicle_classes, vehicle_confs, vehicle_model,
            frame_number, original_frame,
        )
        violations.extend(wrong_lane_viols)

        # ── B. Kiểm tra xe đi ngược chiều ───────────────────
        wrong_way_viols = self.wrong_way_checker.check(
            vehicle_boxes, vehicle_track_ids,
            vehicle_classes, vehicle_confs,
            vehicle_model, frame_number, original_frame,
        )
        violations.extend(wrong_way_viols)

        return violations

    def _check_wrong_lane(
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
        """Logic gốc: phát hiện sai làn tại ngã tư."""
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

            # RECONNECT logic (giống code gốc)
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
                            if old_id in self.lane_map:
                                self.lane_map[track_id] = self.lane_map[old_id]
                            break

            self.last_positions[track_id] = (cx, cy, frame_number, (x1, y1, x2, y2), cls_name)

            if track_id not in self.valid_ids:
                r_left  = _box_in_poly_ratio(x1, y1, x2, y2, self.lane_left, h, w)
                r_mid   = _box_in_poly_ratio(x1, y1, x2, y2, self.lane_mid, h, w)
                r_right = _box_in_poly_ratio(x1, y1, x2, y2, self.lane_right, h, w)

                if max(r_left, r_mid, r_right) > 0.5:
                    self.valid_ids.add(track_id)

            if track_id not in self.valid_ids:
                continue

            if track_id not in self.lane_map:
                ratios = {
                    "left":     _box_in_poly_ratio(x1, y1, x2, y2, self.lane_left, h, w),
                    "straight": _box_in_poly_ratio(x1, y1, x2, y2, self.lane_mid, h, w),
                    "right":    _box_in_poly_ratio(x1, y1, x2, y2, self.lane_right, h, w),
                }
                best_lane = max(ratios, key=ratios.get)
                if ratios[best_lane] > 0.5:
                    self.lane_map[track_id] = best_lane

            if _in_poly(cx, cy, self.exit_straight):
                self.direction_map[track_id] = "straight"
            elif _in_poly(cx, cy, self.exit_left):
                self.direction_map[track_id] = "left"
            elif _in_poly(cx, cy, self.exit_right):
                self.direction_map[track_id] = "right"

            if track_id in self.lane_map and track_id in self.direction_map:
                if self.lane_map[track_id] != self.direction_map[track_id]:
                    if track_id not in self.violated_ids:
                        self.violated_ids.add(track_id)

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
        self.wrong_way_checker.reset()

    def draw_zones(self, frame: np.ndarray):
        """Vẽ origin lanes + exit zones + wrong-way zones."""
        # ── Wrong lane zones (ngã tư) ──
        if self.sign_detected:
            _draw_lane_zone(frame, self.lane_left,    (255, 180,  60), "<- LEFT",    alpha=0.18)
            _draw_lane_zone(frame, self.lane_mid,     ( 60, 220,  60), "^ STRAIGHT", alpha=0.18)
            _draw_lane_zone(frame, self.lane_right,   ( 60, 160, 255), "-> RIGHT",   alpha=0.18)

            _draw_lane_zone(frame, self.exit_left,     (200, 120,  20), "EXIT <-", alpha=0.15, border_color=(255,200,100))
            _draw_lane_zone(frame, self.exit_straight, ( 20, 160,  20), "EXIT ^",  alpha=0.15, border_color=(120,255,120))
            _draw_lane_zone(frame, self.exit_right,    ( 20, 100, 200), "EXIT ->", alpha=0.15, border_color=(120,180,255))

        # ── Wrong way zones ──
        self.wrong_way_checker.draw_zones(frame)

    # Expose wrong_way violated_ids cho pipeline annotate
    @property
    def wrong_way_violated_ids(self) -> set:
        return self.wrong_way_checker.violated_ids


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  4. PERSPECTIVE SPEED ESTIMATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import math as _math
import re as _re

_SPEED_MEDIAN_WINDOW        = 15
_SPEED_STABLE_FRAMES        = 7
_SPEED_STABLE_TOLERANCE_KMH = 4.0
_MAX_PHYSICAL_KMH           = 120.0
_MAX_FRAME_GAP              = 4
_UNLOCK_DELTA_KMH           = 6.0
_WARMUP_FRAMES              = 10
_VALID_SPEED_LIMITS         = {20, 30, 40, 50, 60, 70, 80, 90, 100, 120}
_MIN_SPEED_TO_CHECK         = 5
_VIOLATION_THRESHOLD        = 1.0
_COOLDOWN_FRAMES            = 90
_VALID_RADIUS_RATIO         = 1.0

_CAMERA_HEIGHT_M            = 6.0
_FOCAL_LENGTH_PX            = 400.0
_AVG_VEHICLE_HEIGHT_M       = 1.5
_NEAR_ZONE_RATIO            = 0.25
_FAR_ZONE_RATIO             = 0.05
_GLOBAL_SPEED_SCALE         = 0.8


class PerspectiveSpeedEstimator:
    """
    Ước tính tốc độ dựa trên Perspective Projection (Pinhole Camera Model).
    """

    def __init__(self, frame_w: int, frame_h: int, fps: float):
        self.fps       = fps
        self.frame_w   = frame_w
        self.frame_h   = frame_h
        self._tracks: dict = {}

    def estimate_speed(self, track_id: int, bbox: tuple, frame_idx: int) -> int:
        x1, y1, x2, y2 = bbox
        cx_px = (x1 + x2) / 2.0
        cy_px = (y1 + y2) / 2.0
        bh_px = float(y2 - y1)

        if bh_px < 8:
            return 0

        if track_id not in self._tracks:
            self._tracks[track_id] = {
                "history":      deque(maxlen=12),
                "speeds":       deque(maxlen=_SPEED_MEDIAN_WINDOW),
                "stable_count": 0,
                "locked_speed": None,
                "ema_speed":    0.0,
                "zone_frames":  0,
            }

        track = self._tracks[track_id]
        track["zone_frames"] += 1
        history = track["history"]

        if history:
            prev_cx, prev_cy, prev_bh, prev_idx = history[-1]
            frame_gap = frame_idx - prev_idx

            if frame_gap > _MAX_FRAME_GAP:
                self._reset_track(track, cx_px, cy_px, bh_px, frame_idx)
                return 0

            dt = frame_gap / self.fps if self.fps > 0 else 0.0
            if dt > 0:
                avg_bh  = (bh_px + prev_bh) / 2.0
                depth_m = (_CAMERA_HEIGHT_M * _FOCAL_LENGTH_PX) / max(avg_bh, 1.0)
                dpx     = _math.hypot(cx_px - prev_cx, cy_px - prev_cy)
                dist_m  = dpx * depth_m / _FOCAL_LENGTH_PX
                dist_m *= _GLOBAL_SPEED_SCALE
                inst_kmh = (dist_m / dt) * 3.6

                if inst_kmh > _MAX_PHYSICAL_KMH:
                    inst_kmh = track["speeds"][-1] if track["speeds"] else 0.0

                history.append((cx_px, cy_px, bh_px, frame_idx))

                if track["zone_frames"] <= _WARMUP_FRAMES:
                    return 0

                track["speeds"].append(inst_kmh)
                median_speed = float(np.median(track["speeds"]))

                if track["ema_speed"] == 0.0:
                    track["ema_speed"] = median_speed
                else:
                    track["ema_speed"] = 0.75 * track["ema_speed"] + 0.25 * median_speed

                smoothed = track["ema_speed"]
                self._update_lock(track, smoothed)

                if track["locked_speed"] is not None:
                    return int(round(track["locked_speed"]))
                return int(round(smoothed))

        history.append((cx_px, cy_px, bh_px, frame_idx))
        return 0

    def remove_track(self, track_id: int):
        self._tracks.pop(track_id, None)

    def reset(self):
        self._tracks.clear()

    def _reset_track(self, track, cx, cy, bh, frame_idx):
        track["history"].clear()
        track["speeds"].clear()
        track.update({
            "stable_count": 0, "locked_speed": None,
            "ema_speed": 0.0,  "zone_frames": 1,
        })
        track["history"].append((cx, cy, bh, frame_idx))

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


# Alias backward-compat
FisheyeSpeedEstimator = PerspectiveSpeedEstimator


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  5. SPEED LIMIT VIOLATION CHECKER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SpeedLimitChecker:
    """Phát hiện xe vượt quá tốc độ cho phép."""

    def __init__(self):
        self.current_speed_limit: int = 0
        self.sign_detected: bool = False
        self.violated_ids: set = set()
        self.cooldown_dict: dict = {}
        self.speed_map: dict = {}
        self._estimator: Optional[FisheyeSpeedEstimator] = None
        self._prev_ids: set = set()

    def _parse_speed_limit(self, sign_results, sign_model) -> Optional[int]:
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
            if limit_val is None:
                cls_id_val = int(box.cls[0])
                if cls_id_val in _VALID_SPEED_LIMITS:
                    limit_val = cls_id_val
            if limit_val and conf > best_conf:
                best_conf, best_limit = conf, limit_val
        return best_limit

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

        if self._estimator is None:
            h, w = original_frame.shape[:2]
            self._estimator = FisheyeSpeedEstimator(w, h, fps=float(source_fps))

        new_limit = self._parse_speed_limit(speed_sign_results, speed_sign_model)
        if new_limit:
            self.current_speed_limit = new_limit
            self.sign_detected = True

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

            if speed >= 0:
                self.speed_map[track_id] = speed

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

        cv2.putText(crop, f"ID:{track_id} {speed}km/h",
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(crop, f"Limit:{self.current_speed_limit}km/h",
                    (5, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"speed_id{track_id}_f{frame_number}_{speed}kmh_{ts}.jpg"
        cv2.imwrite(str(EVIDENCE_DIR / filename), crop)
        return filename

    def draw_safe_zone(self, frame: np.ndarray):
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
    poly = np.array(pts, np.int32).reshape((-1, 1, 2))
    bcolor = border_color if border_color else fill_color

    overlay = frame.copy()
    cv2.fillPoly(overlay, [poly], fill_color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.polylines(frame, [poly], isClosed=True, color=bcolor, thickness=2, lineType=cv2.LINE_AA)

    cx, cy = _poly_centroid(pts)
    text = f"{arrow} {label}" if arrow else label
    font = cv2.FONT_HERSHEY_DUPLEX
    scale, thick = 0.52, 1

    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(
        frame,
        (cx - tw // 2 - 3, cy - th - 3),
        (cx + tw // 2 + 3, cy + 3),
        (0, 0, 0), -1,
    )
    cv2.putText(frame, text, (cx - tw // 2, cy),
                font, scale, bcolor, thick, cv2.LINE_AA)


def _draw_zone_overlay(frame, polygon, color, label, alpha=0.25):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], color)
    frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=2)
    if label:
        origin = tuple(polygon[0][0])
        cv2.putText(frame, label, origin,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)