# =============================================================
# CORE/VIOLATION_DETECTOR.PY
# So sánh tốc độ xe với giới hạn → quyết định vi phạm
# =============================================================

from dataclasses import dataclass, field
from typing import Optional
import time
from config import VIOLATION_THRESHOLD, MIN_SPEED_TO_CHECK


@dataclass
class ViolationEvent:
    track_id:    int
    class_name:  str
    speed_kmh:   int
    limit_kmh:   int
    bbox:        tuple          # (x1, y1, x2, y2)
    frame_idx:   int
    timestamp:   float = field(default_factory=time.time)
    image_path:  Optional[str] = None   # điền sau khi crop


class ViolationDetector:
    """
    Logic đơn giản: speed > limit * threshold → vi phạm.

    Có cooldown per track_id để tránh log cùng xe nhiều lần liên tiếp.
    """

    def __init__(self, cooldown_frames: int = 90):
        """
        cooldown_frames: số frame tối thiểu giữa 2 lần log cùng 1 xe.
        90 frames ≈ 3 giây ở 30fps — đủ để ghi nhận 1 lần/lần vượt tốc.
        """
        self.cooldown_frames = cooldown_frames
        self._last_violation: dict[int, int] = {}   # track_id → frame_idx

    def check(
        self,
        track_id:   int,
        class_name: str,
        speed_kmh:  int,
        limit_kmh:  int,
        bbox:       tuple,
        frame_idx:  int,
    ) -> Optional[ViolationEvent]:
        """
        Trả về ViolationEvent nếu vi phạm, ngược lại None.
        """
        # Bỏ qua tốc độ quá thấp (xe đứng yên hoặc nhiễu)
        if speed_kmh < MIN_SPEED_TO_CHECK:
            return None

        # Kiểm tra vi phạm
        if speed_kmh <= limit_kmh * VIOLATION_THRESHOLD:
            return None

        # Cooldown — tránh log cùng xe nhiều lần
        last = self._last_violation.get(track_id, -self.cooldown_frames - 1)
        if frame_idx - last < self.cooldown_frames:
            return None

        # Ghi nhận và trả về event
        self._last_violation[track_id] = frame_idx
        return ViolationEvent(
            track_id=track_id,
            class_name=class_name,
            speed_kmh=speed_kmh,
            limit_kmh=limit_kmh,
            bbox=bbox,
            frame_idx=frame_idx,
        )
