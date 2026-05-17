# =============================================================
# CORE/SPEED_LIMIT_MANAGER.PY
# Đọc kết quả từ model biển báo → lưu giới hạn tốc độ hiện tại
# =============================================================

import re
from config import DEFAULT_SPEED_LIMIT


class SpeedLimitManager:
    """
    Nhận diện và lưu giới hạn tốc độ từ kết quả detect biển báo.

    Cách dùng:
        manager = SpeedLimitManager()
        manager.update_from_detections(sign_results)  # mỗi frame
        limit = manager.get_current_limit()           # lấy giá trị hiện tại
    """

    # Các mức tốc độ hợp lệ ở Việt Nam (km/h)
    VALID_LIMITS = {15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120}

    def __init__(self, default_limit: int = DEFAULT_SPEED_LIMIT):
        self.current_limit  = default_limit
        self.default_limit  = default_limit
        self.frames_since_sign = 0
        self.sign_timeout   = 300  # frames — reset về default sau ~10s nếu không thấy biển

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_from_detections(self, sign_results) -> bool:
        """
        Nhận results từ model biển báo (ultralytics format).
        Trả về True nếu giới hạn vừa được cập nhật.

        sign_results: kết quả từ sign_model.predict() hoặc sign_model.track()
        """
        self.frames_since_sign += 1

        # Timeout — quay về default nếu lâu không thấy biển
        if self.frames_since_sign > self.sign_timeout:
            self.current_limit = self.default_limit

        if sign_results is None or len(sign_results) == 0:
            return False

        boxes = sign_results[0].boxes
        if boxes is None or len(boxes) == 0:
            return False

        # Lấy biển có confidence cao nhất
        best_conf = 0.0
        best_limit = None

        for i, box in enumerate(boxes):
            conf = float(box.conf[0])
            if conf < 0.5:
                continue

            # Thử đọc class name hoặc label từ model
            limit_value = self._extract_limit_from_box(sign_results[0], i)
            if limit_value and conf > best_conf:
                best_conf  = conf
                best_limit = limit_value

        if best_limit:
            self.current_limit     = best_limit
            self.frames_since_sign = 0
            return True

        return False

    def get_current_limit(self) -> int:
        return self.current_limit

    def override(self, limit_kmh: int):
        """Hard-code giới hạn tốc độ thủ công (dùng khi test)."""
        self.current_limit = limit_kmh

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_limit_from_box(self, result, idx: int):
        """
        Cố gắng đọc con số km/h từ class name của box.

        Model biển báo thường có class name dạng:
          - "speed_50", "50kmh", "limit_60", "sign_80", v.v.
        Hoặc nếu class name không có số → dùng class id ánh xạ thủ công.
        """
        try:
            names = result.names  # dict {id: name}
            cls_id = int(result.boxes.cls[idx])
            class_name = names.get(cls_id, "")

            # Tìm số trong class name
            numbers = re.findall(r'\d+', class_name)
            for n in numbers:
                val = int(n)
                if val in self.VALID_LIMITS:
                    return val

            # Fallback: dùng class id trực tiếp nếu là số tốc độ hợp lệ
            if cls_id in self.VALID_LIMITS:
                return cls_id

        except Exception:
            pass

        return None
