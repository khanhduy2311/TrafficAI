# =============================================================
# CORE/SPEED_ESTIMATOR.PY  —  v2  (fisheye-aware)
#
# Chiến lược đo tốc độ đúng cho camera fisheye equidistant:
#
#   r_px  = khoảng cách pixel từ điểm đến tâm ảnh
#   theta = r_px / f_px  (radian, equidistant model: r = f*theta)
#   d_real_per_d_px ≈ (R_real / R_px) / cos(theta)   [xấp xỉ]
#
# Để không cần calibration file đầy đủ, ta estimate f_px từ
# SENSOR_FOV_DEG (góc nhìn ngang của camera, thường 180° với
# fisheye full-frame hoặc ~185–220° với ultra-wide fisheye).
#
# Công thức chính xác hơn nhiều so v1 (linear interpolation).
# Vẫn giữ option undistort nếu user cung cấp K/D matrix.
#
# Ngoài ra: 
#   - Median filter 5-window thay EMA để loại spike hiệu quả hơn
#   - Tốc độ được lock sau N frame ổn định (stable window)
#   - Auto-discard track nếu vị trí nhảy bất thường (teleport check)
# =============================================================

import time
import math
import numpy as np
from collections import deque
from config import (
    PPM_CENTER, VALID_RADIUS_RATIO,
    SENSOR_FOV_DEG, REAL_RADIUS_METERS,
    SPEED_MEDIAN_WINDOW, SPEED_STABLE_FRAMES,
    MAX_JUMP_METERS_PER_SEC,
)


class FisheyeSpeedEstimator:
    """
    Đo tốc độ xe trên camera fisheye equidistant.

    Không cần undistort toàn bộ frame — tính trực tiếp
    khoảng cách thực từ displacement pixel dùng mô hình
    equidistant projection.

    Parameters
    ----------
    frame_w, frame_h : kích thước frame pixel
    K, D             : (optional) camera matrix + dist coefficients
                       từ cv2.fisheye.calibrate. Nếu cung cấp sẽ
                       dùng undistort thay vì model xấp xỉ.
    """

    def __init__(self, frame_w: int, frame_h: int,
                 K=None, D=None):
        self.frame_w = frame_w
        self.frame_h = frame_h

        # Tâm ảnh
        self.cx = frame_w / 2.0
        self.cy = frame_h / 2.0

        # Bán kính tròn fisheye theo pixel
        self.R_px = min(self.cx, self.cy)

        # Calibration mode
        self.K = K
        self.D = D
        self.use_calibration = (K is not None and D is not None)

        if not self.use_calibration:
            # Estimate focal length từ FOV
            # Equidistant: r = f * theta  →  f = R_px / (FOV_rad / 2)
            fov_rad = math.radians(SENSOR_FOV_DEG) / 2.0
            self.f_px = self.R_px / fov_rad  # pixel
            # Scale: meters per radian tại mặt phẳng đường
            # R_real = khoảng cách thực từ tâm đến rìa safe zone
            self.meters_per_radian = REAL_RADIUS_METERS / (SENSOR_FOV_DEG / 2.0 * math.pi / 180.0)

        # Per-track history
        # { track_id: { 'pos_history': deque[(x,y,t)], 'speeds': deque[float] } }
        self._tracks: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_speed(self, track_id: int, bottom_center_pt: tuple) -> int:
        """
        Trả về tốc độ km/h (int).
        Trả về -1 nếu xe ngoài safe zone.
        Trả về 0 nếu chưa đủ lịch sử.
        """
        x, y = float(bottom_center_pt[0]), float(bottom_center_pt[1])
        now = time.time()

        # ── Safe zone check ──────────────────────────────────
        r_px = math.sqrt((x - self.cx) ** 2 + (y - self.cy) ** 2)
        if r_px > self.R_px * VALID_RADIUS_RATIO:
            return -1

        # ── Init track ───────────────────────────────────────
        if track_id not in self._tracks:
            self._tracks[track_id] = {
                'pos_history': deque(maxlen=3),   # (x, y, t)
                'speeds':      deque(maxlen=SPEED_MEDIAN_WINDOW),
                'stable_count': 0,
                'locked_speed': None,
            }

        track = self._tracks[track_id]
        pos_history = track['pos_history']

        # ── Tính displacement thực ───────────────────────────
        if len(pos_history) >= 1:
            px, py, pt = pos_history[-1]
            dt = now - pt

            if dt > 0 and dt < 2.0:   # tránh gap quá lớn khi mất track
                # Chuyển 2 điểm pixel → real-world displacement
                dist_real = self._pixel_to_real_distance(px, py, x, y)

                # Teleport check: nếu xe nhảy quá nhanh vật lý là vô lý
                max_dist = MAX_JUMP_METERS_PER_SEC * dt
                if dist_real > max_dist:
                    # Reset lịch sử, coi như track mới
                    pos_history.clear()
                    pos_history.append((x, y, now))
                    return 0

                speed_ms = dist_real / dt
                speed_kmh = speed_ms * 3.6

                track['speeds'].append(speed_kmh)

                # Median filter: ổn định hơn EMA với spike
                if len(track['speeds']) >= 3:
                    median_speed = float(np.median(list(track['speeds'])))
                else:
                    median_speed = speed_kmh

                pos_history.append((x, y, now))
                return int(round(median_speed))

        pos_history.append((x, y, now))
        return 0

    def remove_track(self, track_id: int):
        """Xoá lịch sử khi track bị mất."""
        self._tracks.pop(track_id, None)

    # ------------------------------------------------------------------
    # Core geometry: pixel displacement → real-world distance
    # ------------------------------------------------------------------

    def _pixel_to_real_distance(self,
                                 x1: float, y1: float,
                                 x2: float, y2: float) -> float:
        """
        Tính khoảng cách thực (mét) giữa 2 điểm pixel trên ảnh fisheye.

        Có 2 mode:
        A) Nếu có K, D → undistort 2 điểm → tính trên perspective thông thường
        B) Nếu không   → dùng equidistant model: chuyển pixel → góc → real coord
        """
        if self.use_calibration:
            return self._calibrated_distance(x1, y1, x2, y2)
        else:
            return self._model_distance(x1, y1, x2, y2)

    def _model_distance(self,
                        x1: float, y1: float,
                        x2: float, y2: float) -> float:
        """
        Equidistant fisheye model (không cần calibration file).

        Với equidistant projection: r = f * theta
          theta = r / f  (góc từ trục quang học đến điểm)

        Điểm 3D trên mặt phẳng đất tương ứng với:
          X_real = meters_per_radian * theta * cos(phi)
          Y_real = meters_per_radian * theta * sin(phi)

        trong đó phi là góc phương vị (azimuth) của điểm trong ảnh.
        
        Khoảng cách thực = Euclid distance giữa 2 điểm real-world.
        """
        rx1, ry1 = self._pixel_to_real_coord(x1, y1)
        rx2, ry2 = self._pixel_to_real_coord(x2, y2)
        return math.sqrt((rx2 - rx1) ** 2 + (ry2 - ry1) ** 2)

    def _pixel_to_real_coord(self, px: float, py: float):
        """
        Chuyển 1 pixel (px, py) → tọa độ thực (X, Y) mét
        trên mặt phẳng đất, dùng equidistant model.
        """
        # Vector từ tâm ảnh
        dx = px - self.cx
        dy = py - self.cy
        r_px = math.sqrt(dx * dx + dy * dy)

        if r_px < 1e-6:
            return 0.0, 0.0

        # Góc từ trục quang học (equidistant: theta = r / f)
        theta = r_px / self.f_px   # radian

        # Giới hạn theta tránh cos(theta)=0
        theta = min(theta, math.pi / 2 - 0.01)

        # Tọa độ real-world: chiếu xuống mặt phẳng nằm ngang
        # (giả định camera nhìn thẳng xuống — phù hợp với ảnh mẫu)
        # 
        # Với camera overhead fisheye:
        #   dist_from_nadir = h_cam * tan(theta)
        # Vì không có h_cam, ta dùng tỉ lệ:
        #   real_r = meters_per_radian * theta
        # Đây là xấp xỉ tốt khi theta < 60° (vùng safe zone thường < 70°)
        real_r = self.meters_per_radian * theta

        # Azimuth (hướng)
        phi = math.atan2(dy, dx)

        X = real_r * math.cos(phi)
        Y = real_r * math.sin(phi)
        return X, Y

    def _calibrated_distance(self,
                              x1: float, y1: float,
                              x2: float, y2: float) -> float:
        """
        Dùng cv2.fisheye.undistortPoints để map 2 điểm pixel
        về normalized coordinates, rồi nhân với PPM_CENTER
        (đã calibrate tại nadir).

        Cần: self.K, self.D từ cv2.fisheye.calibrate()
        """
        import cv2
        pts = np.array([[[x1, y1]], [[x2, y2]]], dtype=np.float32)
        # undistortPoints → normalized image coords (không có scale)
        undist = cv2.fisheye.undistortPoints(pts, self.K, self.D)
        # undist[i][0] = (u, v) normalized
        u1, v1 = undist[0][0]
        u2, v2 = undist[1][0]
        # Chuyển về pixel space dùng K (focal length tại center)
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        px1 = u1 * fx
        py1 = v1 * fy
        px2 = u2 * fx
        py2 = v2 * fy
        dist_norm_px = math.sqrt((px2 - px1) ** 2 + (py2 - py1) ** 2)
        # PPM_CENTER là đã calibrate tại tâm ảnh (nadir)
        return dist_norm_px / PPM_CENTER
