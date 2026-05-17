# =============================================================
# PIPELINE.PY  v2
# Thay đổi chính:
#   1. Speed limit: detect 1 lần từ N frame random đầu video
#      (camera góc cố định → biển báo luôn ở cùng vị trí)
#   2. Dùng FisheyeSpeedEstimator thay DynamicPixelSpeed
#   3. Cleanup track khi tracker mất ID
# =============================================================

import cv2
import os
import random
from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from pathlib import Path

import config as cfg
from core.speed_estimator      import FisheyeSpeedEstimator
from core.speed_limit_manager  import SpeedLimitManager
from core.violation_detector   import ViolationDetector
from output.annotator          import Annotator
from output.evidence_saver     import EvidenceSaver
from output.report_logger      import ReportLogger


def _detect_speed_limit_once(cap, sign_model, total_frames: int) -> int:
    """
    Sample SIGN_DETECT_NFRAMES frame ngẫu nhiên từ 1/4 đầu video,
    chạy predict, vote majority → trả về limit km/h.
    Fallback về DEFAULT_SPEED_LIMIT nếu không tìm thấy.

    Camera không di chuyển → biển báo luôn cùng vị trí
    → 1 lần detect đủ cho toàn video.
    """
    manager = SpeedLimitManager()
    # Chỉ sample trong 25% đầu video để tránh frame không có biển
    sample_range = max(1, total_frames // 4)
    sample_frames = sorted(random.sample(
        range(1, sample_range + 1),
        min(cfg.SIGN_DETECT_NFRAMES, sample_range)
    ))

    votes = []
    for fno in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        ret, frame = cap.read()
        if not ret:
            continue
        results = sign_model.predict(
            frame,
            imgsz=cfg.IMG_SIZE,
            conf=cfg.SIGN_DETECT_CONF,
            verbose=False
        )
        updated = manager.update_from_detections(results)
        if updated:
            votes.append(manager.get_current_limit())

    # Reset về đầu video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if votes:
        # Majority vote (nếu tất cả giống nhau thì votes[0])
        limit = max(set(votes), key=votes.count)
        print(f"  [SpeedLimit] Phát hiện giới hạn tốc độ: {limit} km/h "
              f"(từ {len(votes)}/{len(sample_frames)} frame sample)")
        return limit

    print(f"  [SpeedLimit] Không phát hiện biển → dùng default {cfg.DEFAULT_SPEED_LIMIT} km/h")
    return cfg.DEFAULT_SPEED_LIMIT


def run(source: str, output_dir: str = "output",
        K=None, D=None):
    """
    K, D: camera matrix + distortion coefficients từ cv2.fisheye.calibrate.
          Nếu None → dùng equidistant model xấp xỉ.
    """
    print("Đang load model xe...")
    vehicle_model = YOLO(cfg.VEHICLE_MODEL_PATH)

    print("Đang load model biển báo...")
    sign_model = YOLO(cfg.SIGN_MODEL_PATH)

    # ── Mở video ─────────────────────────────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Không mở được video: {source}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {frame_w}x{frame_h} @ {fps:.1f}fps, {total} frames")

    # ── Detect speed limit 1 lần ─────────────────────────────
    if cfg.SIGN_DETECT_ONCE:
        current_limit = _detect_speed_limit_once(cap, sign_model, total)
    else:
        current_limit = cfg.DEFAULT_SPEED_LIMIT

    # ── Khởi tạo modules ─────────────────────────────────────
    speed_estimator    = FisheyeSpeedEstimator(frame_w, frame_h, K=K, D=D)
    limit_manager      = SpeedLimitManager(default_limit=current_limit)
    violation_detector = ViolationDetector(cooldown_frames=90)
    annotator          = Annotator(frame_w, frame_h)
    evidence_saver     = EvidenceSaver()
    report_logger      = ReportLogger()

    # ── Setup output ─────────────────────────────────────────
    run_dir  = increment_path(Path(output_dir) / "run", exist_ok=False)
    os.makedirs(run_dir, exist_ok=True)
    out_path = str(run_dir / "result_violation.mp4")
    writer   = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (frame_w, frame_h)
    )

    tracker_cfg = cfg.TRACKER_CONFIG if os.path.exists(cfg.TRACKER_CONFIG) else "bytetrack.yaml"

    print(f"Bắt đầu xử lý... Output: {out_path}")
    frame_idx        = 0
    total_violations = 0
    prev_track_ids   = set()   # để cleanup stale tracks

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if frame_idx % 100 == 0:
                pct = frame_idx / total * 100 if total else 0
                print(f"  Frame {frame_idx}/{total} ({pct:.1f}%) | Vi phạm: {total_violations}")

            # ── Speed limit: chỉ update nếu không dùng once mode ──
            if not cfg.SIGN_DETECT_ONCE and frame_idx % 5 == 0:
                sign_results = sign_model.predict(
                    frame, imgsz=cfg.IMG_SIZE, conf=cfg.SIGN_DETECT_CONF, verbose=False
                )
                limit_manager.update_from_detections(sign_results)

            current_limit = limit_manager.get_current_limit()

            # ── Draw safe zone ───────────────────────────────
            annotator.draw_fisheye_zones(frame)

            # ── Track xe ─────────────────────────────────────
            results = vehicle_model.track(
                frame,
                imgsz=cfg.IMG_SIZE,
                conf=cfg.CONF_THRESHOLD,
                tracker=tracker_cfg,
                persist=True,
                verbose=False,
            )

            has_violation_this_frame = False
            current_track_ids = set()

            if results[0].boxes.id is not None:
                track_ids  = results[0].boxes.id.int().cpu().tolist()
                bboxes     = results[0].boxes.xyxy.cpu().numpy()
                class_ids  = results[0].boxes.cls.int().cpu().tolist()

                for i, track_id in enumerate(track_ids):
                    cls_id = class_ids[i]
                    if cls_id not in cfg.TARGET_CLASSES:
                        continue

                    current_track_ids.add(track_id)
                    cls_name = cfg.TARGET_CLASSES[cls_id]
                    x1, y1, x2, y2 = map(int, bboxes[i])
                    bbox   = (x1, y1, x2, y2)
                    anchor = (int((x1 + x2) / 2), y2)

                    # ── Đo tốc độ (fisheye-aware) ────────────
                    speed = speed_estimator.estimate_speed(track_id, anchor)

                    # ── Kiểm tra vi phạm ─────────────────────
                    is_violation = False
                    if speed > 0:   # -1 = ngoài zone, 0 = chưa đủ history
                        event = violation_detector.check(
                            track_id=track_id,
                            class_name=cls_name,
                            speed_kmh=speed,
                            limit_kmh=current_limit,
                            bbox=bbox,
                            frame_idx=frame_idx,
                        )
                        if event:
                            is_violation = True
                            has_violation_this_frame = True
                            total_violations += 1
                            img_path = evidence_saver.save(frame, event)
                            event.image_path = img_path
                            report_logger.log(event)
                            print(f"  [VI PHAM] ID:{track_id} {cls_name} "
                                  f"{speed}km/h > {current_limit}km/h | {img_path}")

                    annotator.draw_vehicle(frame, track_id, cls_name,
                                           bbox, speed, is_violation)

            # ── Cleanup track đã mất ─────────────────────────
            lost_ids = prev_track_ids - current_track_ids
            for tid in lost_ids:
                speed_estimator.remove_track(tid)
            prev_track_ids = current_track_ids

            if has_violation_this_frame:
                annotator.draw_violation_flash(frame)

            annotator.draw_speed_limit_hud(frame, current_limit)
            writer.write(frame)

    except KeyboardInterrupt:
        print("Dừng sớm theo yêu cầu.")
    finally:
        cap.release()
        writer.release()

    summary = report_logger.summary()
    print(f"\n{'='*50}")
    print(f"DONE! Tổng vi phạm: {summary.get('total_violations', 0)}")
    print(f"Theo loại xe: {summary.get('by_class', {})}")
    print(f"Video output: {out_path}")
    print(f"{'='*50}")
    return out_path
