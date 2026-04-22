"""
He thong phat hien vuot den do - Ban hoan chinh
================================================
Su dung: python red_light_detector.py --source video.mp4 --light-weights models/light.pt --vehicle-weights models/vehicle.pt

Cac cai tien so voi ban cu:
  - Track tren original_frame (khong phai frame da ve overlay)
  - Reset vehicles_in_zone1 khi xe roi khoi ca 2 vung
  - Cooldown timer tranh bat trung
  - Frame threshold (3 frame lien tiep) tranh false positive
  - Trang thai den Unknown khi khong detect duoc
  - ByteTrack thay vi BotSORT
  - Luu anh bang chung (crop + toan canh) va JSON log
  - Class map day du cho tat ca trang thai den
  - Output thu muc tu dong tao
"""

import argparse
import cv2
import json
import hashlib
import numpy as np
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO


# ──────────────────────────────────────────────────────────
# 1. ARGUMENT PARSER
# ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="He thong phat hien vuot den do - Ban hoan chinh",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--source",           required=True,                            help="Duong dan video dau vao")
    parser.add_argument("--light-weights",    default="models/yolo11n_opt.pt",                help="Model nhan dien den giao thong")
    parser.add_argument("--vehicle-weights",  default="models/Xe.pt",              help="Model tracking xe")
    parser.add_argument("--output",           default="outputs/results/output.mp4",     help="Duong dan video ket qua")
    parser.add_argument("--evidence-dir",     default="outputs/violations",             help="Thu muc luu bang chung vi pham")
    parser.add_argument("--log-file",         default="outputs/violations/log.json",    help="File JSON luu log vi pham")

    parser.add_argument("--roi1",             type=str,  default="",                    help="Toa do Zone 1 (Khu vuc truoc vach). Vd: '100,500,...'")
    parser.add_argument("--roi2",             type=str,  default="",                    help="Toa do Zone 2 (Khu vuc vi pham). Vd: '100,550,...'")
    parser.add_argument("--light-roi",        type=str,  default=[], nargs='+',        help="Toa do cac vung quet den (ho tro nhieu vung). Vd: '--light-roi \"x1,y1...\" \"x3,y3...\"'")

    parser.add_argument("--light-max-y",      type=float, default=0.5,                 help="Chi nhan den o phan tren man hinh (0.0-1.0)")
    parser.add_argument("--light-conf",       type=float, default=0.25,                help="Nguong confidence cho den do (khuyen nghi 0.3-0.5)")
    parser.add_argument("--vehicle-conf",     type=float, default=0.4,                 help="Nguong confidence cho phat hien xe")
    parser.add_argument("--frame-threshold",  type=int,   default=3,                   help="So frame lien tiep trong Zone 2 de xac nhan vi pham")
    parser.add_argument("--cooldown-sec",     type=float, default=5.0,                 help="Thoi gian cho giua 2 vi pham cung mot xe (giay)")
    parser.add_argument("--smooth-window",    type=int,   default=10,                  help="Do dai cua so lam muot den (so frame de bo phieu)")
    parser.add_argument("--save-clip",        action="store_true",                     help="Luu clip ngan 5 giay cho moi vi pham (RAM cao hon)")
    parser.add_argument("--no-view",          action="store_true",                     help="Khong hien thi cua so xem truc tiep")
    parser.add_argument("--hide-light-roi",   action="store_true",                     help="Khong hien thi vung quet den tren video")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────
# 2. TIEN ICH
# ──────────────────────────────────────────────────────────

def get_light_class_map(model) -> dict:
    """
    Xay dung mapping { 'red': [id,...], 'green': [id,...], 'yellow': [id,...], 'off': [id,...] }
    tu ten cac class trong model den giao thong.
    """
    mapping = {"red": [], "green": [], "yellow": [], "off": []}
    for cls_id, cls_name in model.names.items():
        name = str(cls_name).lower()
        for key in mapping:
            if key in name:
                mapping[key].append(cls_id)
    # Bao cao
    for k, v in mapping.items():
        if not v:
            print(f"  WARNING: Khong tim thay class '{k}' trong model den. Kiem tra lai model!")
        else:
            print(f"  Den '{k}' -> class_id(s): {v}")
    return mapping


def parse_polygon(roi_str: str, default_poly: np.ndarray) -> np.ndarray:
    """Parse chuoi toa do 'x1,y1,x2,y2,...' thanh numpy polygon."""
    if not roi_str:
        return default_poly
    try:
        pts = list(map(int, roi_str.strip().split(",")))
        if len(pts) % 2 != 0 or len(pts) < 6:
            print("WARNING: ROI string sai dinh dang, dung default.")
            return default_poly
        return np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    except ValueError:
        print("WARNING: Khong parse duoc ROI string, dung default.")
        return default_poly


def sha256_of_file(path: Path) -> str:
    """Tinh SHA-256 checksum cua file de dam bao tinh toan ven bang chung."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def draw_zone(frame, polygon, color, label, alpha=0.25):
    """Ve vung polygon ban trong mo + vien day."""
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], color)
    frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=2)
    
    if label:
        origin = tuple(polygon[0][0])
        cv2.putText(frame, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────
# 3. MAIN
# ──────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # --- Kiem tra dau vao ---
    source_path = Path(args.source).resolve()
    if not source_path.exists():
        print(f"ERROR: Khong tim thay video: {source_path}")
        return

    # --- Tao thu muc dau ra ---
    evidence_dir = Path(args.evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    print("\n[1/4] Load models...")
    light_model   = YOLO(args.light_weights)
    vehicle_model = YOLO(args.vehicle_weights)
    print("  Light model classes :")
    light_class_map = get_light_class_map(light_model)

    # --- Mo video ---
    print("\n[2/4] Mo video...")
    cap    = cv2.VideoCapture(str(source_path))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  {width}x{height} @ {fps}fps, ~{total} frames")

    out = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height)
    )

    # --- Dinh nghia ROI ---
    print("\n[3/4] Cau hinh vung ROI...")
    default_roi1 = np.array([
        [0,     int(height * 0.50)],
        [width, int(height * 0.50)],
        [width, int(height * 0.60)],
        [0,     int(height * 0.60)],
    ], dtype=np.int32).reshape((-1, 1, 2))

    default_roi2 = np.array([
        [0,     int(height * 0.60)],
        [width, int(height * 0.60)],
        [width, height],
        [0,     height],
    ], dtype=np.int32).reshape((-1, 1, 2))

    roi1_polygon = parse_polygon(args.roi1, default_roi1)
    roi2_polygon = parse_polygon(args.roi2, default_roi2)
    
    # Light ROIs (Ho tro nhieu vung)
    light_roi_polygons = []
    if args.light_roi:
        for roi_str in args.light_roi:
            poly = parse_polygon(roi_str, None)
            if poly is not None:
                light_roi_polygons.append(poly)

    light_y_threshold = int(height * args.light_max_y)
    cooldown_frames   = int(args.cooldown_sec * fps)

    # --- State tracking ---
    # track_id -> trang thai
    vehicles_in_zone1   = set()           # Da di qua zone1
    zone1_light_state   = {}              # track_id -> trang thai den LUC XE DI QUA ROI1
    violators           = set()           # Da bi ghi nhan vi pham (dung cho mau sac)
    zone2_frame_count   = defaultdict(int) # So frame lien tiep trong zone2
    last_violation_frame = {}             # track_id -> frame_count cuoi cung vi pham
    violation_log       = []              # Danh sach vi pham de xuat JSON
    
    # Bo loc lam muot trang thai den
    light_history       = deque(maxlen=args.smooth_window)

    # Buffer de luu clip (neu --save-clip)
    CLIP_BUFFER_SEC = 3  # Luu 3 giay truoc vi pham
    clip_buffer: deque = deque(maxlen=CLIP_BUFFER_SEC * fps) if args.save_clip else None

    print("\n[4/4] Bat dau xu ly video...")
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # ── QUAN TRONG: Luu frame goc truoc khi ve bat cu thu gi ──
        original_frame = frame.copy()

        if args.save_clip:
            clip_buffer.append(original_frame.copy())

        # ──────────────────────────────────────────────
        # BUOC 1: NHAN DIEN DEN GIAO THONG
        # Chay tren original_frame (sach, khong overlay)
        # ──────────────────────────────────────────────
        light_results = light_model(original_frame, verbose=False)[0]
        light_status  = "unknown"   # Trang thai mac dinh: chua biet

        for box in light_results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_y = (y1 + y2) // 2

            # Bo qua den o qua nua duoi man hinh (co the la den hau xe, ao trong)
            if center_y > light_y_threshold:
                continue

            # 1. Kiem tra vi tri so voi tat ca Light ROIs
            is_in_light_roi = False
            active_roi_idx = -1
            if not light_roi_polygons:
                is_in_light_roi = True # Neu khong set ROI thi mac dinh la True
            else:
                center_x = (x1 + x2) // 2
                for idx, poly in enumerate(light_roi_polygons):
                    if cv2.pointPolygonTest(poly, (float(center_x), float(center_y)), False) >= 0:
                        is_in_light_roi = True
                        active_roi_idx = idx + 1
                        break

            # 2. Xac dinh mau sac va ve Box (Ve tat ca de debug)
            if cls_id in light_class_map["red"]:
                draw_color = (0, 0, 255)
            elif cls_id in light_class_map["yellow"]:
                draw_color = (0, 165, 255)
            elif cls_id in light_class_map["green"]:
                draw_color = (0, 255, 0)
            else:
                draw_color = (128, 128, 128)

            label = f"{light_model.names[cls_id]} {conf:.2f}"
            if not is_in_light_roi:
                label += " (OUT)"
            if conf < args.light_conf:
                label += " (L-CONF)"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
            cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 2, cv2.LINE_AA)

            # 3. CAP NHAT TRANG THAI (Chi khi thoa man tat ca dieu kien)
            if conf >= args.light_conf and is_in_light_roi:
                cls_name = light_model.names[cls_id]
                # Log debug nhanh
                if cls_id in light_class_map["yellow"] or cls_id in light_class_map["red"]:
                     roi_msg = f"inside Light ROI #{active_roi_idx}" if active_roi_idx != -1 else "inside Light ROI"
                     print(f"  [DEBUG] Frame {frame_count}: Detected {cls_name} (conf={conf:.2f}) {roi_msg}")
                
                if cls_id in light_class_map["red"]:
                    light_status = "red"
                elif cls_id in light_class_map["yellow"]:
                    if light_status != "red":
                        light_status = "yellow"
                elif cls_id in light_class_map["green"]:
                    if light_status not in ("red", "yellow"):
                        light_status = "green"

        # --- AP DUNG BO LOC LAM MUOT (SMOOTHING) ---
        # Thuc hien sau khi da quet tat ca cac box trong frame
        light_history.append(light_status)
        # Trang thai den thuc te la gia tri xuat hien nhieu nhat trong history
        light_status = max(set(light_history), key=list(light_history).count)

        is_red = (light_status == "red")

        # ──────────────────────────────────────────────
        # Ve zones
        # ──────────────────────────────────────────────
        zone2_color = (0, 0, 255) if is_red else (0, 200, 0)
        draw_zone(frame, roi1_polygon, (255, 200, 0), "Zone 1")
        draw_zone(frame, roi2_polygon, zone2_color,   "Zone 2")
        if light_roi_polygons and not args.hide_light_roi:
            for i, p in enumerate(light_roi_polygons):
                draw_zone(frame, p, (255, 255, 255), label="", alpha=0.05)

        # Hien thi trang thai den
        STATUS_COLORS = {
            "red":     (0, 0, 255),
            "yellow":  (0, 165, 255),
            "green":   (0, 255, 0),
            "off":     (150, 150, 150),
            "unknown": (200, 200, 200),
        }
        status_color = STATUS_COLORS.get(light_status, (200, 200, 200))
        status_label = {
            "red":     "DEN DO  [RED]",
            "yellow":  "DEN VANG  [YELLOW]",
            "green":   "DEN XANH  [GREEN]",
            "off":     "DEN TAT  [OFF]",
            "unknown": "CHUA XAC DINH  [?]",
        }.get(light_status, "?")

        cv2.putText(frame, status_label, (30, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3, cv2.LINE_AA)

        # ──────────────────────────────────────────────
        # BUOC 2: TRACKING XE
        # Chay tren original_frame (sach) - rat quan trong!
        # ──────────────────────────────────────────────
        vehicle_results = vehicle_model.track(
            original_frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=args.vehicle_conf,
            verbose=False
        )[0]

        current_track_ids = set()  # De biet xe nao con trong frame

        if vehicle_results.boxes.id is not None:
            boxes     = vehicle_results.boxes.xyxy.cpu().numpy()
            track_ids = vehicle_results.boxes.id.int().cpu().numpy()
            classes   = vehicle_results.boxes.cls.int().cpu().numpy()
            confs     = vehicle_results.boxes.conf.cpu().numpy()

            for box, track_id, cls_id, det_conf in zip(boxes, track_ids, classes, confs):
                x1, y1, x2, y2 = map(int, box)
                track_id = int(track_id)
                current_track_ids.add(track_id)

                # Dung bottom-center lam diem kiem tra vi tri
                bc_x = (x1 + x2) // 2
                bc_y = y2
                cv2.circle(frame, (bc_x, bc_y), 5, (255, 255, 0), -1)

                # ── Kiem tra vi tri xe so voi cac zone ──
                inside_roi1 = cv2.pointPolygonTest(roi1_polygon, (float(bc_x), float(bc_y)), False) >= 0
                inside_roi2 = cv2.pointPolygonTest(roi2_polygon, (float(bc_x), float(bc_y)), False) >= 0

                # Cap nhat lich su zone1 + GHI NHAN TRANG THAI DEN
                # Chi ghi nhan trang thai den LAN DAU xe buoc vao ROI1
                if inside_roi1:
                    if track_id not in vehicles_in_zone1:
                        # Lan dau xe cham vach dung -> ghi nhan trang thai den
                        zone1_light_state[track_id] = light_status
                        print(f"  [ZONE1] Frame {frame_count}: ID:{track_id} vao ROI1, den={light_status}")
                    vehicles_in_zone1.add(track_id)

                # Reset trang thai zone1 CHI KHI xe chua tung vao zone2
                # (Neu xe da qua zone1, dang tren duong vao zone2 nhung co gap giua 2 zone,
                #  khong duoc xoa de tranh bo lot vi pham)
                if not inside_roi1 and not inside_roi2:
                    if zone2_frame_count[track_id] == 0 and track_id not in violators:
                        vehicles_in_zone1.discard(track_id)
                        zone1_light_state.pop(track_id, None)  # Xoa trang thai den cu

                # Cap nhat frame counter zone2 (lien tiep)
                if inside_roi2:
                    zone2_frame_count[track_id] += 1
                else:
                    zone2_frame_count[track_id] = 0

                # ──────────────────────────────────────────────
                # BUOC 3: KIEM TRA DIEU KIEN VI PHAM
                # Logic: Xe vi pham khi BUOC QUA VACH (ROI1) LUC DEN DO
                #   -> Dieu kien chinh la den DO tai thoi diem qua ROI1
                #   -> KHONG phai den do tai thoi diem xe o ROI2
                # ──────────────────────────────────────────────
                consecutive_in_z2 = zone2_frame_count[track_id]
                last_vio_frame    = last_violation_frame.get(track_id, -999999)
                cooldown_ok       = (frame_count - last_vio_frame) > cooldown_frames

                # Lay trang thai den LUC XE DI QUA ROI1 (vach dung)
                crossed_on_red = zone1_light_state.get(track_id) == "red"

                violation_detected = (
                    crossed_on_red                              # 1. Den DO luc xe buoc qua vach (ROI1)
                    and inside_roi2                             # 2. Xe dang trong vung vi pham (ROI2)
                    and track_id in vehicles_in_zone1          # 3. Xe da di tu Zone1 vao (dung huong)
                    and consecutive_in_z2 >= args.frame_threshold  # 4. Du so frame lien tiep
                    and cooldown_ok                            # 5. Het cooldown
                )

                if violation_detected:
                    violators.add(track_id)
                    last_violation_frame[track_id] = frame_count
                    zone2_frame_count[track_id] = 0  # Reset counter sau khi ghi nhan vi pham
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cls_name      = vehicle_model.names[cls_id]
                    light_at_cross = zone1_light_state.get(track_id, "unknown")

                    print(f"[Frame {frame_count}] *** VI PHAM *** ID:{track_id} | {cls_name} | Den luc qua vach: {light_at_cross} | {timestamp_str}")

                    # ── Luu anh bang chung ──
                    # 1. Anh crop xe vi pham (them padding)
                    pad    = 20
                    cx1    = max(0, x1 - pad)
                    cy1    = max(0, y1 - pad)
                    cx2    = min(width,  x2 + pad)
                    cy2    = min(height, y2 + pad)
                    crop   = original_frame[cy1:cy2, cx1:cx2]
                    crop_path = evidence_dir / f"vio_id{track_id}_f{frame_count}_{timestamp_str}_crop.jpg"
                    cv2.imwrite(str(crop_path), crop)

                    # 2. Anh toan canh co ve thong tin vi pham
                    evidence_frame = original_frame.copy()
                    cv2.rectangle(evidence_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(evidence_frame, f"VIOLATION ID:{track_id}", (x1, max(0, y1 - 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(evidence_frame, f"DEN DO LUC QUA VACH - {timestamp_str}", (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
                    full_path = evidence_dir / f"vio_id{track_id}_f{frame_count}_{timestamp_str}_full.jpg"
                    # (full_path da duoc ghi o dong tiep theo)
                    cv2.imwrite(str(full_path), evidence_frame)

                    # 3. Clip ngan (neu bat --save-clip)
                    clip_path = None
                    if args.save_clip and clip_buffer:
                        clip_path = evidence_dir / f"vio_id{track_id}_f{frame_count}_{timestamp_str}_clip.mp4"
                        clip_writer = cv2.VideoWriter(
                            str(clip_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps, (width, height)
                        )
                        for cf in clip_buffer:
                            clip_writer.write(cf)
                        clip_writer.release()

                    # 4. Tinh checksum de dam bao tinh toan ven phap ly
                    crop_hash = sha256_of_file(crop_path)
                    full_hash = sha256_of_file(full_path)

                    # 5. Ghi vao violation log
                    log_entry = {
                        "track_id":    track_id,
                        "vehicle_type": cls_name,
                        "confidence":  float(det_conf),
                        "frame":       frame_count,
                        "timestamp":   timestamp_str,
                        "light_state_at_roi1": light_at_cross,   # Trang thai den luc xe qua vach
                        "light_state_current": light_status,      # Trang thai den hien tai (luc o ROI2)
                        "bbox":        [x1, y1, x2, y2],
                        "evidence": {
                            "crop_image":      str(crop_path),
                            "crop_sha256":     crop_hash,
                            "full_image":      str(full_path),
                            "full_sha256":     full_hash,
                            "clip_video":      str(clip_path) if clip_path else None,
                        }
                    }
                    violation_log.append(log_entry)

                # ── Mau sac box xe ──
                if track_id in violators:
                    box_color  = (0, 0, 255)       # Do: ke vi pham
                    box_label  = "VI PHAM"
                elif track_id in vehicles_in_zone1:
                    box_color  = (255, 0, 255)     # Hong: da qua zone1, dang theo doi
                    box_label  = "THEO DOI"
                else:
                    box_color  = (255, 120, 0)     # Cam: xe binh thuong
                    box_label  = ""

                cls_name = vehicle_model.names[cls_id]
                label_text = f"ID:{track_id} {cls_name} {box_label}".strip()
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, label_text, (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2, cv2.LINE_AA)

        # Don dep state cho xe da bien mat khoi frame
        # Tranh track_id cu bi tai su dung cho xe moi ke thua lich su sai
        gone_ids = set(vehicles_in_zone1) - current_track_ids
        for tid in gone_ids:
            vehicles_in_zone1.discard(tid)   # Xoa lich su Zone 1
            zone1_light_state.pop(tid, None) # Xoa trang thai den da ghi
            zone2_frame_count.pop(tid, None) # Giai phong bo nho
            # Don dep last_violation_frame neu xe da bien mat du lau (qua cooldown)
            if frame_count - last_violation_frame.get(tid, -999999) > cooldown_frames * 2:
                last_violation_frame.pop(tid, None)

        # ── Hien thi thong ke tren frame ──
        cv2.putText(frame, f"Tong vi pham: {len(violators)}",
                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Frame: {frame_count}/{total}",
                    (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        out.write(frame)

        if not args.no_view:
            display = cv2.resize(frame, (1280, 720))
            cv2.imshow("Red Light Violation Detector", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nDa nhan Q - Dung som.")
                break

    # ──────────────────────────────────────────────
    # BUOC 5: XUAT KET QUA CUOI CUNG
    # ──────────────────────────────────────────────
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Luu JSON log
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "source_video":    str(source_path),
            "processed_at":    datetime.now().isoformat(),
            "total_frames":    frame_count,
            "total_violations": len(violators),
            "violations":      violation_log,
        }, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 55)
    print(f"  HOAN TAT XU LY")
    print(f"  Tong frames xu ly : {frame_count}")
    print(f"  Tong luot vi pham : {len(violators)}")
    print(f"  Video output      : {args.output}")
    print(f"  Bang chung luu tai: {evidence_dir}/")
    print(f"  JSON log          : {log_path}")
    print("=" * 55)


if __name__ == "__main__":
    main()