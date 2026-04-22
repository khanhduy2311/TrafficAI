import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import timedelta
from collections import deque
from ultralytics import YOLO

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
CFG = {
    # I/O
    'video_path':   '/kaggle/input/datasets/quannh206/hihihi/video_chaynguocchieu.mp4',
    'output_video': '/kaggle/working/output_annotated.mp4',
    'output_csv':   '/kaggle/working/violations.csv',

    # Model
    'model_weights':       '/kaggle/input/datasets/quannh206/model-yl/yolo11m-big-25-2stg.pt',
    'conf_threshold':      0.08,
    'iou_threshold':       0.45,
    'vehicle_class_names': ['Bus', 'Bike', 'Car', 'Pedestrian', 'Truck'],
    'exclude_wrongway':    [],

    # Fisheye
    'apply_undistort': False,
    'K_diag':          [600, 600],
    'dist_coeffs':     [-0.3, 0.1, 0.0, 0.0],

    # ROI — polygon pixel tuyệt đối (bạn tự vẽ, đã cập nhật)
    'roi_polygon_px': [
        (700,  331),
        (1080, 1100),
        (1116, 1100),
        (1260, 789),
        (1278, 500),
        (1150, 405),
        (1000,  300),
        (809,  300),
        (755,  327),
    ],

    # Scene flow
    'scene_flow_window':   20,

    # Track
    'track_history_len':   40,
    'min_frames_to_judge': 6,

    # Thresholds mặc định (Car / Bus / Truck)
    'relative_dy_thresh':  0.012,
    'wrong_way_dy_thresh': 0.004,
    'min_displacement_px': 8,
    'min_speed_dy_norm':   0.002,
    'confirm_frames':      5,
    'max_bbox_area_pct':   0.60,

    'disable_ego_lane_check':    True,
    'overtake_lateral_thresh':   0.05,
    'overtake_same_lane_x_band': 0.30,

    # FIX v4: Hạ threshold cho Bike/Pedestrian + class merge
    'class_overrides': {
        'Bike': {
            'relative_dy_thresh':  0.005,   # hạ từ 0.008 → 0.005
            'wrong_way_dy_thresh': 0.003,
            'min_displacement_px': 4,
            'min_speed_dy_norm':   0.001,
            'confirm_frames':      4,       # hạ từ 5 → 4
            'min_frames_to_judge': 4,       # hạ từ 5 → 4
        },
        'Pedestrian': {
            'relative_dy_thresh':  0.005,   # hạ từ 0.008 → 0.005
            'wrong_way_dy_thresh': 0.003,
            'min_displacement_px': 4,
            'min_speed_dy_norm':   0.001,
            'confirm_frames':      4,
            'min_frames_to_judge': 4,
        },
    },

    # FIX v4: Class merge — Bike & Pedestrian hay bị nhận nhầm nhau
    # Nếu track đang là Bike bị detect thành Pedestrian (hoặc ngược lại)
    # → KHÔNG tạo TrackState mới, giữ nguyên history
    'merge_bike_pedestrian': True,

    'show_tracks': True,
    'debug_mode':  True,
}

# ══════════════════════════════════════════════════════════════════════════════
# ROI
# ══════════════════════════════════════════════════════════════════════════════
def build_roi_mask_px(h, w, polygon_px):
    pts  = np.array(polygon_px, dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask, pts

def in_roi(cx, cy, mask):
    x, y = int(cx), int(cy)
    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
        return mask[y, x] > 0
    return False

# ══════════════════════════════════════════════════════════════════════════════
# SCENE FLOW ESTIMATOR
# ══════════════════════════════════════════════════════════════════════════════
class SceneFlowEstimator:
    def __init__(self, window=20):
        self.dy_buffer = deque(maxlen=window)

    def update(self, frame_dy_list):
        positive = [dy for dy in frame_dy_list if dy > 0.001]
        if positive:
            self.dy_buffer.append(float(np.median(positive)))

    def get_scene_flow(self):
        if len(self.dy_buffer) < 5:
            return None
        return float(np.mean(self.dy_buffer))

    def is_wrong_way(self, track_dy, relative_thresh, fallback_thresh):
        sf = self.get_scene_flow()
        if sf is None:
            return track_dy < -fallback_thresh, None, None
        rel = track_dy - sf
        return rel < -relative_thresh, sf, rel

# ══════════════════════════════════════════════════════════════════════════════
# FISHEYE
# ══════════════════════════════════════════════════════════════════════════════
class FisheyeUndistorter:
    def __init__(self, frame_w, frame_h, cfg):
        fx, fy = cfg['K_diag']
        cx, cy = frame_w / 2, frame_h / 2
        K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
        D = np.array(cfg['dist_coeffs'], dtype=np.float64)
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), K, (frame_w, frame_h), cv2.CV_16SC2)

    def undistort(self, frame):
        return cv2.remap(frame, self.map1, self.map2,
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

# ══════════════════════════════════════════════════════════════════════════════
# TRACK STATE
# ══════════════════════════════════════════════════════════════════════════════
class TrackState:
    def __init__(self, track_id, base_cfg, class_name, H, W):
        self.id         = track_id
        self.H, self.W  = H, W
        self.class_name = class_name
        self.cfg        = {**base_cfg, **base_cfg.get('class_overrides', {}).get(class_name, {})}
        self.history    = deque(maxlen=self.cfg['track_history_len'])

        self.consec_wrong     = 0
        self.confirmed        = False
        self.viol_frame       = None
        self.viol_time        = None
        self.color            = tuple(np.random.randint(50, 255, 3).tolist())
        self.last_relative_dy = None
        self.last_scene_flow  = None

    def update(self, frame_idx, cx, cy):
        self.history.append((frame_idx, cx, cy))

    def _avg_dy(self):
        if len(self.history) < self.cfg['min_frames_to_judge']:
            return 0.0
        pts = list(self.history)
        dy_list = [(pts[i][2] - pts[i-1][2]) / self.H for i in range(1, len(pts))]
        return float(np.mean(dy_list))

    def _displacement_toward_camera(self):
        if len(self.history) < 2:
            return 0.0
        return self.history[-1][2] - self.history[0][2]

    def evaluate(self, frame_idx, fps, scene_flow_est: SceneFlowEstimator):
        c    = self.cfg
        dy   = self._avg_dy()
        disp = self._displacement_toward_camera()

        is_wrong_rel, sf, rel = scene_flow_est.is_wrong_way(
            dy,
            c.get('relative_dy_thresh', 0.012),
            c.get('wrong_way_dy_thresh', 0.004),
        )
        self.last_scene_flow  = sf
        self.last_relative_dy = rel

        wrong = is_wrong_rel

        if abs(disp) < c['min_displacement_px']:
            wrong = False
        if abs(dy) < c['min_speed_dy_norm']:
            wrong = False

        self.consec_wrong = (self.consec_wrong + 1) if wrong else max(0, self.consec_wrong - 1)

        if not self.confirmed and self.consec_wrong >= c['confirm_frames']:
            self.confirmed  = True
            self.viol_frame = frame_idx
            self.viol_time  = str(timedelta(seconds=frame_idx / fps))

        return self.confirmed

    def debug_info(self, sf):
        dy    = self._avg_dy()
        disp  = self._displacement_toward_camera()
        sf_s  = f'{sf:.4f}' if sf is not None else 'N/A'
        rel_s = f'{self.last_relative_dy:+.4f}' if self.last_relative_dy is not None else 'N/A'
        return (f"dy={dy:+.4f} sf={sf_s} rel={rel_s} "
                f"disp={disp:+.1f}px consec={self.consec_wrong}")

# ══════════════════════════════════════════════════════════════════════════════
# FIX v4: CLASS MERGE HELPER
# ══════════════════════════════════════════════════════════════════════════════
BIKE_PED_GROUP = {'Bike', 'Pedestrian'}

def get_or_create_track(track_states, tid, base_cfg, cls_name, H, W, do_merge):
    """
    Trả về TrackState cho tid.
    Nếu track đã tồn tại:
      - Bike ↔ Pedestrian: GIỮ NGUYÊN state (merge), chỉ cập nhật class_name hiển thị
      - Các class khác đổi nhau: giữ nguyên (tracker đã assign cùng ID)
    """
    if tid not in track_states:
        track_states[tid] = TrackState(tid, base_cfg, cls_name, H, W)
    else:
        existing = track_states[tid]
        if do_merge and existing.class_name in BIKE_PED_GROUP and cls_name in BIKE_PED_GROUP:
            # Bike ↔ Pedestrian: không reset, chỉ cập nhật tên để hiển thị đúng
            existing.class_name = cls_name
        # Các trường hợp khác: giữ nguyên track, tracker đã lo ID
    return track_states[tid]

# ══════════════════════════════════════════════════════════════════════════════
# DRAW HELPERS
# ══════════════════════════════════════════════════════════════════════════════
FONT = cv2.FONT_HERSHEY_DUPLEX
MONO = cv2.FONT_HERSHEY_PLAIN

def draw_banner(frame, viol_ids):
    if not viol_ids:
        return
    H, W = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (W,48), (0,0,180), -1)
    cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)
    txt = 'WRONG-WAY  ID: ' + '  '.join(str(v) for v in sorted(viol_ids))
    cv2.putText(frame, txt, (12,33), FONT, 0.8, (255,255,255), 2, cv2.LINE_AA)

def draw_counter(frame, fi, total):
    H, W = frame.shape[:2]
    txt = f'Frame {fi:05d}/{total:05d}'
    (tw,th),_ = cv2.getTextSize(txt, MONO, 1.3, 1)
    cv2.rectangle(frame, (W-tw-16,H-th-20), (W-4,H-4), (0,0,0), -1)
    cv2.putText(frame, txt, (W-tw-10,H-10), MONO, 1.3, (200,255,200), 1, cv2.LINE_AA)

def draw_scene_flow_hud(frame, sf):
    H = frame.shape[0]
    txt = f'SceneFlow: {sf:.4f}' if sf is not None else 'SceneFlow: warming up...'
    cv2.putText(frame, txt, (10, H-12), MONO, 1.2, (0,255,255), 1, cv2.LINE_AA)

def draw_track(frame, state, is_viol):
    color = (0,0,255) if is_viol else state.color
    pts   = [(int(p[1]), int(p[2])) for p in state.history]
    for i in range(1, len(pts)):
        cv2.line(frame, pts[i-1], pts[i], color, 2, cv2.LINE_AA)
    if pts:
        cv2.circle(frame, pts[-1], 5, color, -1)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def run_detection(cfg=CFG):
    model      = YOLO(cfg['model_weights'])
    name_to_id = {v:k for k,v in model.names.items()}
    print('Model classes:', model.names)

    vehicle_ids = [name_to_id[n] for n in cfg['vehicle_class_names'] if n in name_to_id]
    exclude_ids = {name_to_id[n] for n in cfg['exclude_wrongway']    if n in name_to_id}
    if not vehicle_ids:
        raise ValueError('Không tìm thấy class nào hợp lệ.')

    do_merge = cfg.get('merge_bike_pedestrian', True)

    print('\n--- Config per class ---')
    for cls in cfg['vehicle_class_names']:
        m = {**cfg, **cfg.get('class_overrides',{}).get(cls,{})}
        print(f"  {cls:12s}: rel_dy>{m.get('relative_dy_thresh',0.012):.3f}  "
              f"confirm={m['confirm_frames']}f  "
              f"judge={m['min_frames_to_judge']}f")
    print(f"  merge Bike↔Pedestrian: {'ON' if do_merge else 'OFF'}")
    print()

    cap = cv2.VideoCapture(cfg['video_path'])
    assert cap.isOpened(), f"Không mở được video: {cfg['video_path']}"

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0
    TOT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    FA  = W * H
    print(f'Video: {W}x{H}  FPS={FPS:.1f}  Frames={TOT}')

    for px, py in cfg['roi_polygon_px']:
        if not (0 <= px <= W and 0 <= py <= H):
            print(f'  ⚠️  Điểm ({px},{py}) nằm ngoài frame {W}x{H} — clip lại')
    poly_clipped = [(min(max(px,0),W-1), min(max(py,0),H-1))
                    for px,py in cfg['roi_polygon_px']]

    undist            = FisheyeUndistorter(W, H, cfg) if cfg['apply_undistort'] else None
    roi_mask, roi_pts = build_roi_mask_px(H, W, poly_clipped)
    scene_flow_est    = SceneFlowEstimator(window=cfg.get('scene_flow_window', 20))

    writer = cv2.VideoWriter(cfg['output_video'],
                             cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W,H))

    track_states   = {}
    all_viol_ids   = set()
    all_violations = []
    fi = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fi += 1
        if undist:
            frame = undist.undistort(frame)

        results = model.track(
            frame, persist=True,
            conf=cfg['conf_threshold'], iou=cfg['iou_threshold'],
            classes=vehicle_ids, tracker='bytetrack.yaml', verbose=False,
        )

        # Vẽ polygon ROI
        cv2.polylines(frame, [roi_pts], isClosed=True, color=(0,255,255), thickness=2, lineType=cv2.LINE_AA)
        ov = frame.copy()
        cv2.fillPoly(ov, [roi_pts], (0,255,255))
        cv2.addWeighted(ov, 0.06, frame, 0.94, 0, frame)

        frame_viol_ids = set()

        if results[0].boxes.id is None:
            draw_banner(frame, frame_viol_ids)
            draw_counter(frame, fi, TOT)
            if cfg.get('debug_mode'):
                draw_scene_flow_hud(frame, scene_flow_est.get_scene_flow())
            writer.write(frame)
            continue

        boxes_all   = results[0].boxes.xyxy.cpu().numpy()
        ids_all     = results[0].boxes.id.cpu().numpy().astype(int)
        classes_all = results[0].boxes.cls.cpu().numpy().astype(int)
        confs_all   = results[0].boxes.conf.cpu().numpy()

        # ── Pass 1: update history + scene flow ──
        frame_dy_list = []
        for box, tid, cls, conf in zip(boxes_all, ids_all, classes_all, confs_all):
            x1,y1,x2,y2 = box
            cx,cy = (x1+x2)/2, (y1+y2)/2

            if not in_roi(cx, cy, roi_mask):
                continue
            if (x2-x1)*(y2-y1)/FA > cfg['max_bbox_area_pct']:
                continue

            cls_name = model.names[cls]

            # FIX v4: dùng get_or_create_track thay vì tạo thẳng
            state = get_or_create_track(track_states, tid, cfg, cls_name, H, W, do_merge)
            state.update(fi, cx, cy)

            dy = state._avg_dy()
            if dy != 0.0:
                frame_dy_list.append(dy)

        scene_flow_est.update(frame_dy_list)
        current_sf = scene_flow_est.get_scene_flow()

        # ── Pass 2: evaluate + vẽ ──
        for box, tid, cls, conf in zip(boxes_all, ids_all, classes_all, confs_all):
            x1,y1,x2,y2 = box
            cx,cy = (x1+x2)/2, (y1+y2)/2

            if not in_roi(cx, cy, roi_mask):
                continue
            if (x2-x1)*(y2-y1)/FA > cfg['max_bbox_area_pct']:
                continue

            if tid not in track_states:
                continue

            cls_name = model.names[cls]
            state    = track_states[tid]
            is_viol  = (cls not in exclude_ids) and state.evaluate(fi, FPS, scene_flow_est)

            if cfg.get('debug_mode') and fi % 30 == 0 and len(state.history) >= state.cfg['min_frames_to_judge']:
                print(f"  [DBG] ID:{tid:3d} {cls_name:10s} {state.debug_info(current_sf)}"
                      f"{' ← VIOLATION' if is_viol else ''}")

            if is_viol:
                frame_viol_ids.add(tid)
                all_viol_ids.add(tid)
                if not any(r['id'] == tid for r in all_violations):
                    all_violations.append({
                        'id':    tid,
                        'class': state.class_name,   # dùng class_name của state (đã merge)
                        'frame': state.viol_frame,
                        'time':  state.viol_time,
                    })

            color = (0,0,255) if is_viol else (0,220,0)
            cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), color, 2, cv2.LINE_AA)

            if cfg.get('debug_mode') and state.last_relative_dy is not None:
                lbl = f'ID:{tid} {state.class_name} rel={state.last_relative_dy:+.3f}'
            else:
                lbl = f'ID:{tid} {state.class_name}'
            if is_viol:
                lbl += ' !!!'
            cv2.putText(frame, lbl, (int(x1), int(y1)-8),
                        FONT, 0.42, color, 1, cv2.LINE_AA)

            if cfg['show_tracks']:
                draw_track(frame, state, is_viol)

        draw_banner(frame, frame_viol_ids)
        draw_counter(frame, fi, TOT)
        if cfg.get('debug_mode'):
            draw_scene_flow_hud(frame, current_sf)

        writer.write(frame)

        if fi % 50 == 0:
            elapsed = time.time() - t0
            sf_s    = f'{current_sf:.4f}' if current_sf is not None else 'warming'
            print(f'  [{fi/TOT*100:5.1f}%] {fi}/{TOT}  '
                  f'scene_flow={sf_s}  '
                  f'vi_pham={len(all_viol_ids)}  '
                  f'ids={sorted(all_viol_ids)}  '
                  f't={elapsed:.1f}s')

    cap.release()
    writer.release()

    print(f'\n{"═"*40}')
    print(f'Tổng vi phạm : {len(all_viol_ids)} xe')
    print(f'ID vi phạm   : {sorted(all_viol_ids)}')
    print(f'{"═"*40}')

    df = pd.DataFrame(all_violations)
    if len(df):
        print('\nChi tiết:')
        print(df[['id','class','time']].to_string(index=False))
    df.to_csv(cfg['output_csv'], index=False)
    print(f'\nSaved -> {cfg["output_csv"]}')
    return df


violations_df = run_detection()

# ══════════════════════════════════════════════════════════════════════════════
# THUMBNAILS
# ══════════════════════════════════════════════════════════════════════════════
if len(violations_df):
    cap2 = cv2.VideoCapture(CFG['video_path'])
    n    = min(len(violations_df), 4)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, violations_df.iterrows()):
        cap2.set(cv2.CAP_PROP_POS_FRAMES, row['frame'])
        ret, thumb = cap2.read()
        if ret:
            ax.imshow(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))
            ax.set_title(f"ID:{row['id']}  {row['time']}", fontsize=9)
        ax.axis('off')
    cap2.release()
    plt.tight_layout()
    plt.savefig('/kaggle/working/violation_thumbnails.png', dpi=120)
    plt.show()