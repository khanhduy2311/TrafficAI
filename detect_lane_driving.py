from ultralytics import YOLO
import cv2
import numpy as np
import math

# ================= CONFIG =================
tracker_yaml_path = "/kaggle/working/bytetrack.yaml"

model_sign = YOLO('/kaggle/input/datasets/quannh206/model-yolo/yolov11s.pt')
model_vehicle = YOLO('/kaggle/input/datasets/quannh206/jjjjjj/yolo11m-big-25-2stg.pt')

video_path = '/kaggle/input/datasets/quannh206/556516/video_frame.mp4'
output_path = '/kaggle/working/output.mp4'

# ================= POLYGON =================
lane_left = [(19, 553), (51, 523), (243, 585), (93, 689), (48, 614)]
lane_mid = [(100, 694), (314, 541), (666, 541), (644, 651), (595, 784), (370, 783), (203, 783)]
lane_right = [(606, 785), (669, 546), (813, 522), (792, 635), (699, 748)]

exit_right = [(647, 383), (759, 509), (841, 491), (843, 451), (737, 386)]
exit_left = [(79, 497), (161, 352), (71, 353), (19, 467), (54, 462)]
exit_straight = [(185, 350), (190, 371), (348, 364), (307, 342)]

# ================= HELPER =================
def in_poly(cx, cy, poly):
    return cv2.pointPolygonTest(np.array(poly, np.int32), (cx, cy), False) >= 0

def box_in_poly_ratio(x1, y1, x2, y2, poly, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(poly, np.int32)], 1)

    box_mask = np.zeros((h, w), dtype=np.uint8)
    box_mask[y1:y2, x1:x2] = 1

    overlap = np.logical_and(mask, box_mask).sum()
    box_area = (x2 - x1) * (y2 - y1)

    if box_area == 0:
        return 0
    return overlap / box_area

# ================= VIDEO =================
cap = cv2.VideoCapture(video_path)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(output_path,
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (w, h))

# ================= STATE =================
lane_map = {}
direction_map = {}
valid_ids = set()
violation_ids = set()
violation_log = {}

last_positions = {}
frame_count = 0

# ================= LOOP =================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_frame_ids = set()

    # ===== MODEL =====
    r_vehicle = model_vehicle.track(
        frame,
        persist=True,
        conf=0.1,
        tracker=tracker_yaml_path,
        verbose=False
    )[0]

    r_sign = model_sign(frame, conf=0.1, verbose=False)[0]

    # ================= 🔥 DRAW POLYGON =================
    cv2.polylines(frame, [np.array(lane_left)], True, (255,0,0), 2)
    cv2.polylines(frame, [np.array(lane_mid)], True, (0,255,0), 2)
    cv2.polylines(frame, [np.array(lane_right)], True, (0,0,255), 2)

    cv2.polylines(frame, [np.array(exit_left)], True, (255,0,255), 2)
    cv2.polylines(frame, [np.array(exit_right)], True, (0,255,255), 2)
    cv2.polylines(frame, [np.array(exit_straight)], True, (255,255,0), 2)

    # ================= 🔥 DRAW SIGN =================
    for box in r_sign.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        label = f"{model_sign.names[cls_id]} {conf:.2f}"

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
        cv2.putText(frame, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    # ================= TRACK IDS =================
    if r_vehicle.boxes.id is not None:
        for box in r_vehicle.boxes:
            if box.id is not None:
                current_frame_ids.add(int(box.id[0]))

    for box in r_vehicle.boxes:
        if box.id is None:
            continue

        track_id = int(box.id[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        cls_id = int(box.cls[0])
        class_name = model_vehicle.names[cls_id]

        BIKE_PED_GROUP = {'Bike', 'Pedestrian'}

        # ===== CLASS MERGE (SAME TRACK ID) =====
        if track_id in last_positions:
            old_cls = last_positions[track_id][4]
            if old_cls in BIKE_PED_GROUP and class_name in BIKE_PED_GROUP:
                class_name = old_cls

        # ===== RECONNECT =====
        if track_id not in last_positions:
            for old_id, (old_cx, old_cy, old_frame, old_box, old_cls) in list(last_positions.items()):
                if old_id not in current_frame_ids:

                    dist = math.hypot(cx - old_cx, cy - old_cy)
                    frames_passed = frame_count - old_frame
                    dy = cy - old_cy

                    old_x1, old_y1, old_x2, old_y2 = old_box
                    old_w = old_x2 - old_x1
                    new_w = x2 - x1

                    cls_match = (old_cls == class_name) or (old_cls in BIKE_PED_GROUP and class_name in BIKE_PED_GROUP)

                    if (
                        dist < 50
                        and frames_passed < 45
                        and dy < 0
                        and abs(old_w - new_w) < 40
                        and cls_match
                    ):
                        if old_cls in BIKE_PED_GROUP and class_name in BIKE_PED_GROUP:
                            class_name = old_cls
                            
                        # Transfer all previous tracking states
                        if old_id in valid_ids:
                            valid_ids.add(track_id)
                        if old_id in lane_map:
                            lane_map[track_id] = lane_map[old_id]
                        if old_id in direction_map:
                            direction_map[track_id] = direction_map[old_id]
                        if old_id in violation_ids:
                            violation_ids.add(track_id)
                        if old_id in violation_log:
                            violation_log[track_id] = violation_log[old_id]
                        break

        # ===== FILTER AFTER MERGE =====
        if class_name == "Pedestrian":
            continue

        last_positions[track_id] = (cx, cy, frame_count, (x1, y1, x2, y2), class_name)

        # ===== VALID =====
        if track_id not in valid_ids:
            r_left = box_in_poly_ratio(x1, y1, x2, y2, lane_left, h, w)
            r_mid = box_in_poly_ratio(x1, y1, x2, y2, lane_mid, h, w)
            r_right = box_in_poly_ratio(x1, y1, x2, y2, lane_right, h, w)

            if max(r_left, r_mid, r_right) > 0.5:
                valid_ids.add(track_id)

        if track_id not in valid_ids:
            continue

        # ===== LANE =====
        if track_id not in lane_map:
            ratios = {
                "left": box_in_poly_ratio(x1, y1, x2, y2, lane_left, h, w),
                "straight": box_in_poly_ratio(x1, y1, x2, y2, lane_mid, h, w),
                "right": box_in_poly_ratio(x1, y1, x2, y2, lane_right, h, w)
            }
            best_lane = max(ratios, key=ratios.get)
            if ratios[best_lane] > 0.5:
                lane_map[track_id] = best_lane

        # ===== DIRECTION =====
        if in_poly(cx, cy, exit_straight):
            direction_map[track_id] = "straight"
        elif in_poly(cx, cy, exit_left):
            direction_map[track_id] = "left"
        elif in_poly(cx, cy, exit_right):
            direction_map[track_id] = "right"

        # ===== VIOLATION =====
        violation = False
        if track_id in lane_map and track_id in direction_map:
            if lane_map[track_id] != direction_map[track_id]:
                violation = True

        if violation:
            violation_ids.add(track_id)

            if track_id not in violation_log:
                violation_log[track_id] = round(frame_count / fps, 2)

        # ===== DRAW =====
        color = (0,0,255) if violation else (0,255,0)

        label = f"{class_name} ID:{track_id}"
        if track_id in lane_map:
            label += f" | {lane_map[track_id]}"
        if track_id in direction_map:
            label += f" -> {direction_map[track_id]}"
        if violation:
            label += " VIOLATION"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # ===== INFO =====
    cv2.putText(frame,
                f"Violations: {len(violation_ids)}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                3)

    cv2.putText(frame,
                f"Frame: {frame_count}",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,255,255),
                2)

    out.write(frame)

# ================= DONE =================
cap.release()
out.release()

print("DONE:", output_path)

print("\n=== VIOLATION LOG ===")
for vid, t in violation_log.items():
    print(f"Vehicle ID {vid} violated at {t} sec")