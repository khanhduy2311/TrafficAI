import os
import cv2
import numpy as np

# =========================
# CONFIG 
# =========================
IMG_PATH = "/path/to/your/image.jpg"  # ⚠️ SỬA ĐƯỜNG DẪN ẢNH
LABEL_PATH = "/path/to/your/labels.txt"  # ⚠️ SỬA ĐƯỜNG DẪN FILE NHẬN DIỆN

OUT_IMG_DIR = "/path/to/output/images"
OUT_LABEL_DIR = "/path/to/output/labels"

FOV_DEGREE = 120.0  

# ⚠️ SỬA CLASS NAMES THEO DATASET
CLASS_NAMES = [
    "Bus", "Bike", "Car", "Pedestrian", "Truck"
]

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LABEL_DIR, exist_ok=True)

omega = np.radians(FOV_DEGREE / 2.0)
tan_omega = np.tan(omega)

# =========================
# COLOR
# =========================
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 0), (0, 128, 0), (0, 0, 128),
    (255, 128, 0), (128, 0, 255)
]

def get_color(cls):
    return COLORS[cls % len(COLORS)]

def get_class_name(cls):
    if cls < len(CLASS_NAMES):
        return CLASS_NAMES[cls]
    return str(cls)

# =========================
# 1. FISHEYE
# =========================
def apply_fisheye_physical(img):
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    R = min(cx, cy)

    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))

    dx = map_x - cx
    dy = map_y - cy
    r_dst = np.sqrt(dx**2 + dy**2)

    r_dst_norm = np.clip(r_dst / R, 0, 1)
    r_src_norm = np.tan(r_dst_norm * omega) / tan_omega
    r_src = r_src_norm * R

    r_dst_safe = np.maximum(r_dst, 1e-5)
    scale = r_src / r_dst_safe

    src_x = cx + dx * scale
    src_y = cy + dy * scale

    new_img = cv2.remap(
        img,
        src_x.astype(np.float32),
        src_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), int(R), 255, -1)
    new_img[mask == 0] = 0

    return new_img

# =========================
# 2. BBOX
# =========================
def yolo_to_xyxy(cls, cx, cy, bw, bh, w, h):
    cx *= w
    cy *= h
    bw *= w
    bh *= h

    x1 = int(cx - bw / 2)
    y1 = int(cy - bh / 2)
    x2 = int(cx + bw / 2)
    y2 = int(cy + bh / 2)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w - 1, x2)
    y2 = min(h - 1, y2)

    return cls, [x1, y1, x2, y2]

def xyxy_to_yolo(cls, bbox, w, h):
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2.0) / w
    cy = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / float(w)
    bh = (y2 - y1) / float(h)
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

def convert_bbox(bbox, w, h, grid_size=30):
    x1, y1, x2, y2 = bbox

    xs = np.linspace(x1, x2, grid_size)
    ys = np.linspace(y1, y2, grid_size)
    grid_x, grid_y = np.meshgrid(xs, ys)

    pts_x = grid_x.ravel()
    pts_y = grid_y.ravel()

    cx, cy = w / 2.0, h / 2.0
    R = min(cx, cy)

    dx = pts_x - cx
    dy = pts_y - cy
    r_src = np.sqrt(dx**2 + dy**2)
    r_src_safe = np.maximum(r_src, 1e-5)

    r_src_norm = r_src_safe / R
    r_dst_norm = np.arctan(r_src_norm * tan_omega) / omega
    r_dst = r_dst_norm * R

    valid_mask = r_dst <= R
    if not np.any(valid_mask):
        return None

    valid_dx = dx[valid_mask]
    valid_dy = dy[valid_mask]
    valid_r_src = r_src_safe[valid_mask]
    valid_r_dst = r_dst[valid_mask]

    scale = valid_r_dst / valid_r_src
    dst_x = cx + valid_dx * scale
    dst_y = cy + valid_dy * scale

    new_x1 = np.clip(np.min(dst_x), 0, w - 1)
    new_y1 = np.clip(np.min(dst_y), 0, h - 1)
    new_x2 = np.clip(np.max(dst_x), 0, w - 1)
    new_y2 = np.clip(np.max(dst_y), 0, h - 1)

    if (new_x2 - new_x1) < 4 or (new_y2 - new_y1) < 4:
        return None

    return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]

# =========================
# 3. VISUALIZE
# =========================
def draw_box(img, cls, box):
    x1, y1, x2, y2 = box
    color = get_color(cls)
    name = get_class_name(cls)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    label = name
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
    cv2.putText(img, label, (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

def visualize_original(img, labels):
    vis_img = img.copy()
    for cls, box in labels:
        draw_box(vis_img, cls, box)
    return vis_img

# =========================
# MAIN
# =========================
def run(img_path, label_path):
    print("⏳ Processing...")

    img = cv2.imread(img_path)
    if img is None:
        print("❌ Cannot read image")
        return

    h, w = img.shape[:2]

    # ===== READ YOLO =====
    labels = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls = int(parts[0])
                    cx = float(parts[1])
                    cy = float(parts[2])
                    bw = float(parts[3])
                    bh = float(parts[4])

                    if bw <= 0 or bh <= 0:
                        continue

                    cls, box = yolo_to_xyxy(cls, cx, cy, bw, bh, w, h)
                    labels.append((cls, box))

    # ===== ORIGINAL =====
    orig_vis = visualize_original(img, labels)

    # ===== FISHEYE =====
    new_img = apply_fisheye_physical(img)
    debug_img = new_img.copy()

    # ===== PROCESS BBOX =====
    new_labels = []
    valid_boxes = 0
    dropped_boxes = 0

    for cls, box in labels:
        new_box = convert_bbox(box, w, h, grid_size=30)

        if new_box is not None:
            new_labels.append(xyxy_to_yolo(cls, new_box, w, h))
            draw_box(debug_img, cls, new_box)
            valid_boxes += 1
        else:
            dropped_boxes += 1

    # ===== SAVE =====
    name = os.path.basename(img_path)

    cv2.imwrite(os.path.join(OUT_IMG_DIR, name), new_img)
    cv2.imwrite(os.path.join(OUT_IMG_DIR, "debug_" + name), debug_img)
    cv2.imwrite(os.path.join(OUT_IMG_DIR, "original_" + name), orig_vis)

    with open(os.path.join(OUT_LABEL_DIR, name.replace(".jpg", ".txt")), "w") as f:
        for l in new_labels:
            f.write(l + "\n")

    print("==================================")
    print("✅ DONE!")
    print(f"📦 Keep: {valid_boxes}")
    print(f"🗑 Drop: {dropped_boxes}")
    print("==================================")


if __name__ == "__main__":
    run(IMG_PATH, LABEL_PATH)