import os
import json
import argparse
import cv2
from tqdm import tqdm
import concurrent.futures

# --- Cấu hình cố định ---
COCO_CATEGORIES = [
    {"id": 0, "name": "Bus"},
    {"id": 1, "name": "Bike"},
    {"id": 2, "name": "Car"},
    {"id": 3, "name": "Pedestrian"},
    {"id": 4, "name": "Truck"}
]

CLASS_NAME_TO_COCO_ID = {cat['name'].lower(): cat['id'] for cat in COCO_CATEGORIES}

DEFINED_YOLO_CLASS_NAMES = {
    0: 'Bus',
    1: 'Bike',
    2: 'Car',
    3: 'Pedestrian',
    4: 'Truck'
}

def process_image(index, filename, images_dir, labels_dir, include_confidence):
    image_path = os.path.join(images_dir, filename)
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None, []
        height, width, _ = image.shape
    except:
        return None, []

    image_entry = {
        "file_name": filename,
        "height": height,
        "width": width,
        "id": index
    }

    annotations = []
    label_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + ".txt")
    if os.path.exists(label_path):
        with open(label_path, 'r') as f_label:
            for line in f_label:
                parts = line.strip().split()
                if len(parts) == 6:
                    yolo_class_id, x_center, y_center, w, h, score = map(float, parts)
                elif len(parts) == 5:
                    yolo_class_id, x_center, y_center, w, h = map(float, parts)
                    score = None
                else:
                    continue

                yolo_class_id = int(yolo_class_id)
                class_name = DEFINED_YOLO_CLASS_NAMES.get(yolo_class_id)
                if class_name is None:
                    continue
                coco_category_id = CLASS_NAME_TO_COCO_ID.get(class_name.lower())
                if coco_category_id is None:
                    continue

                abs_w = w * width
                abs_h = h * height
                x_min = x_center * width - abs_w / 2
                y_min = y_center * height - abs_h / 2

                annotation = {
                    "image_id": index,
                    "category_id": coco_category_id,
                    "bbox": [round(x_min, 2), round(y_min, 2), round(abs_w, 2), round(abs_h, 2)],
                    "area": round(abs_w * abs_h, 2),
                    "iscrowd": 0
                }
                if include_confidence and score is not None:
                    annotation["score"] = round(score, 3)
                
                annotations.append(annotation)

    return image_entry, annotations

def yolo_to_coco(images_dir, labels_dir, output_json_path, include_confidence, num_workers=4):
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    image_filenames = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(supported_extensions)])

    if not image_filenames:
        print(f"Không tìm thấy ảnh nào trong: {images_dir}")
        return

    coco_output = {
        "info": {},
        "licenses": [],
        "categories": COCO_CATEGORIES,
        "images": [],
        "annotations": []
    }

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_image, idx, fname, images_dir, labels_dir, include_confidence)
                   for idx, fname in enumerate(image_filenames)]

        annotation_id = 1
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Xử lý"):
            image_entry, annotations = future.result()
            if image_entry:
                coco_output["images"].append(image_entry)
                for ann in annotations:
                    ann['id'] = annotation_id
                    coco_output["annotations"].append(ann)
                    annotation_id += 1

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_output, f, indent=4)

    print(f"\nHoàn tất! Đã tạo file COCO JSON tại: {output_json_path}")
    print(f"Tổng số ảnh: {len(coco_output['images'])}")
    print(f"Tổng số annotations: {len(coco_output['annotations'])}")

def parse_args():
    parser = argparse.ArgumentParser(description="Chuyển đổi bộ dữ liệu YOLO sang COCO JSON.")
    parser.add_argument('--images_dir', required=True)
    parser.add_argument('--labels_dir', required=True)
    parser.add_argument('--output_json', required=True)
    parser.add_argument('--conf', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4, help="Số luồng xử lý song song")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    yolo_to_coco(args.images_dir, args.labels_dir, args.output_json, args.conf, args.num_workers)