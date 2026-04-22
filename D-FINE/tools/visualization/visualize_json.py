import os
import json
import argparse
import cv2
import random
from tqdm import tqdm

def visualize_annotations_cv2(json_path, images_dir, output_dir, num_images_to_visualize,
                              class_thresholds_str, default_threshold_val, specific_image_filename=None):
    """
    Load COCO-format JSON annotations, filter by class-specific confidence thresholds,
    and save annotated images to output_dir using OpenCV.
    All classes use the same color (Red) for bounding boxes and labels.
    """
    # Use only Red color (BGR format)
    box_color = (0, 0, 255)  # Red

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file JSON: {json_path}")
        return
    except json.JSONDecodeError:
        print(f"Lỗi: File JSON không hợp lệ: {json_path}")
        return

    images_data = data.get('images', [])
    annotations_data = data.get('annotations', [])
    
    # Parse class-specific thresholds
    class_thresholds_by_name = {}
    try:
        parsed_thresholds = json.loads(class_thresholds_str)
        if isinstance(parsed_thresholds, dict):
            for k, v in parsed_thresholds.items():
                if not isinstance(k, str) or not isinstance(v, (int, float)):
                    raise ValueError("Keys must be strings and values must be numbers.")
                if not (0.0 <= float(v) <= 1.0):
                    raise ValueError(f"Threshold for class '{k}' must be in range 0.0 to 1.0.")
                class_thresholds_by_name[k.lower()] = float(v)
            if class_thresholds_by_name:
                print(f"Using class-specific thresholds: {class_thresholds_by_name}")
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON for class_thresholds. Using default.")
    except ValueError as e:
        print(f"Error in class_thresholds: {e}. Using default.")

    print(f"Default confidence threshold: {default_threshold_val}")

    # Load category name map
    categories_from_json_raw = {c['id']: c['name'] for c in data.get('categories', [])}
    categories_from_json = {k: v.lower() for k, v in categories_from_json_raw.items()}

    selected_images_info = []
    if specific_image_filename:
        found_specific_image = False
        target_basename = os.path.basename(specific_image_filename)
        for img_info_item in images_data:
            if os.path.basename(img_info_item['file_name']) == target_basename:
                selected_images_info = [img_info_item]
                found_specific_image = True
                print(f"Selected specific image: {img_info_item['file_name']}")
                break
        if not found_specific_image:
            print(f"Image '{specific_image_filename}' not found.")
            return
    else:
        if not images_data:
            print("No images found in JSON.")
            return
        if num_images_to_visualize <= 0:
            selected_images_info = images_data
        elif len(images_data) > num_images_to_visualize:
            selected_images_info = random.sample(images_data, num_images_to_visualize)
        else:
            selected_images_info = images_data

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    annotations_by_image_id = {}
    for ann in annotations_data:
        img_id = ann['image_id']
        annotations_by_image_id.setdefault(img_id, []).append(ann)

    processed_image_count = 0
    skipped_img_count = 0
    for img_info in tqdm(selected_images_info, desc="Visualizing"):
        img_path = os.path.join(images_dir, img_info['file_name'])

        if not os.path.isfile(img_path):
            skipped_img_count += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            skipped_img_count += 1
            continue
        
        anns = annotations_by_image_id.get(img_info['id'], [])
        for ann in anns:
            try:
                x, y, w, h = map(int, ann['bbox'])
            except (ValueError, TypeError):
                continue

            category_id = ann['category_id']
            score = ann.get('score', None)
            class_name_display = categories_from_json_raw.get(category_id, f'ID:{category_id}')
            class_name_lookup = categories_from_json.get(category_id, f'id:{category_id}')
            threshold = class_thresholds_by_name.get(class_name_lookup, default_threshold_val)

            if score is not None:
                if score < threshold:
                    continue
            elif threshold > 0.0:
                continue

            cv2.rectangle(img, (x, y), (x + w, y + h), box_color, 1)
            label = f"{class_name_display}"
            if score is not None:
                label += f" {score:.2f}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            ((text_width, text_height), _) = cv2.getTextSize(label, font, font_scale, thickness)

            cv2.rectangle(img, (x, y - text_height - 4), (x + text_width, y), box_color, -1)
            cv2.putText(img, label, (x, y - 2), font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

        out_path = os.path.join(output_dir, os.path.basename(img_info['file_name']))
        try:
            cv2.imwrite(out_path, img)
            processed_image_count += 1
        except Exception as e:
            print(f"Error saving image {out_path}: {e}")
            skipped_img_count += 1

    print(f"\nSkipped images: {skipped_img_count}")
    print(f"Processed images saved: {processed_image_count}")

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize COCO annotations using one color (Red).')
    parser.add_argument('-j', '--json', required=True, help='Path to COCO JSON file')
    parser.add_argument('-i', '--images_dir', required=True, help='Directory containing images')
    parser.add_argument('-o', '--output_dir', required=True, help='Directory to save annotated images')
    parser.add_argument('-n', '--num_images', type=int, default=10, help='Number of images to visualize (default: 10)')
    parser.add_argument('--specific_image_filename', type=str, default=None, help='Specific image filename to visualize')
    parser.add_argument('--class_thresholds', type=str, default='{}', help='JSON string of class-specific confidence thresholds')
    parser.add_argument('--default_threshold', type=float, default=0.0, help='Default confidence threshold (0.0–1.0)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if not (0.0 <= args.default_threshold <= 1.0):
        print("Error: default_threshold must be between 0.0 and 1.0.")
    else:
        visualize_annotations_cv2(
            args.json,
            args.images_dir,
            args.output_dir,
            args.num_images,
            args.class_thresholds,
            args.default_threshold,
            args.specific_image_filename
        )
