"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from PIL import Image


def resize_image_and_update_annotations(image_path, annotations, output_path, max_size=640):
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            if max(w, h) <= max_size:
                img.save(output_path)  # still copy to new dir
                return annotations, w, h, True  # copied but not resized

            scale = max_size / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            img.save(output_path)

            for ann in annotations:
                ann["area"] = ann["area"] * (scale ** 2)
                ann["bbox"] = [coord * scale for coord in ann["bbox"]]
                if "orig_size" in ann:
                    ann["orig_size"] = (new_w, new_h)
                if "size" in ann:
                    ann["size"] = (new_w, new_h)

            return annotations, new_w, new_h, True

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def resize_images_and_update_annotations(data_dir, json_name, max_size=640, num_workers=4):
    image_dir = os.path.join(data_dir, "images")
    output_image_dir = os.path.join(data_dir, "images_resized")
    os.makedirs(output_image_dir, exist_ok=True)

    json_path = os.path.join(data_dir, json_name)
    if not os.path.isfile(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return

    print(f"Loading annotations from: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)

    image_annotations = {img["id"]: [] for img in data["images"]}
    for ann in data["annotations"]:
        image_annotations[ann["image_id"]].append(ann)

    def process_image(image_info):
        if "file_name" not in image_info:
            print(f"Missing 'file_name' in image info: {image_info}")
            return None
        input_path = os.path.join(image_dir, image_info["file_name"])
        output_path = os.path.join(output_image_dir, image_info["file_name"])
        results = resize_image_and_update_annotations(
            input_path, image_annotations[image_info["id"]], output_path, max_size
        )
        if results is None:
            return None
        updated_annotations, new_w, new_h, _ = results
        return image_info, updated_annotations, new_w, new_h

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_image, data["images"]))

    new_images = []
    new_annotations = []

    for res in results:
        if res is None:
            continue
        image_info, updated_annotations, new_w, new_h = res
        image_info["width"] = new_w
        image_info["height"] = new_h
        new_images.append(image_info)
        new_annotations.extend(updated_annotations)

    new_data = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": data["categories"],
    }

    new_json_path = os.path.join(data_dir, json_name.replace(".json", "_resized.json"))
    with open(new_json_path, "w") as f:
        json.dump(new_data, f)
    print(f"Saved resized annotations to: {new_json_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Resize images and update dataset annotations for train/val set."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing images/, train.json or val.json",
    )
    parser.add_argument(
        "--json_name",
        type=str,
        required=True,
        help="Name of the annotation file (e.g., train.json or val.json)",
    )
    parser.add_argument(
        "--max_size",
        type=int,
        default=640,
        help="Maximum size for the longer side of the image (default: 640)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel threads (default: 4)",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    resize_images_and_update_annotations(
        data_dir=args.data_dir,
        json_name=args.json_name,
        max_size=args.max_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
