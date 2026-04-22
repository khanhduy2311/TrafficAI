import os
import json
import argparse

def remove_coco_images(images_dir):
    removed_filenames = []
    for fname in os.listdir(images_dir):
        if fname.startswith("coco"):
            file_path = os.path.join(images_dir, fname)
            if os.path.isfile(file_path):
                os.remove(file_path)
                removed_filenames.append(fname)
    return removed_filenames

def update_coco_json(json_path, removed_filenames):
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Find image IDs of images to remove
    removed_image_ids = set()
    kept_images = []
    for img in coco["images"]:
        if img["file_name"] in removed_filenames:
            removed_image_ids.add(img["id"])
        else:
            kept_images.append(img)
    coco["images"] = kept_images

    # Remove annotations for those images
    if "annotations" in coco:
        coco["annotations"] = [ann for ann in coco["annotations"] if ann["image_id"] not in removed_image_ids]
    # Remove predictions for those images (if present)
    if "predictions" in coco:
        coco["predictions"] = [pred for pred in coco["predictions"] if pred["image_id"] not in removed_image_ids]

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description="Remove images starting with 'coco' and update COCO-format JSON.")
    parser.add_argument("--images_folder", required=True, help="Path to images folder")
    parser.add_argument("--json_file", required=True, help="Path to COCO-format JSON file (train.json)")
    args = parser.parse_args()

    print("Removing images starting with 'coco' from folder...")
    removed_filenames = remove_coco_images(args.images_folder)
    print(f"Removed {len(removed_filenames)} images.")

    print("Updating COCO-format JSON...")
    update_coco_json(args.json_file, removed_filenames)
    print("COCO-format JSON updated.")

if __name__ == "__main__":
    main()