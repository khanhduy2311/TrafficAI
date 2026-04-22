import json
import argparse
from collections import Counter
import pandas as pd

def analyze_coco_annotations(json_path):
    class_names = {
        0: "Bus",
        1: "Bike",
        2: "Car",
        3: "Pedestrian",
        4: "Truck"
    }

    with open(json_path, 'r') as f:
        coco = json.load(f)

    image_ids = set()
    class_counter = Counter()

    for ann in coco['annotations']:
        class_id = ann['category_id']
        class_counter[class_id] += 1
        image_ids.add(ann['image_id'])

    # Build output table
    data = []
    for class_id in range(5):  # Classes 0 to 4
        data.append({
            "Class ID": class_id,
            "Class Name": class_names[class_id],
            "Instances": class_counter[class_id]
        })

    df = pd.DataFrame(data)
    total_images = len(set(img['id'] for img in coco['images']))

    print("Total Training Samples (Images):", total_images)
    print("\nClass Distribution:")
    print(df.to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description="Analyze COCO-format JSON annotations")
    parser.add_argument("--json_path", type=str, required=True,
                        help="Path to COCO-format annotation JSON file")

    args = parser.parse_args()
    analyze_coco_annotations(args.json_path)

if __name__ == "__main__":
    main()

