import json
import argparse
from collections import defaultdict

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(boxAArea + boxBArea - inter)

# Map original classes to 5 target categories (0-indexed)
CLASS_MAP = {
    "bus": ("bus", 0),
    "bicycle": ("bike", 1),
    "cargo trike": ("bike", 1),
    "motorcycle": ("bike", 1),
    "rickshaw": ("car", 2),
    "car": ("car", 2),
    "van": ("car", 2),
    "person": ("pedestrian", 3),
    "backhoe loader": ("truck", 4),
    "small truck": ("truck", 4),
    "tractor": ("truck", 4),
    "truck": ("truck", 4),
}

NEW_CATEGORIES = [
    {"id": 0, "name": "bus"},
    {"id": 1, "name": "bike"},
    {"id": 2, "name": "car"},
    {"id": 3, "name": "pedestrian"},
    {"id": 4, "name": "truck"},
]

def convert_coco(input_json, output_json):
    with open(input_json, 'r') as f:
        coco = json.load(f)

    images = coco['images']
    annotations = coco['annotations']
    categories = coco['categories']

    id_to_name = {cat['id']: cat['name'] for cat in categories}
    name_to_new_id = {v[0]: v[1] for v in CLASS_MAP.values()}
    img_to_anns = defaultdict(list)
    for ann in annotations:
        img_to_anns[ann['image_id']].append(ann)

    new_annotations = []

    for img in images:
        img_id = img['id']
        anns = img_to_anns[img_id]

        motorcycles = []
        persons = []
        others = []

        bike_like_classes = {"motorcycle", "bicycle", "cargo trike"}

        for ann in anns:
            orig_class_name = id_to_name[ann['category_id']]
            if orig_class_name == "ignore" or orig_class_name not in CLASS_MAP:
                continue
            if orig_class_name in bike_like_classes:
                motorcycles.append((ann, orig_class_name))
            elif orig_class_name == "person":
                persons.append(ann)
            else:
                others.append(ann)

        # Step 1: Init original bike boxes
        bike_id_to_box = {}
        for moto_ann, _ in motorcycles:
            ann_id = moto_ann['id']
            x1 = moto_ann['bbox'][0]
            y1 = moto_ann['bbox'][1]
            x2 = x1 + moto_ann['bbox'][2]
            y2 = y1 + moto_ann['bbox'][3]
            bike_id_to_box[ann_id] = [x1, y1, x2, y2]

        # Step 2: Merge each person into best-matching bike IF above it
        used_person_ids = set()

        for person in persons:
            best_iou = 0
            best_bike_id = None
            person_box = [
                person['bbox'][0],
                person['bbox'][1],
                person['bbox'][0] + person['bbox'][2],
                person['bbox'][1] + person['bbox'][3],
            ]
            person_center_y = (person_box[1] + person_box[3]) / 2

            for moto_ann, _ in motorcycles:
                bike_box = bike_id_to_box[moto_ann['id']]
                bike_center_y = (bike_box[1] + bike_box[3]) / 2
                score = iou(bike_box, person_box)

                if score > best_iou and person_center_y < bike_center_y:
                    best_iou = score
                    best_bike_id = moto_ann['id']

            if best_iou > 0 and best_bike_id is not None:
                used_person_ids.add(person['id'])
                b = bike_id_to_box[best_bike_id]
                bike_id_to_box[best_bike_id] = [
                    min(b[0], person_box[0]),
                    min(b[1], person_box[1]),
                    max(b[2], person_box[2]),
                    max(b[3], person_box[3])
                ]

        # Step 3: Write updated bike boxes
        for moto_ann, _ in motorcycles:
            b = bike_id_to_box[moto_ann['id']]
            moto_ann['bbox'] = [b[0], b[1], b[2] - b[0], b[3] - b[1]]
            moto_ann['category_id'] = name_to_new_id["bike"]
            new_annotations.append(moto_ann)

        # Step 4: Add others (remapped)
        for ann in others:
            cname = id_to_name[ann['category_id']]
            new_name, new_id = CLASS_MAP[cname]
            ann['category_id'] = new_id
            new_annotations.append(ann)

        # Step 5: Add unmerged pedestrians
        for person in persons:
            if person['id'] in used_person_ids:
                continue

            person_box = [
                person['bbox'][0],
                person['bbox'][1],
                person['bbox'][0] + person['bbox'][2],
                person['bbox'][1] + person['bbox'][3],
            ]

            merged = False
            for moto_ann, _ in motorcycles:
                b = bike_id_to_box[moto_ann['id']]
                bike_box = [b[0], b[1], b[2], b[3]]
                if iou(person_box, bike_box) >= 0.9:
                    bike_id_to_box[moto_ann['id']] = [
                        min(bike_box[0], person_box[0]),
                        min(bike_box[1], person_box[1]),
                        max(bike_box[2], person_box[2]),
                        max(bike_box[3], person_box[3])
                    ]
                    merged = True
                    break

            if not merged:
                person['category_id'] = name_to_new_id["pedestrian"]
                new_annotations.append(person)

    new_coco = {
        "images": images,
        "annotations": new_annotations,
        "categories": NEW_CATEGORIES
    }

    with open(output_json, 'w') as f:
        json.dump(new_coco, f, indent=2)
    print(f"[INFO] Saved cleaned COCO JSON to: {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge riders above bikes and remap COCO classes to 5 categories.")
    parser.add_argument("--input", type=str, required=True, help="Path to input COCO JSON file")
    parser.add_argument("--output", type=str, required=True, help="Path to output cleaned COCO JSON file")
    args = parser.parse_args()

    convert_coco(args.input, args.output)
