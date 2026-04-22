import os
import argparse
from collections import Counter
import pandas as pd

def analyze_yolo_labels(labels_dir):
    class_names = {
        0: "Bus",
        1: "Bike",
        2: "Car",
        3: "Pedestrian",
        4: "Truck"
    }

    class_counter = Counter()
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

    for file_name in label_files:
        file_path = os.path.join(labels_dir, file_name)
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1:
                    class_id = int(parts[0])
                    class_counter[class_id] += 1

    # Prepare table
    data = []
    for class_id in range(5):  # Classes 0 to 4
        data.append({
            "Class ID": class_id,
            "Class Name": class_names[class_id],
            "Instances": class_counter[class_id]
        })

    df = pd.DataFrame(data)
    total_images = len(label_files)

    print("Total Training Samples (Images):", total_images)
    print("\nClass Distribution:")
    print(df.to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description="Analyze YOLO label distribution")
    parser.add_argument("--labels_dir", type=str, required=True,
                        help="Path to the directory containing YOLO label .txt files")

    args = parser.parse_args()
    analyze_yolo_labels(args.labels_dir)

if __name__ == "__main__":
    main()

