import argparse
import json
import os
from pycocotools.coco import COCO
# Note: This script requires your custom 'cocoeval_modified.py' file.
from pycocotools.cocoeval_modified import COCOeval

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate COCO-style object detection results against a ground truth."
    )
    parser.add_argument(
        '-gt', '--ground_truth',
        required=True,
        type=str,
        help="Path to the ground truth COCO-style JSON file."
    )
    parser.add_argument(
        '-d', '--detections',
        required=True,
        type=str,
        help="Path to the detection results JSON file."
    )
    return parser.parse_args()

def main():
    """Main function to run the evaluation."""
    args = parse_args()

    # --- 1. Load Ground Truth Data ---
    print(f"Loading ground truth from: {args.ground_truth}")
    if not os.path.exists(args.ground_truth):
        print(f"Error: Ground truth file not found at {args.ground_truth}")
        return
    coco_gt = COCO(args.ground_truth)

    # Use a set for efficient lookups
    gt_image_ids = set(coco_gt.getImgIds())
    print(f"Found {len(gt_image_ids)} images in the ground truth.")

    # --- 2. Load and Filter Detections ---
    print(f"Loading detections from: {args.detections}")
    if not os.path.exists(args.detections):
        print(f"Error: Detections file not found at {args.detections}")
        return
        
    with open(args.detections, 'r') as f:
        detection_data = json.load(f)

    # Filter detections to only include images present in the ground truth
    print(f"Filtering {len(detection_data)} total detections...")
    filtered_detections = [
        item for item in detection_data if item.get('image_id') in gt_image_ids
    ]
    
    if not filtered_detections:
        print("\nWarning: No detections matched the image IDs in the ground truth.")
        print("Please ensure 'image_id' fields in your detection file match the 'id' fields in the ground truth 'images' list.")
        return
        
    print(f"Found {len(filtered_detections)} detections matching ground truth image IDs.")

    # --- 3. Run COCO Evaluation ---
    # Load filtered detections directly from the list in memory (no temp file needed)
    coco_dt = coco_gt.loadRes(filtered_detections)
    
    print("\nRunning COCO evaluation...")
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    
    print("\n--- Standard COCO Summary ---")
    coco_eval.summarize()  # Prints the standard 12-metric summary

    # --- 4. Print Custom Summary ---
    print("\n--- Custom Metric Summary ---")
    stats = coco_eval.stats
    print(f'AP @ IoU=0.50:0.95 (Primary Challenge Metric): {stats[0]:.4f}')
    print(f'AP @ IoU=0.50 (PASCAL VOC Metric):          {stats[1]:.4f}')
    print(f'AP for small objects:                       {stats[3]:.4f}')
    print(f'AP for medium objects:                      {stats[4]:.4f}')
    print(f'AP for large objects:                       {stats[5]:.4f}')
    
    # Safely check for the custom F1 score from your modified evaluator
    if len(stats) > 20:
        print(f'F1 Score (from modified eval):              {stats[20]:.4f}')
    else:
        print("F1 score (stat 20) not found. Is 'cocoeval_modified' correctly implemented?")
    print('----------------------------------------')

if __name__ == "__main__":
    main()