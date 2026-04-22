"""
filter_predictions.py

A standalone script to filter a COCO-style prediction JSON file based on 
time-of-day (ToD) specific, per-class confidence thresholds.

This script takes a raw prediction file and applies the optimal thresholds 
found by an evaluation script (like the one provided) to produce a final, 
optimized submission file.

Example Usage:
python filter_predictions.py \
    --pred_file "path/to/raw_predictions.json" \
    --gt_file "path/to/ground_truth.json" \
    --output_file "path/to/filtered_submission.json" \
    --day_thresholds "0.65,0.55,0.70,0.50,0.60" \
    --night_thresholds "0.50,0.45,0.60,0.40,0.55" \
    --default_threshold 0.1
"""

import json
import os
import argparse
from collections import defaultdict

def parse_threshold_list_to_map(threshold_str_list):
    """
    Parses a comma-separated string of threshold values into a class_id:threshold_value map.
    This function is consistent with the one in the provided inference script.
    """
    threshold_map = {}
    if not threshold_str_list:
        return threshold_map
    try:
        # Class IDs are 0-indexed, so the list maps directly
        # Example: "0.5,0.4" -> {0: 0.5, 1: 0.4}
        threshold_values = [float(t.strip()) for t in threshold_str_list.split(',')]
        threshold_map = {i: threshold for i, threshold in enumerate(threshold_values)}
    except ValueError as e:
        raise ValueError(f"Invalid format for threshold list. Expected comma-separated floats. Got: '{threshold_str_list}'. Error: {e}")
    return threshold_map


def build_image_to_tod_map(gt_data):
    """
    Builds a dictionary mapping image_id to its time-of-day ('day' or 'night').
    
    Args:
        gt_data (dict): The loaded COCO-style ground truth JSON data.

    Returns:
        dict: A map of {image_id (int): tod_string (str)}.
    """
    image_to_tod = {}
    if 'images' not in gt_data:
        raise KeyError("Ground truth data is missing the 'images' key.")
        
    for img_info in gt_data['images']:
        image_id = img_info['id']
        file_name = img_info.get('file_name', '')
        
        # Night images are marked with '_N_' or '_E_' in their filenames
        if "_N_" in file_name or "_E_" in file_name:
            image_to_tod[image_id] = 'night'
        else:
            image_to_tod[image_id] = 'day'
            
    return image_to_tod


def main(args):
    """Main function to run the prediction filtering process."""
    # 1. Load input files
    print(f"Loading predictions from: {args.pred_file}")
    with open(args.pred_file, 'r') as f:
        all_detections = json.load(f)

    print(f"Loading ground truth from: {args.gt_file} to determine Time-of-Day for each image.")
    with open(args.gt_file, 'r') as f:
        gt_data = json.load(f)

    # 2. Parse the command-line thresholds into dictionaries
    try:
        day_threshold_map = parse_threshold_list_to_map(args.day_thresholds)
        night_threshold_map = parse_threshold_list_to_map(args.night_thresholds)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
        
    print("\n--- Applying Thresholds ---")
    print(f"Day Thresholds (Class ID: Threshold): {day_threshold_map}")
    print(f"Night Thresholds (Class ID: Threshold): {night_threshold_map}")
    print(f"Default Fallback Threshold: {args.default_threshold}")
    
    # 3. Create the mapping from image_id to time-of-day
    image_to_tod_map = build_image_to_tod_map(gt_data)
    
    # 4. Filter the detections
    print("\nFiltering detections...")
    filtered_detections = []
    
    for det in all_detections:
        image_id = det['image_id']
        category_id = det['category_id']
        score = det['score']
        
        # Determine which threshold map to use
        image_tod = image_to_tod_map.get(image_id)
        
        if image_tod == 'night':
            tod_specific_map = night_threshold_map
        elif image_tod == 'day':
            tod_specific_map = day_threshold_map
        else:
            # This case handles detections for images not present in the GT file.
            # We'll use the default threshold for all classes in this scenario.
            tod_specific_map = {} # Empty map ensures fallback to default
            print(f"Warning: Image ID {image_id} from predictions not found in GT file. Using default threshold.")

        # Get the specific threshold for the class, or use the default
        threshold_for_class = tod_specific_map.get(category_id, args.default_threshold)
        
        # Apply the filter
        if score >= threshold_for_class:
            filtered_detections.append(det)

    # 5. Save the filtered results
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(args.output_file, 'w') as f:
        json.dump(filtered_detections, f, indent=2)

    # 6. Print summary
    print("\n--- Filtering Complete ---")
    print(f"Total detections in original file: {len(all_detections)}")
    print(f"Total detections in filtered file:  {len(filtered_detections)}")
    print(f"Filtered predictions saved to: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter COCO-style predictions using Time-of-Day (ToD) specific, per-class thresholds.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument("-p", "--pred_file", required=True, 
                        help="Path to the raw COCO format predictions JSON file.")
    parser.add_argument("-g", "--gt_file", required=True, 
                        help="Path to the COCO format ground truth JSON file (used to map image_id to filename/ToD).")
    parser.add_argument("-o", "--output_file", required=True, 
                        help="Path to save the new, filtered predictions JSON file.")
    
    parser.add_argument("-day", "--day_thresholds", type=str, default="",
                        help='Per-class thresholds for DAY images as a comma-separated string.\n'
                             'Order corresponds to class IDs 0, 1, 2, ...\n'
                             'Example: "0.5,0.4,0.6"')
    parser.add_argument("-night", "--night_thresholds", type=str, default="",
                        help='Per-class thresholds for NIGHT images as a comma-separated string.\n'
                             'Example: "0.4,0.3,0.5"')
    parser.add_argument("--default_threshold", type=float, default=0.1, 
                        help="Default confidence threshold to use if a class is not in the specific ToD map.")
                        
    args = parser.parse_args()
    
    # Basic input validation
    if not os.path.exists(args.pred_file):
        print(f"Error: Prediction file not found at {args.pred_file}")
        exit(1)
    if not os.path.exists(args.gt_file):
        print(f"Error: Ground truth file not found at {args.gt_file}")
        exit(1)
        
    main(args)
