import json
import argparse
from typing import Dict, List, Any

def is_night_image(image_id: int) -> bool:
    """
    Determine if an image is a night image based on scene indicator.
    Night images have "_E_" or "_N_" in their filename.
    This extracts the scene index directly from the image_id.
    """
    try:
        # Convert to string to extract components
        id_str = str(image_id)
        
        # Handle different ID formats more robustly
        if len(id_str) < 2:
            return False  # Default to day if ID format is unexpected
        
        # Extract scene index (second digit)
        scene_idx = int(id_str[1])
        
        # scene_list = ['M', 'A', 'E', 'N'] -> indices [0, 1, 2, 3]
        # Night scenes are 'E' (index 2) and 'N' (index 3)
        return scene_idx in [2, 3]
        
    except (ValueError, IndexError):
        # If parsing fails, default to day image (safer assumption)
        print(f"Warning: Could not parse image_id {image_id}, assuming day image")
        return False

def filter_predictions(
    predictions: List[Dict[str, Any]], 
    day_thresholds: Dict[int, float], 
    night_thresholds: Dict[int, float]
) -> List[Dict[str, Any]]:
    """
    Filter predictions based on confidence thresholds for each class and time of day.
    
    Args:
        predictions: List of prediction dictionaries
        day_thresholds: Dictionary mapping category_id to confidence threshold for day images
        night_thresholds: Dictionary mapping category_id to confidence threshold for night images
    
    Returns:
        List of filtered predictions
    """
    filtered_predictions = []
    
    for pred in predictions:
        image_id = pred["image_id"]
        category_id = pred["category_id"]
        score = pred["score"]
        
        # Determine if it's a night image
        is_night = is_night_image(image_id)
        
        # Select appropriate threshold
        if is_night:
            threshold = night_thresholds.get(category_id, 0.5)  # Default to 0.5 if not specified
        else:
            threshold = day_thresholds.get(category_id, 0.5)    # Default to 0.5 if not specified
        
        # Keep prediction if score meets threshold
        if score >= threshold:
            filtered_predictions.append(pred)
    
    return filtered_predictions

def main():
    parser = argparse.ArgumentParser(description="Filter predictions by confidence thresholds")
    parser.add_argument("input_file", help="Input JSON file with predictions")
    parser.add_argument("output_file", help="Output JSON file for filtered predictions")
    parser.add_argument("--day-thresholds", type=str, 
                       help="Day thresholds as comma-separated values, e.g., '0.5,0.4,0.6,0.7,0.5' (Bus,Bike,Car,Pedestrian,Truck)")
    parser.add_argument("--night-thresholds", type=str,
                       help="Night thresholds as comma-separated values, e.g., '0.4,0.3,0.5,0.6,0.4' (Bus,Bike,Car,Pedestrian,Truck)")
    
    args = parser.parse_args()
    
    # Default thresholds (can be customized)
    default_day_thresholds = {
        0: 0.5,  # Bus
        1: 0.5,  # Bike
        2: 0.5,  # Car
        3: 0.5,  # Pedestrian
        4: 0.5   # Truck
    }
    
    default_night_thresholds = {
        0: 0.4,  # Bus (lower threshold for night)
        1: 0.4,  # Bike
        2: 0.4,  # Car
        3: 0.4,  # Pedestrian
        4: 0.4   # Truck
    }
    
    # Parse custom thresholds if provided
    day_thresholds = default_day_thresholds
    night_thresholds = default_night_thresholds
    
    if args.day_thresholds:
        threshold_values = [float(x) for x in args.day_thresholds.split(',')]
        if len(threshold_values) != 5:
            raise ValueError("Day thresholds must have exactly 5 values (one for each class)")
        day_thresholds = {i: threshold_values[i] for i in range(5)}
    
    if args.night_thresholds:
        threshold_values = [float(x) for x in args.night_thresholds.split(',')]
        if len(threshold_values) != 5:
            raise ValueError("Night thresholds must have exactly 5 values (one for each class)")
        night_thresholds = {i: threshold_values[i] for i in range(5)}
    
    print("Day thresholds:", day_thresholds)
    print("Night thresholds:", night_thresholds)
    
    # Load predictions
    with open(args.input_file, 'r') as f:
        predictions = json.load(f)
    
    print(f"Loaded {len(predictions)} predictions")
    
    # Filter predictions
    filtered_predictions = filter_predictions(predictions, day_thresholds, night_thresholds)
    
    print(f"Filtered to {len(filtered_predictions)} predictions")
    
    # Save filtered predictions
    with open(args.output_file, 'w') as f:
        json.dump(filtered_predictions, f, indent=2)
    
    print(f"Saved filtered predictions to {args.output_file}")

if __name__ == "__main__":
    main()

# Example usage:
# python filter_predictions.py input.json output.json --day-thresholds '0.6,0.5,0.4,0.7,0.5' --night-thresholds '0.4,0.3,0.3,0.5,0.4'

# Alternative usage with default thresholds:
# python filter_predictions.py input.json output.json