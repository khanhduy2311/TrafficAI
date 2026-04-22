import json
import os
import numpy as np
import matplotlib.pyplot as plt
# Standard COCO import
from pycocotools.coco import COCO
# Import your modified COCOeval from the pycocotools package
from pycocotools.cocoeval_modified import COCOeval
import shutil
import tempfile
import copy
import argparse # For command-line arguments

# --- Helper Functions ---

def save_coco_subset_annotations(original_gt_data, image_ids_subset, category_id_subset=None):
    """
    Creates a new COCO-like dictionary containing only the specified image_ids
    and their corresponding annotations, optionally filtered by category_id.
    This is used to create temporary GT structures for COCO() initialization.
    """
    subset_images = [img for img in original_gt_data['images'] if img['id'] in image_ids_subset]
    subset_image_ids_set = {img['id'] for img in subset_images}

    subset_annotations = []
    for ann in original_gt_data['annotations']:
        if ann['image_id'] in subset_image_ids_set:
            if category_id_subset is None or ann['category_id'] == category_id_subset:
                subset_annotations.append(ann)
    
    categories_to_use = original_gt_data['categories']
    if category_id_subset is not None:
        categories_to_use = [cat for cat in original_gt_data['categories'] if cat['id'] == category_id_subset]


    subset_coco_dict = {
        'images': subset_images,
        'annotations': subset_annotations,
        'categories': categories_to_use
    }
    if 'info' in original_gt_data: subset_coco_dict['info'] = original_gt_data['info']
    if 'licenses' in original_gt_data: subset_coco_dict['licenses'] = original_gt_data['licenses']

    return subset_coco_dict


def get_f1_score(gt_coco_obj, # This should be a COCO object for the relevant subset (e.g., day images)
                dt_detections_list, # This is a list of detection dicts
                temp_dir_for_files,
                eval_params_template, # COCOeval.params object template
                specific_cat_id_to_eval=None,
                img_ids_to_eval_explicitly=None): # Explicit list of image IDs for this eval run
    """
    Calculates F1 score using COCOeval.
    - gt_coco_obj: A COCO object, potentially already subsetted for ToD and/or category.
    - dt_detections_list: A list of detection dictionaries to evaluate.
    - temp_dir_for_files: Path to a temporary directory for writing detection files.
    - eval_params_template: A COCOeval.Params object to use as a base.
    - specific_cat_id_to_eval: If provided, F1 is for that category.
    - img_ids_to_eval_explicitly: Explicitly sets imgIds for COCOeval from this list.
    """
    if not dt_detections_list: # No detections to evaluate
        return 0.0

    # Ensure detections are for images present in the gt_coco_obj
    gt_present_image_ids = set(gt_coco_obj.getImgIds())
    filtered_detections_for_gt_imgs = [
        det for det in dt_detections_list if det['image_id'] in gt_present_image_ids
    ]

    if not filtered_detections_for_gt_imgs:
        return 0.0

    # loadRes needs a file path
    temp_dt_path = os.path.join(temp_dir_for_files, "temp_dt_for_current_eval.json")
    with open(temp_dt_path, 'w') as f:
        json.dump(filtered_detections_for_gt_imgs, f)

    coco_dt = gt_coco_obj.loadRes(temp_dt_path)
    
    current_eval_params = copy.deepcopy(eval_params_template) # Use a copy

    coco_eval = COCOeval(gt_coco_obj, coco_dt, current_eval_params.iouType)
    coco_eval.params = current_eval_params # Assign the copied and potentially modified params

    # Override imgIds and catIds for this specific evaluation run if needed
    if img_ids_to_eval_explicitly:
        # Intersect with images actually in gt_coco_obj to be safe
        valid_img_ids = sorted(list(set(img_ids_to_eval_explicitly).intersection(gt_present_image_ids)))
        if not valid_img_ids: return 0.0 # No common images
        coco_eval.params.imgIds = valid_img_ids
    else: # Default to all image IDs in the provided gt_coco_obj
        coco_eval.params.imgIds = sorted(list(gt_present_image_ids))
        if not coco_eval.params.imgIds: return 0.0


    if specific_cat_id_to_eval is not None:
        # Ensure this category is actually in gt_coco_obj categories
        gt_cat_ids = set(c['id'] for c in gt_coco_obj.dataset['categories'])
        if specific_cat_id_to_eval not in gt_cat_ids:
            # print(f"Warning: Category {specific_cat_id_to_eval} not in GT categories for this subset. F1=0.")
            return 0.0
        coco_eval.params.catIds = [specific_cat_id_to_eval]
    else: # Evaluate all categories present in gt_coco_obj
        coco_eval.params.catIds = sorted(list(c['id'] for c in gt_coco_obj.dataset['categories']))
        if not coco_eval.params.catIds: return 0.0


    if not coco_eval.params.imgIds or not coco_eval.params.catIds:
        return 0.0

    # print(f"Running eval with {len(coco_eval.params.imgIds)} images and categories {coco_eval.params.catIds}")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize() # Optional: print detailed summary for each small run

    f1_score_val = coco_eval.stats[20] # F1 score for maxDets=100 (index for F1 @ IoU=0.50:0.95 | area=all | maxDets=100)
    return f1_score_val if f1_score_val != -1 else 0.0

def ensemble_detections(detections1, detections2, iou_threshold=0.5, weight1=1.0, weight2=1.0):
    """
    Ensemble detections from two prediction sources using NMS and weighted scores.
    
    Args:
        detections1: List of detection dictionaries from first model
        detections2: List of detection dictionaries from second model
        iou_threshold: IoU threshold for considering two boxes as the same object
        weight1: Weight for scores from the first model
        weight2: Weight for scores from the second model
    
    Returns:
        List of merged detection dictionaries
    """
    # Group all detections by image_id and category_id
    all_detections_by_img_cat = {}
    
    # Process detections from first model
    for det in detections1:
        img_id = det['image_id']
        cat_id = det['category_id']
        key = (img_id, cat_id)
        if key not in all_detections_by_img_cat:
            all_detections_by_img_cat[key] = []
        # Add source identifier and apply weight
        det_copy = copy.deepcopy(det)
        det_copy['score'] *= weight1
        det_copy['source'] = 'model1'
        all_detections_by_img_cat[key].append(det_copy)
    
    # Process detections from second model
    for det in detections2:
        img_id = det['image_id']
        cat_id = det['category_id']
        key = (img_id, cat_id)
        if key not in all_detections_by_img_cat:
            all_detections_by_img_cat[key] = []
        # Add source identifier and apply weight
        det_copy = copy.deepcopy(det)
        det_copy['score'] *= weight2
        det_copy['source'] = 'model2'
        all_detections_by_img_cat[key].append(det_copy)
    
    ensembled_detections = []
    
    # Process each image and category group
    for (img_id, cat_id), detections in all_detections_by_img_cat.items():
        # Sort by score in descending order
        detections.sort(key=lambda x: x['score'], reverse=True)
        
        # Apply NMS
        kept_detections = []
        for det in detections:
            # Check if current detection overlaps with any kept detection
            should_keep = True
            for kept_det in kept_detections:
                iou = calculate_iou(det['bbox'], kept_det['bbox'])
                if iou > iou_threshold:
                    # If current detection has higher score, replace the kept one
                    if det['score'] > kept_det['score']:
                        kept_detections.remove(kept_det)
                        should_keep = True
                        break
                    else:
                        should_keep = False
                        break
            
            if should_keep:
                # Make a clean copy without source field for final output
                clean_det = {
                    'image_id': det['image_id'],
                    'category_id': det['category_id'],
                    'bbox': det['bbox'],
                    'score': det['score'],
                    'area': det.get('area', det['bbox'][2] * det['bbox'][3]),
                    'id': len(ensembled_detections)  # New unique ID
                }
                if 'segmentation' in det:
                    clean_det['segmentation'] = det['segmentation']
                
                kept_detections.append(det)
                ensembled_detections.append(clean_det)
    
    return ensembled_detections

def calculate_iou(bbox1, bbox2):
    """
    Calculate IoU between two bounding boxes in COCO format [x, y, width, height]
    """
    # Convert [x, y, width, height] to [x1, y1, x2, y2]
    box1 = [bbox1[0], bbox1[1], bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]]
    box2 = [bbox2[0], bbox2[1], bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]]
    
    # Get coordinates of intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate area of intersection and union
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    intersection = width * height
    
    area1 = bbox1[2] * bbox1[3]
    area2 = bbox2[2] * bbox2[3]
    union = area1 + area2 - intersection
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0
    return iou

def optimize_confidence_thresholds(gt_json_file_path, pred_json_file_path, output_dir, model_name, main_temp_dir):
    """
    Optimizes confidence thresholds for day and night for each category
    """
    print(f"\n=== Optimizing confidence thresholds for {model_name} ===")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load original GT and Detections
    with open(gt_json_file_path, 'r') as f:
        original_gt_data = json.load(f)
    
    coco_gt_original_obj = COCO(gt_json_file_path)

    with open(pred_json_file_path, 'r') as f:
        all_detections_data = json.load(f)

    # 2. Segregate Image IDs into Day and Night
    all_image_ids_original_gt = coco_gt_original_obj.getImgIds()
    day_image_ids = []
    night_image_ids = []
    # Store sets for faster lookups later
    day_image_ids_set = set()
    night_image_ids_set = set()

    for img_id in all_image_ids_original_gt:
        img_info = coco_gt_original_obj.loadImgs(img_id)[0]
        if "_N_" in img_info['file_name'] or "_E_" in img_info['file_name']:
            night_image_ids.append(img_id)
            night_image_ids_set.add(img_id)
        else:
            day_image_ids.append(img_id)
            day_image_ids_set.add(img_id)

    print(f"Total images in original GT: {len(all_image_ids_original_gt)}")
    print(f"Day images: {len(day_image_ids)}")
    print(f"Night images: {len(night_image_ids)}")

    coco_gt_day_obj, coco_gt_night_obj = None, None
    if day_image_ids:
        day_gt_dict = save_coco_subset_annotations(original_gt_data, day_image_ids)
        temp_gt_day_path = os.path.join(main_temp_dir, f"temp_gt_day_{model_name}.json")
        with open(temp_gt_day_path, 'w') as f: json.dump(day_gt_dict, f)
        coco_gt_day_obj = COCO(temp_gt_day_path)
    if night_image_ids:
        night_gt_dict = save_coco_subset_annotations(original_gt_data, night_image_ids)
        temp_gt_night_path = os.path.join(main_temp_dir, f"temp_gt_night_{model_name}.json")
        with open(temp_gt_night_path, 'w') as f: json.dump(night_gt_dict, f)
        coco_gt_night_obj = COCO(temp_gt_night_path)

    all_category_ids = sorted(coco_gt_original_obj.getCatIds())
    category_names = {cat['id']: cat['name'] for cat in coco_gt_original_obj.loadCats(all_category_ids)}

    threshold_range = np.arange(0.2, 0.9, 0.025) # Start from 0.2 with 0.025 steps

    optimal_thresholds = {'day': {}, 'night': {}}
    all_f1_curves = {'day': {}, 'night': {}}

    base_eval_params = COCOeval(iouType='bbox').params
    DEFAULT_FALLBACK_THRESHOLD = 0.5 # Define a default fallback threshold

    # 3. Optimize Thresholds
    for tod_label, current_gt_coco_obj, current_tod_image_ids in [
        ('day', coco_gt_day_obj, day_image_ids),
        ('night', coco_gt_night_obj, night_image_ids)
    ]:
        if not current_gt_coco_obj or not current_tod_image_ids:
            print(f"Skipping {tod_label} as there are no images or GT object.")
            optimal_thresholds[tod_label] = {} # Ensure it exists as an empty dict
            all_f1_curves[tod_label] = {}
            continue
        
        print(f"\n--- Optimizing for {tod_label.upper()} ---")
        tod_detections_all_classes = [d for d in all_detections_data if d['image_id'] in current_tod_image_ids]
        categories_in_tod_gt = current_gt_coco_obj.loadCats(current_gt_coco_obj.getCatIds())

        for cat_info in categories_in_tod_gt:
            cat_id = cat_info['id']
            cat_name = category_names[cat_id]
            print(f"  Optimizing for class: {cat_name} (ID: {cat_id})")
            
            f1_scores_for_cat = []
            # Filter detections for the current category AND current ToD images
            cat_tod_detections = [d for d in all_detections_data 
                                if d['category_id'] == cat_id and d['image_id'] in current_tod_image_ids]
            
            if not cat_tod_detections:
                print(f"    No detections for class {cat_name} in {tod_label} images. Setting optimal threshold to 0.05, F1=0.")
                optimal_thresholds[tod_label][cat_id] = 0.05 # Default low threshold
                all_f1_curves[tod_label][cat_id] = (threshold_range, [0.0] * len(threshold_range))
                continue

            # Create a GT object specific to this ToD and this single category
            single_cat_tod_gt_dict = save_coco_subset_annotations(original_gt_data, current_tod_image_ids, category_id_subset=cat_id)
            temp_single_cat_tod_gt_path = os.path.join(main_temp_dir, f"temp_gt_{tod_label}_cat{cat_id}_{model_name}.json")
            with open(temp_single_cat_tod_gt_path, 'w') as f: json.dump(single_cat_tod_gt_dict, f)
            coco_gt_single_cat_tod_obj = COCO(temp_single_cat_tod_gt_path)
            
            # Ensure there are annotations for this category in this subset GT
            if not coco_gt_single_cat_tod_obj.getAnnIds(catIds=[cat_id]):
                print(f"    No GT annotations for class {cat_name} in {tod_label} images. Setting optimal threshold to 0.05, F1=0.")
                optimal_thresholds[tod_label][cat_id] = 0.05
                all_f1_curves[tod_label][cat_id] = (threshold_range, [0.0] * len(threshold_range))
                continue

            for conf_thresh in threshold_range:
                current_threshold_cat_detections = [
                    d for d in cat_tod_detections if d['score'] >= conf_thresh
                ]
                f1 = get_f1_score(
                    gt_coco_obj=coco_gt_single_cat_tod_obj, 
                    dt_detections_list=current_threshold_cat_detections,
                    temp_dir_for_files=main_temp_dir,
                    eval_params_template=base_eval_params,
                    specific_cat_id_to_eval=cat_id,
                    img_ids_to_eval_explicitly=current_tod_image_ids # Evaluate on images relevant to this ToD
                )
                f1_scores_for_cat.append(f1)

            if not f1_scores_for_cat or max(f1_scores_for_cat) == 0.0: # Check if any F1 > 0 was achieved
                optimal_thresholds[tod_label][cat_id] = threshold_range[np.argmax(f1_scores_for_cat)] if f1_scores_for_cat else 0.05 # Use best or default
                all_f1_curves[tod_label][cat_id] = (threshold_range, f1_scores_for_cat if f1_scores_for_cat else [0.0] * len(threshold_range))
                print(f"    No positive F1 scores for class {cat_name} in {tod_label}. Optimal threshold set to {optimal_thresholds[tod_label][cat_id]:.2f} (F1: {max(f1_scores_for_cat) if f1_scores_for_cat else 0.0:.3f}).")
                continue

            best_f1_idx = np.argmax(f1_scores_for_cat)
            optimal_thresholds[tod_label][cat_id] = threshold_range[best_f1_idx]
            all_f1_curves[tod_label][cat_id] = (threshold_range, f1_scores_for_cat)
            print(f"    Optimal threshold for {cat_name} ({tod_label}): {optimal_thresholds[tod_label][cat_id]:.2f} (F1: {f1_scores_for_cat[best_f1_idx]:.3f})")

    # 4. Plot F1 curves and save to output_dir
    for tod_label in ['day', 'night']:
        if not all_f1_curves.get(tod_label): continue # Use .get for safety if ToD was skipped
        
        plotted_curves = {k: v for k, v in all_f1_curves[tod_label].items() if v[1]} # v[1] is f1_values list
        num_cats_to_plot = len(plotted_curves)
        if num_cats_to_plot == 0: continue

        cols = 2
        rows = (num_cats_to_plot + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)
        axes_flat = axes.flatten()
        
        plot_idx = 0
        for cat_id, (thresholds, f1_values) in plotted_curves.items():
            if not f1_values: continue # Skip if f1_values is empty
            if plot_idx < len(axes_flat):
                ax = axes_flat[plot_idx]
                cat_name = category_names.get(cat_id, f"Unknown Cat {cat_id}")
                ax.plot(thresholds, f1_values, marker='o', linestyle='-')
                
                best_thresh_for_cat = optimal_thresholds[tod_label].get(cat_id)
                if best_thresh_for_cat is not None:
                    # Find the F1 value at the best_thresh_for_cat.
                    best_f1_val_for_plot = 0.0
                    try:
                        thresh_idx_for_plot = list(thresholds).index(best_thresh_for_cat)
                        best_f1_val_for_plot = f1_values[thresh_idx_for_plot]
                    except ValueError: # If best_thresh_for_cat not exactly in thresholds (e.g. due to default 0.05)
                        # Fallback to max f1 or 0
                        best_f1_val_for_plot = max(f1_values) if f1_values else 0.0

                    ax.scatter([best_thresh_for_cat], [best_f1_val_for_plot], color='red', s=100, zorder=5, label=f"Optimal: {best_thresh_for_cat:.2f} (F1: {best_f1_val_for_plot:.3f})")
                
                ax.set_title(f"F1 for {cat_name} ({tod_label.capitalize()})")
                ax.set_xlabel("Confidence Threshold")
                ax.set_ylabel("F1 Score (for this class)")
                ax.set_ylim(0, 1.05)
                ax.grid(True)
                ax.legend()
                plot_idx += 1

        for i in range(plot_idx, len(axes_flat)): # Hide unused subplots
            if i < len(axes_flat): fig.delaxes(axes_flat[i])

        fig.suptitle(f"F1 Score Optimization ({tod_label.capitalize()}) - {model_name}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plot_filename = os.path.join(output_dir, f"f1_curves_{tod_label}_{model_name}.png")
        plt.savefig(plot_filename)
        print(f"Saved F1 curves plot to {plot_filename}")
        plt.close(fig)

    # Return optimized thresholds and image ID sets
    return {
        'optimal_thresholds': optimal_thresholds,
        'day_image_ids': day_image_ids,
        'night_image_ids': night_image_ids,
        'day_image_ids_set': day_image_ids_set,
        'night_image_ids_set': night_image_ids_set,
        'all_category_ids': all_category_ids,
        'category_names': category_names
    }

def apply_optimized_confidence_thresholds(detections, optimal_thresholds, day_image_ids_set, night_image_ids_set):
    """
    Apply optimized confidence thresholds to detections based on time of day
    """
    filtered_detections = []
    DEFAULT_FALLBACK_THRESHOLD = 0.5
    
    for det in detections:
        img_id = det['image_id']
        cat_id = det['category_id']
        
        # Determine time of day
        if img_id in day_image_ids_set:
            tod_label = 'day'
        elif img_id in night_image_ids_set:
            tod_label = 'night'
        else:
            # If image not in day or night set, use default threshold
            if det['score'] >= DEFAULT_FALLBACK_THRESHOLD:
                filtered_detections.append(det)
            continue
        
        # Get threshold for this category and time of day
        threshold = DEFAULT_FALLBACK_THRESHOLD
        if tod_label in optimal_thresholds and cat_id in optimal_thresholds[tod_label]:
            threshold = optimal_thresholds[tod_label][cat_id]
        
        # Apply threshold
        if det['score'] >= threshold:
            filtered_detections.append(det)
    
    return filtered_detections

def optimize_iou_threshold_for_ensemble(gt_coco_obj, detections1, detections2, day_image_ids, night_image_ids, 
                                        optimal_thresholds1, optimal_thresholds2, 
                                        day_image_ids_set, night_image_ids_set, 
                                        temp_dir, output_dir, weight1=1.0, weight2=1.0):
    """
    Optimize IoU threshold for ensemble and find the best weights
    """
    print("\n=== Optimizing IoU threshold and weights for ensemble ===")
    
    # Apply optimized confidence thresholds to both detection sets
    filtered_detections1 = apply_optimized_confidence_thresholds(
        detections1, optimal_thresholds1, day_image_ids_set, night_image_ids_set
    )
    filtered_detections2 = apply_optimized_confidence_thresholds(
        detections2, optimal_thresholds2, day_image_ids_set, night_image_ids_set
    )
    
    print(f"After applying confidence thresholds: {len(filtered_detections1)} detections from model1, {len(filtered_detections2)} detections from model2")
    
    # Define ranges for parameters to optimize
    iou_thresholds = np.arange(0.1, 0.9, 0.05)
    weight_combinations = [
        (1.0, 1.0),
        (0.3, 0.5),
        (0.5, 0.3),
        (1.0, 0.8),
        (0.8, 1.0),
        (1.0, 0.6),
        (0.6, 1.0),
        (0.7, 0.7)
    ]
    
    # Evaluation parameters
    base_eval_params = COCOeval(iouType='bbox').params
    
    # Store results for all combinations
    results = []
    
    # First, evaluate each model separately with optimized confidence thresholds
    f1_model1 = get_f1_score(
        gt_coco_obj=gt_coco_obj,
        dt_detections_list=filtered_detections1,
        temp_dir_for_files=temp_dir,
        eval_params_template=base_eval_params,
        specific_cat_id_to_eval=None,
        img_ids_to_eval_explicitly=None
    )
    
    f1_model2 = get_f1_score(
        gt_coco_obj=gt_coco_obj,
        dt_detections_list=filtered_detections2,
        temp_dir_for_files=temp_dir,
        eval_params_template=base_eval_params,
        specific_cat_id_to_eval=None,
        img_ids_to_eval_explicitly=None
    )
    
    print(f"Baseline F1 scores after confidence optimization:")
    print(f"  Model1: {f1_model1:.4f}")
    print(f"  Model2: {f1_model2:.4f}")
    
    # Track best F1 and parameters
    best_f1 = max(f1_model1, f1_model2)
    best_params = {
        'iou_threshold': 0.5,
        'weight1': 1.0,
        'weight2': 1.0,
        'is_ensemble': False,
        'model': 'model1' if f1_model1 >= f1_model2 else 'model2'
    }
    
    # Grid search for best parameters
    for iou_threshold in iou_thresholds:
        for weight1, weight2 in weight_combinations:
            print(f"Testing IoU={iou_threshold:.2f}, weights=({weight1:.1f}, {weight2:.1f})")
            
            # Perform ensemble with current parameters
            ensembled_detections = ensemble_detections(
                filtered_detections1, 
                filtered_detections2, 
                iou_threshold=iou_threshold,
                weight1=weight1,
                weight2=weight2
            )
            
            # Evaluate ensemble
            f1_ensemble = get_f1_score(
                gt_coco_obj=gt_coco_obj,
                dt_detections_list=ensembled_detections,
                temp_dir_for_files=temp_dir,
                eval_params_template=base_eval_params,
                specific_cat_id_to_eval=None,
                img_ids_to_eval_explicitly=None
            )
            
            print(f"  Ensemble F1: {f1_ensemble:.4f}")
            
            # Store result
            results.append({
                'iou_threshold': iou_threshold,
                'weight1': weight1,
                'weight2': weight2,
                'f1_score': f1_ensemble
            })
            
            # Update best parameters if better F1 score
            if f1_ensemble > best_f1:
                best_f1 = f1_ensemble
                best_params = {
                    'iou_threshold': iou_threshold,
                    'weight1': weight1,
                    'weight2': weight2,
                    'is_ensemble': True
                }
    
    # Sort results by F1 score
    results.sort(key=lambda x: x['f1_score'], reverse=True)
    
    # Print top 5 results
    print("\nTop 5 ensemble configurations:")
    for i, result in enumerate(results[:5]):
        print(f"{i+1}. IoU={result['iou_threshold']:.2f}, weights=({result['weight1']:.1f}, {result['weight2']:.1f}): F1={result['f1_score']:.4f}")
    
    # Print best configuration
    print(f"\nBest configuration:")
    if best_params['is_ensemble']:
        print(f"Ensemble with IoU={best_params['iou_threshold']:.2f}, weights=({best_params['weight1']:.1f}, {best_params['weight2']:.1f})")
        print(f"F1 score: {best_f1:.4f}")
    else:
        print(f"Single model ({best_params['model']}) performs better than ensemble")
        print(f"F1 score: {best_f1:.4f}")
    
    # Plot results
    plot_ensemble_results(results, best_params, output_dir)
    
    # Generate final ensemble if it's better than single models
    if best_params['is_ensemble']:
        final_detections = ensemble_detections(
            filtered_detections1,
            filtered_detections2,
            iou_threshold=best_params['iou_threshold'],
            weight1=best_params['weight1'],
            weight2=best_params['weight2']
        )
    else:
        final_detections = filtered_detections1 if best_params['model'] == 'model1' else filtered_detections2
    
    # Save final detections
    final_detections_path = os.path.join(output_dir, "final_optimized_detections.json")
    with open(final_detections_path, 'w') as f:
        json.dump(final_detections, f)
    print(f"Saved final optimized detections to {final_detections_path}")
    
    return best_params, final_detections

def plot_ensemble_results(results, best_params, output_dir):
    """
    Plot ensemble results as heatmaps
    """
    # Group results by weights to create a 2D grid of IoU vs F1
    weight_groups = {}
    for result in results:
        key = (result['weight1'], result['weight2'])
        if key not in weight_groups:
            weight_groups[key] = []
        weight_groups[key].append(result)
    
    # Create a figure for each weight combination
    for weights, group_results in weight_groups.items():
        weight1, weight2 = weights
        
        # Sort by IoU threshold
        group_results.sort(key=lambda x: x['iou_threshold'])
        
        # Extract data for plotting
        iou_values = [r['iou_threshold'] for r in group_results]
        f1_scores = [r['f1_score'] for r in group_results]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        plt.plot(iou_values, f1_scores, 'o-', linewidth=2)
        
        # Highlight best point if it matches current weights
        if best_params['is_ensemble'] and best_params['weight1'] == weight1 and best_params['weight2'] == weight2:
            best_idx = iou_values.index(best_params['iou_threshold'])
            plt.scatter([iou_values[best_idx]], [f1_scores[best_idx]], 
                        color='red', s=100, zorder=5, 
                        label=f"Best: IoU={best_params['iou_threshold']:.2f}, F1={f1_scores[best_idx]:.4f}")
        
        plt.title(f"Ensemble Performance with Weights ({weight1:.1f}, {weight2:.1f})")
        plt.xlabel("IoU Threshold")
        plt.ylabel("F1 Score")
        plt.grid(True)
        plt.legend()
        
        # Save figure
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"ensemble_iou_w1_{weight1:.1f}_w2_{weight2:.1f}.png")
        plt.savefig(plot_path)
        plt.close()
    
    # Create a combined plot with all weight combinations
    plt.figure(figsize=(12, 8))
    
    for weights, group_results in weight_groups.items():
        weight1, weight2 = weights
        
        # Sort by IoU threshold
        group_results.sort(key=lambda x: x['iou_threshold'])
        
        # Extract data for plotting
        iou_values = [r['iou_threshold'] for r in group_results]
        f1_scores = [r['f1_score'] for r in group_results]
        
        plt.plot(iou_values, f1_scores, 'o-', linewidth=2, 
                 label=f"Weights ({weight1:.1f}, {weight2:.1f})")
    
    # Highlight overall best point
    if best_params['is_ensemble']:
        for result in results:
            if (result['iou_threshold'] == best_params['iou_threshold'] and 
                result['weight1'] == best_params['weight1'] and 
                result['weight2'] == best_params['weight2']):
                plt.scatter([result['iou_threshold']], [result['f1_score']], 
                            color='red', s=150, zorder=10, 
                            label=f"Best: IoU={result['iou_threshold']:.2f}, W=({result['weight1']:.1f}, {result['weight2']:.1f}), F1={result['f1_score']:.4f}")
                break
    
    plt.title("Ensemble Performance Across All Weight Combinations")
    plt.xlabel("IoU Threshold")
    plt.ylabel("F1 Score")
    plt.grid(True)
    plt.legend()
    
    # Save figure
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "ensemble_all_weights.png")
    plt.savefig(plot_path)
    plt.close()

# --- Main Script ---
def main(gt_json_file_path, pred_json_file_path1, pred_json_file_path2, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # Create a temporary directory for intermediate files
    main_temp_dir = tempfile.mkdtemp(prefix="f1_opt_temp_")
    print(f"Using temporary directory for intermediate files: {main_temp_dir}")

    # 1. Load original GT
    with open(gt_json_file_path, 'r') as f:
        original_gt_data = json.load(f)
    
    coco_gt_original_obj = COCO(gt_json_file_path)

    # 2. Load detections from both models
    with open(pred_json_file_path1, 'r') as f:
        detections_model1 = json.load(f)
    
    with open(pred_json_file_path2, 'r') as f:
        detections_model2 = json.load(f)
    
    print(f"Loaded {len(detections_model1)} detections from model1")
    print(f"Loaded {len(detections_model2)} detections from model2")

    # 3. Optimize confidence thresholds for both models
    model1_results = optimize_confidence_thresholds(
        gt_json_file_path, 
        pred_json_file_path1, 
        os.path.join(output_dir, "model1"), 
        "model1",
        main_temp_dir
    )
    
    model2_results = optimize_confidence_thresholds(
        gt_json_file_path, 
        pred_json_file_path2, 
        os.path.join(output_dir, "model2"), 
        "model2",
        main_temp_dir
    )
    
    # 4. Print summary of optimized confidence thresholds
    summary_lines = ["=== Optimized Confidence Thresholds Summary ==="]
    
    for model_name, results in [("Model 1", model1_results), ("Model 2", model2_results)]:
        summary_lines.append(f"\n{model_name}:")
        optimal_thresholds = results['optimal_thresholds']
        category_names = results['category_names']
        
        for tod_label in ['day', 'night']:
            summary_lines.append(f"  {tod_label.capitalize()}:")
            if not optimal_thresholds.get(tod_label):
                summary_lines.append("    No thresholds optimized for this time of day.")
                continue
                
            for cat_id, thresh in optimal_thresholds[tod_label].items():
                cat_name = category_names.get(cat_id, f"Unknown Category {cat_id}")
                summary_lines.append(f"    Class '{cat_name}' (ID: {cat_id}): {thresh:.2f}")
    
    # Write summary to file
    summary_path = os.path.join(output_dir, "confidence_thresholds_summary.txt")
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    print(f"Saved confidence thresholds summary to {summary_path}")
    
    # 5. Optimize IoU threshold and weights for ensemble
    ensemble_results = optimize_iou_threshold_for_ensemble(
        coco_gt_original_obj,
        detections_model1,
        detections_model2,
        model1_results['day_image_ids'],
        model1_results['night_image_ids'],
        model1_results['optimal_thresholds'],
        model2_results['optimal_thresholds'],
        model1_results['day_image_ids_set'],
        model1_results['night_image_ids_set'],
        main_temp_dir,
        output_dir
    )
    
    best_params, final_detections = ensemble_results
    
    # 6. Save final configuration and results
    config = {
        "model1": {
            "confidence_thresholds": model1_results['optimal_thresholds']
        },
        "model2": {
            "confidence_thresholds": model2_results['optimal_thresholds']
        },
        "ensemble": best_params
    }
    
    config_path = os.path.join(output_dir, "final_optimization_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved final configuration to {config_path}")
    
    # Clean up temporary directory
    try:
        shutil.rmtree(main_temp_dir)
        print(f"Cleaned up temporary directory: {main_temp_dir}")
    except Exception as e:
        print(f"Error cleaning up temp directory {main_temp_dir}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize confidence and IoU thresholds for ensemble of two detection models.")
    parser.add_argument("-g", "--gt_file", required=True, help="Path to the COCO format ground truth JSON file.")
    parser.add_argument("-p1","--pred_file1", required=True, help="Path to the first model's COCO format predictions JSON file.")
    parser.add_argument("-p2","--pred_file2", required=True, help="Path to the second model's COCO format predictions JSON file.")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to save plots and summary.")
    
    args = parser.parse_args()

    main(args.gt_file, args.pred_file1, args.pred_file2, args.output_dir)