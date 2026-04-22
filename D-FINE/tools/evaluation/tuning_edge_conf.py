import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval_modified import COCOeval
import shutil
import tempfile
import copy
import argparse

# --- Helper Functions (Same as before) ---

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


def get_f1_score(gt_coco_obj, dt_detections_list, temp_dir_for_files, eval_params_template, 
                specific_cat_id_to_eval=None, img_ids_to_eval_explicitly=None):
    """
    Calculates F1 score using COCOeval.
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
            return 0.0
        coco_eval.params.catIds = [specific_cat_id_to_eval]
    else: # Evaluate all categories present in gt_coco_obj
        coco_eval.params.catIds = sorted(list(c['id'] for c in gt_coco_obj.dataset['categories']))
        if not coco_eval.params.catIds: return 0.0

    if not coco_eval.params.imgIds or not coco_eval.params.catIds:
        return 0.0

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    f1_score_val = coco_eval.stats[20]
    return f1_score_val if f1_score_val != -1 else 0.0


# --- New Function to Calculate Normalized Distance from Center ---
def calculate_normalized_distance(bbox, img_width, img_height):
    """
    Calculate the normalized distance of a bounding box center from the image center.
    
    Args:
        bbox: [x, y, width, height] - COCO format bounding box
        img_width: Width of the image
        img_height: Height of the image
        
    Returns:
        Normalized distance (0 to 1) from the center of the image
    """
    # Calculate bbox center
    bbox_center_x = bbox[0] + bbox[2]/2
    bbox_center_y = bbox[1] + bbox[3]/2
    
    # Calculate image center
    img_center_x = img_width/2
    img_center_y = img_height/2
    
    # Calculate Euclidean distance
    distance = np.sqrt((bbox_center_x - img_center_x)**2 + (bbox_center_y - img_center_y)**2)
    
    # Normalize by the maximum possible distance (corner to center)
    max_distance = np.sqrt((img_width/2)**2 + (img_height/2)**2)
    
    return min(distance / max_distance, 1.0)  # Cap at 1.0 to be safe


# --- Modified Main Function ---
def main(gt_json_file_path, pred_json_file_path, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # Create a temporary directory for intermediate files
    main_temp_dir = tempfile.mkdtemp(prefix="f1_opt_temp_")
    print(f"Using main temporary directory for intermediate files: {main_temp_dir}")

    # 1. Load original GT and Detections
    with open(gt_json_file_path, 'r') as f:
        original_gt_data = json.load(f)
    
    coco_gt_original_obj = COCO(gt_json_file_path)

    with open(pred_json_file_path, 'r') as f:
        all_detections_data = json.load(f)

    # Build image id to dimensions map
    image_dimensions = {}
    for img in original_gt_data['images']:
        image_dimensions[img['id']] = (img['width'], img['height'])

    # 2. Segregate Image IDs into Day and Night
    all_image_ids_original_gt = coco_gt_original_obj.getImgIds()
    day_image_ids = []
    night_image_ids = []
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
        temp_gt_day_path = os.path.join(main_temp_dir, "temp_gt_day.json")
        with open(temp_gt_day_path, 'w') as f: json.dump(day_gt_dict, f)
        coco_gt_day_obj = COCO(temp_gt_day_path)
    if night_image_ids:
        night_gt_dict = save_coco_subset_annotations(original_gt_data, night_image_ids)
        temp_gt_night_path = os.path.join(main_temp_dir, "temp_gt_night.json")
        with open(temp_gt_night_path, 'w') as f: json.dump(night_gt_dict, f)
        coco_gt_night_obj = COCO(temp_gt_night_path)

    all_category_ids = sorted(coco_gt_original_obj.getCatIds())
    category_names = {cat['id']: cat['name'] for cat in coco_gt_original_obj.loadCats(all_category_ids)}

    threshold_range = np.arange(0.2, 0.9, 0.025)
    DEFAULT_FALLBACK_THRESHOLD = 0.5

    base_eval_params = COCOeval(iouType='bbox').params
    
    # --- First Phase: Optimize basic thresholds without distance factor ---
    print("\n=== PHASE 1: Optimizing basic thresholds without distance factor ===")
    
    basic_optimal_thresholds = {'day': {}, 'night': {}}
    all_f1_curves = {'day': {}, 'night': {}}

    # 3. Optimize Thresholds (First phase - basic thresholds)
    for tod_label, current_gt_coco_obj, current_tod_image_ids in [
        ('day', coco_gt_day_obj, day_image_ids),
        ('night', coco_gt_night_obj, night_image_ids)
    ]:
        if not current_gt_coco_obj or not current_tod_image_ids:
            print(f"Skipping {tod_label} as there are no images or GT object.")
            basic_optimal_thresholds[tod_label] = {}
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
            cat_tod_detections = [d for d in all_detections_data 
                                if d['category_id'] == cat_id and d['image_id'] in current_tod_image_ids]
            
            if not cat_tod_detections:
                print(f"    No detections for class {cat_name} in {tod_label} images. Setting optimal threshold to 0.05, F1=0.")
                basic_optimal_thresholds[tod_label][cat_id] = 0.05
                all_f1_curves[tod_label][cat_id] = (threshold_range, [0.0] * len(threshold_range))
                continue

            # Create a GT object specific to this ToD and this single category
            single_cat_tod_gt_dict = save_coco_subset_annotations(original_gt_data, current_tod_image_ids, category_id_subset=cat_id)
            temp_single_cat_tod_gt_path = os.path.join(main_temp_dir, f"temp_gt_{tod_label}_cat{cat_id}.json")
            with open(temp_single_cat_tod_gt_path, 'w') as f: json.dump(single_cat_tod_gt_dict, f)
            coco_gt_single_cat_tod_obj = COCO(temp_single_cat_tod_gt_path)
            
            if not coco_gt_single_cat_tod_obj.getAnnIds(catIds=[cat_id]):
                print(f"    No GT annotations for class {cat_name} in {tod_label} images. Setting optimal threshold to 0.05, F1=0.")
                basic_optimal_thresholds[tod_label][cat_id] = 0.05
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
                    img_ids_to_eval_explicitly=current_tod_image_ids
                )
                f1_scores_for_cat.append(f1)

            if not f1_scores_for_cat or max(f1_scores_for_cat) == 0.0:
                basic_optimal_thresholds[tod_label][cat_id] = threshold_range[np.argmax(f1_scores_for_cat)] if f1_scores_for_cat else 0.05
                all_f1_curves[tod_label][cat_id] = (threshold_range, f1_scores_for_cat if f1_scores_for_cat else [0.0] * len(threshold_range))
                print(f"    No positive F1 scores for class {cat_name} in {tod_label}. Optimal threshold set to {basic_optimal_thresholds[tod_label][cat_id]:.2f} (F1: {max(f1_scores_for_cat) if f1_scores_for_cat else 0.0:.3f}).")
                continue

            best_f1_idx = np.argmax(f1_scores_for_cat)
            basic_optimal_thresholds[tod_label][cat_id] = threshold_range[best_f1_idx]
            all_f1_curves[tod_label][cat_id] = (threshold_range, f1_scores_for_cat)
            print(f"    Optimal threshold for {cat_name} ({tod_label}): {basic_optimal_thresholds[tod_label][cat_id]:.2f} (F1: {f1_scores_for_cat[best_f1_idx]:.3f})")

    # 4. Plot F1 curves and save to output_dir for Phase 1
    for tod_label in ['day', 'night']:
        if not all_f1_curves.get(tod_label): continue
        
        plotted_curves = {k: v for k, v in all_f1_curves[tod_label].items() if v[1]}
        num_cats_to_plot = len(plotted_curves)
        if num_cats_to_plot == 0: continue

        cols = 2
        rows = (num_cats_to_plot + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)
        axes_flat = axes.flatten()
        
        plot_idx = 0
        for cat_id, (thresholds, f1_values) in plotted_curves.items():
            if not f1_values: continue
            if plot_idx < len(axes_flat):
                ax = axes_flat[plot_idx]
                cat_name = category_names.get(cat_id, f"Unknown Cat {cat_id}")
                ax.plot(thresholds, f1_values, marker='o', linestyle='-')
                
                best_thresh_for_cat = basic_optimal_thresholds[tod_label].get(cat_id)
                if best_thresh_for_cat is not None:
                    best_f1_val_for_plot = 0.0
                    try:
                        thresh_idx_for_plot = list(thresholds).index(best_thresh_for_cat)
                        best_f1_val_for_plot = f1_values[thresh_idx_for_plot]
                    except ValueError:
                        best_f1_val_for_plot = max(f1_values) if f1_values else 0.0

                    ax.scatter([best_thresh_for_cat], [best_f1_val_for_plot], color='red', s=100, zorder=5, label=f"Optimal: {best_thresh_for_cat:.2f} (F1: {best_f1_val_for_plot:.3f})")
                
                ax.set_title(f"F1 for {cat_name} ({tod_label.capitalize()})")
                ax.set_xlabel("Confidence Threshold")
                ax.set_ylabel("F1 Score (for this class)")
                ax.set_ylim(0, 1.05)
                ax.grid(True)
                ax.legend()
                plot_idx += 1

        for i in range(plot_idx, len(axes_flat)):
            if i < len(axes_flat): fig.delaxes(axes_flat[i])

        fig.suptitle(f"F1 Score Optimization - Phase 1 ({tod_label.capitalize()})", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plot_filename = os.path.join(output_dir, f"phase1_f1_curves_{tod_label}.png")
        plt.savefig(plot_filename)
        print(f"Saved Phase 1 F1 curves plot to {plot_filename}")
        plt.close(fig)
    
    # --- Print Phase 1 results ---
    summary_lines = ["=== PHASE 1: Basic Optimal Thresholds (Without Distance Factor) ==="]
    print("\n" + summary_lines[0])
    for tod_label in ['day', 'night']:
        line1 = f"  {tod_label.capitalize()}:"
        summary_lines.append(line1)
        print(line1)
        if not basic_optimal_thresholds.get(tod_label):
            line2 = "    No thresholds optimized (e.g., no images for this ToD)."
            summary_lines.append(line2)
            print(line2)
            continue
        if not basic_optimal_thresholds[tod_label]:
            line2 = "    No categories optimized for this ToD."
            summary_lines.append(line2)
            print(line2)
            continue
        for cat_id, thresh in basic_optimal_thresholds[tod_label].items():
            line3 = f"    Class '{category_names.get(cat_id, f'Unknown Cat {cat_id}')}' (ID: {cat_id}): {thresh:.2f}"
            summary_lines.append(line3)
            print(line3)
    
    # Evaluate with Phase 1 Optimal Thresholds
    summary_lines.append("\n--- Phase 1 Evaluation with Basic Optimal Thresholds (Per ToD) ---")
    print("\n" + summary_lines[-1].split('\n')[-1])
    phase1_f1_tod = {}
    for tod_label, current_gt_coco_obj, current_tod_image_ids in [
        ('day', coco_gt_day_obj, day_image_ids),
        ('night', coco_gt_night_obj, night_image_ids)
    ]:
        if not current_gt_coco_obj or not current_tod_image_ids or not basic_optimal_thresholds.get(tod_label):
            line = f"Skipping Phase 1 evaluation for {tod_label} (no images, no GT, or no optimal thresholds for this ToD)."
            summary_lines.append("  " + line)
            print("  " + line)
            phase1_f1_tod[tod_label] = 0.0
            continue
        
        line_eval = f"  Evaluating {tod_label.upper()}:"
        summary_lines.append(line_eval)
        print(line_eval)
        
        final_filtered_detections = []
        tod_specific_detections = [d for d in all_detections_data if d['image_id'] in current_tod_image_ids]

        for det in tod_specific_detections:
            cat_id = det['category_id']
            threshold_for_class = basic_optimal_thresholds[tod_label].get(cat_id, DEFAULT_FALLBACK_THRESHOLD)
            if det['score'] >= threshold_for_class:
                final_filtered_detections.append(det)
        
        overall_f1 = get_f1_score(
            gt_coco_obj=current_gt_coco_obj,
            dt_detections_list=final_filtered_detections,
            temp_dir_for_files=main_temp_dir,
            eval_params_template=base_eval_params,
            specific_cat_id_to_eval=None,
            img_ids_to_eval_explicitly=current_tod_image_ids
        )
        phase1_f1_tod[tod_label] = overall_f1
        line_f1 = f"    Overall F1 Score for {tod_label.upper()} with Phase 1 optimal thresholds: {overall_f1:.4f}"
        summary_lines.append(line_f1)
        print(line_f1)

    # Evaluate ALL IMAGES with Phase 1 Optimal Thresholds
    summary_lines.append("\n--- Phase 1 Evaluation for ALL IMAGES with ToD-Specific Optimal Thresholds ---")
    print("\n" + summary_lines[-1].split('\n')[-1])

    final_filtered_detections_all_images = []

    for det in all_detections_data:
        img_id = det['image_id']
        cat_id = det['category_id']
        
        current_threshold_for_det = DEFAULT_FALLBACK_THRESHOLD
        tod_label_for_image = None

        if img_id in day_image_ids_set:
            tod_label_for_image = 'day'
        elif img_id in night_image_ids_set:
            tod_label_for_image = 'night'

        if tod_label_for_image:
            if tod_label_for_image in basic_optimal_thresholds and basic_optimal_thresholds[tod_label_for_image]:
                cat_specific_thresh = basic_optimal_thresholds[tod_label_for_image].get(cat_id)
                if cat_specific_thresh is not None:
                    current_threshold_for_det = cat_specific_thresh

        if det['score'] >= current_threshold_for_det:
            final_filtered_detections_all_images.append(det)
    
    if not coco_gt_original_obj:
        line = "Skipping Phase 1 final evaluation for ALL IMAGES (original GT COCO object not available)."
        summary_lines.append("  " + line)
        print("  " + line)
        phase1_f1_all = 0.0
    else:
        phase1_f1_all = get_f1_score(
            gt_coco_obj=coco_gt_original_obj,
            dt_detections_list=final_filtered_detections_all_images,
            temp_dir_for_files=main_temp_dir,
            eval_params_template=base_eval_params,
            specific_cat_id_to_eval=None,
            img_ids_to_eval_explicitly=all_image_ids_original_gt
        )
        line_f1_all = f"    Overall F1 Score for ALL IMAGES (using Phase 1 ToD-specific optimal thresholds): {phase1_f1_all:.4f}"
        summary_lines.append(line_f1_all)
        print(line_f1_all)
    
    # --- Phase 2: Optimize with distance factor ---
    print("\n=== PHASE 2: Optimizing thresholds with distance factor ===")
    
    # Define b values to test
    b_values = np.arange(0.1, 1.0, 0.1)
    
    # For each b value, we'll find the optimal threshold per category
    distance_optimal_results = {'day': {}, 'night': {}}
    
    # Store F1 results for each b value to compare later
    b_f1_results = {'day': {}, 'night': {}, 'all': {}}
    
    for b in b_values:
        print(f"\n--- Testing distance factor b = {b:.1f} ---")
        
        # Store optimal thresholds for this b value
        b_optimal_thresholds = {'day': {}, 'night': {}}
        
        # For each ToD (day/night)
        for tod_label, current_gt_coco_obj, current_tod_image_ids in [
            ('day', coco_gt_day_obj, day_image_ids),
            ('night', coco_gt_night_obj, night_image_ids)
        ]:
            if not current_gt_coco_obj or not current_tod_image_ids:
                print(f"Skipping {tod_label} for b={b:.1f} as there are no images or GT object.")
                b_optimal_thresholds[tod_label] = {}
                continue
            
            print(f"  Optimizing for {tod_label.upper()} with b={b:.1f}")
            
            # Get categories for this ToD
            categories_in_tod_gt = current_gt_coco_obj.loadCats(current_gt_coco_obj.getCatIds())
            
            # For each category
            for cat_info in categories_in_tod_gt:
                cat_id = cat_info['id']
                cat_name = category_names[cat_id]
                print(f"    Optimizing for class: {cat_name} (ID: {cat_id})")
                
                # Get detections for this category in this ToD
                cat_tod_detections = [d for d in all_detections_data 
                                     if d['category_id'] == cat_id and d['image_id'] in current_tod_image_ids]
                
                if not cat_tod_detections:
                    print(f"      No detections for class {cat_name} in {tod_label} images with b={b:.1f}. Setting optimal threshold to 0.05, F1=0.")
                    b_optimal_thresholds[tod_label][cat_id] = 0.05
                    continue
                
                # Create GT object for this category and ToD
                single_cat_tod_gt_dict = save_coco_subset_annotations(original_gt_data, current_tod_image_ids, category_id_subset=cat_id)
                temp_single_cat_tod_gt_path = os.path.join(main_temp_dir, f"temp_gt_{tod_label}_cat{cat_id}_b{b}.json")
                with open(temp_single_cat_tod_gt_path, 'w') as f: json.dump(single_cat_tod_gt_dict, f)
                coco_gt_single_cat_tod_obj = COCO(temp_single_cat_tod_gt_path)
                
                if not coco_gt_single_cat_tod_obj.getAnnIds(catIds=[cat_id]):
                    print(f"      No GT annotations for class {cat_name} in {tod_label} images with b={b:.1f}. Setting optimal threshold to 0.05, F1=0.")
                    b_optimal_thresholds[tod_label][cat_id] = 0.05
                    continue
                
                # Try different thresholds
                f1_scores_for_cat = []
                for conf_thresh in threshold_range:
                    # Apply distance-based confidence adjustment to each detection
                    adjusted_detections = []
                    for det in cat_tod_detections:
                        # Get image dimensions
                        if det['image_id'] in image_dimensions:
                            img_width, img_height = image_dimensions[det['image_id']]
                            
                            # Calculate normalized distance from center
                            norm_distance = calculate_normalized_distance(det['bbox'], img_width, img_height)
                            
                            # Apply the distance factor to adjust confidence
                            adjusted_confidence = det['score'] * (1 - b * norm_distance)
                            
                            # Check if detection passes the current threshold
                            if adjusted_confidence >= conf_thresh:
                                # Create a copy of the detection
                                adjusted_det = det.copy()
                                # Use the original score for evaluation
                                # (COCO eval uses the score for sorting detections, not for filtering)
                                adjusted_detections.append(adjusted_det)
                        else:
                            # If we don't have image dimensions, use the original detection
                            if det['score'] >= conf_thresh:
                                adjusted_detections.append(det)
                    
                    # Evaluate with adjusted detections
                    f1 = get_f1_score(
                        gt_coco_obj=coco_gt_single_cat_tod_obj, 
                        dt_detections_list=adjusted_detections,
                        temp_dir_for_files=main_temp_dir,
                        eval_params_template=base_eval_params,
                        specific_cat_id_to_eval=cat_id,
                        img_ids_to_eval_explicitly=current_tod_image_ids
                    )
                    f1_scores_for_cat.append(f1)
                
                if not f1_scores_for_cat or max(f1_scores_for_cat) == 0.0:
                    b_optimal_thresholds[tod_label][cat_id] = threshold_range[np.argmax(f1_scores_for_cat)] if f1_scores_for_cat else 0.05
                    print(f"      No positive F1 scores for class {cat_name} in {tod_label} with b={b:.1f}. Optimal threshold set to {b_optimal_thresholds[tod_label][cat_id]:.2f} (F1: {max(f1_scores_for_cat) if f1_scores_for_cat else 0.0:.3f}).")
                    continue
                
                best_f1_idx = np.argmax(f1_scores_for_cat)
                b_optimal_thresholds[tod_label][cat_id] = threshold_range[best_f1_idx]
                print(f"      Optimal threshold for {cat_name} ({tod_label}) with b={b:.1f}: {b_optimal_thresholds[tod_label][cat_id]:.2f} (F1: {f1_scores_for_cat[best_f1_idx]:.3f})")
        
        # Evaluate with optimal thresholds for this b value
        b_f1_results['day'][b] = 0.0
        b_f1_results['night'][b] = 0.0
        
        # Evaluate day
        if coco_gt_day_obj and day_image_ids and b_optimal_thresholds.get('day'):
            day_detections = []
            for det in all_detections_data:
                if det['image_id'] in day_image_ids:
                    cat_id = det['category_id']
                    threshold_for_class = b_optimal_thresholds['day'].get(cat_id, DEFAULT_FALLBACK_THRESHOLD)
                    
                    # Apply distance-based confidence adjustment
                    if det['image_id'] in image_dimensions:
                        img_width, img_height = image_dimensions[det['image_id']]
                        norm_distance = calculate_normalized_distance(det['bbox'], img_width, img_height)
                        adjusted_confidence = det['score'] * (1 - b * norm_distance)
                        
                        if adjusted_confidence >= threshold_for_class:
                            day_detections.append(det)
                    else:
                        if det['score'] >= threshold_for_class:
                            day_detections.append(det)
            
            day_f1 = get_f1_score(
                gt_coco_obj=coco_gt_day_obj,
                dt_detections_list=day_detections,
                temp_dir_for_files=main_temp_dir,
                eval_params_template=base_eval_params,
                specific_cat_id_to_eval=None,
                img_ids_to_eval_explicitly=day_image_ids
            )
            b_f1_results['day'][b] = day_f1
            print(f"  Day F1 score with b={b:.1f}: {day_f1:.4f}")
        
        # Evaluate night
        if coco_gt_night_obj and night_image_ids and b_optimal_thresholds.get('night'):
            night_detections = []
            for det in all_detections_data:
                if det['image_id'] in night_image_ids:
                    cat_id = det['category_id']
                    threshold_for_class = b_optimal_thresholds['night'].get(cat_id, DEFAULT_FALLBACK_THRESHOLD)
                    
                    # Apply distance-based confidence adjustment
                    if det['image_id'] in image_dimensions:
                        img_width, img_height = image_dimensions[det['image_id']]
                        norm_distance = calculate_normalized_distance(det['bbox'], img_width, img_height)
                        adjusted_confidence = det['score'] * (1 - b * norm_distance)
                        
                        if adjusted_confidence >= threshold_for_class:
                            night_detections.append(det)
                    else:
                        if det['score'] >= threshold_for_class:
                            night_detections.append(det)
            
            night_f1 = get_f1_score(
                gt_coco_obj=coco_gt_night_obj,
                dt_detections_list=night_detections,
                temp_dir_for_files=main_temp_dir,
                eval_params_template=base_eval_params,
                specific_cat_id_to_eval=None,
                img_ids_to_eval_explicitly=night_image_ids
            )
            b_f1_results['night'][b] = night_f1
            print(f"  Night F1 score with b={b:.1f}: {night_f1:.4f}")
        
        # Evaluate all images
        if coco_gt_original_obj:
            all_detections = []
            for det in all_detections_data:
                tod_label = 'day' if det['image_id'] in day_image_ids_set else 'night' if det['image_id'] in night_image_ids_set else None
                
                if tod_label and b_optimal_thresholds.get(tod_label):
                    cat_id = det['category_id']
                    threshold_for_class = b_optimal_thresholds[tod_label].get(cat_id, DEFAULT_FALLBACK_THRESHOLD)
                    
                    # Apply distance-based confidence adjustment
                    if det['image_id'] in image_dimensions:
                        img_width, img_height = image_dimensions[det['image_id']]
                        norm_distance = calculate_normalized_distance(det['bbox'], img_width, img_height)
                        adjusted_confidence = det['score'] * (1 - b * norm_distance)
                        
                        if adjusted_confidence >= threshold_for_class:
                            all_detections.append(det)
                    else:
                        if det['score'] >= threshold_for_class:
                            all_detections.append(det)
                else:
                    # For images not classified as day/night
                    if det['score'] >= DEFAULT_FALLBACK_THRESHOLD:
                        all_detections.append(det)
            
            all_f1 = get_f1_score(
                gt_coco_obj=coco_gt_original_obj,
                dt_detections_list=all_detections,
                temp_dir_for_files=main_temp_dir,
                eval_params_template=base_eval_params,
                specific_cat_id_to_eval=None,
                img_ids_to_eval_explicitly=all_image_ids_original_gt
            )
            b_f1_results['all'][b] = all_f1
            print(f"  Overall F1 score with b={b:.1f}: {all_f1:.4f}")
        
        # Store the optimal thresholds for this b value
        distance_optimal_results['day'][b] = b_optimal_thresholds['day']
        distance_optimal_results['night'][b] = b_optimal_thresholds['night']
    
    # Find the best b value for each ToD and overall
    best_b = {'day': None, 'night': None, 'all': None}
    best_f1 = {'day': 0.0, 'night': 0.0, 'all': 0.0}
    
    for tod in ['day', 'night', 'all']:
        if b_f1_results[tod]:
            best_b[tod] = max(b_f1_results[tod].items(), key=lambda x: x[1])[0]
            best_f1[tod] = b_f1_results[tod][best_b[tod]]
    
    # Plot F1 scores for different b values
    plt.figure(figsize=(12, 8))
    for tod in ['day', 'night', 'all']:
        if b_f1_results[tod]:
            b_values_list = list(b_f1_results[tod].keys())
            f1_values_list = list(b_f1_results[tod].values())
            
            plt.plot(b_values_list, f1_values_list, marker='o', linestyle='-', label=f"{tod.capitalize()}")
            
            if best_b[tod] is not None:
                plt.scatter([best_b[tod]], [best_f1[tod]], color='red' if tod == 'all' else None, 
                           s=100, zorder=5, 
                           label=f"Best {tod.capitalize()}: b={best_b[tod]:.1f} (F1: {best_f1[tod]:.4f})")
    
    plt.title("F1 Score vs. Distance Factor (b)", fontsize=16)
    plt.xlabel("Distance Factor (b)")
    plt.ylabel("F1 Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    b_plot_filename = os.path.join(output_dir, "f1_vs_distance_factor.png")
    plt.savefig(b_plot_filename)
    print(f"Saved F1 vs. Distance Factor plot to {b_plot_filename}")
    plt.close()
    
    # Add Phase 2 results to summary
    summary_lines.append("\n=== PHASE 2: Optimizing with Distance Factor ===")
    summary_lines.append(f"Distance Factor Formula: adjusted_confidence = original_confidence * (1 - b * normalized_distance)")
    summary_lines.append("\nBest Distance Factors (b):")
    for tod in ['day', 'night', 'all']:
        if best_b[tod] is not None:
            summary_lines.append(f"  {tod.capitalize()}: b={best_b[tod]:.1f} (F1: {best_f1[tod]:.4f})")
        else:
            summary_lines.append(f"  {tod.capitalize()}: No optimal b found")
    
    # Compare with Phase 1 results
    summary_lines.append("\nComparison with Phase 1 (Basic Thresholds):")
    for tod in ['day', 'night']:
        if tod in phase1_f1_tod:
            p1_f1 = phase1_f1_tod[tod]
            p2_f1 = best_f1[tod] if best_b[tod] is not None else 0.0
            improvement = ((p2_f1 - p1_f1) / p1_f1 * 100) if p1_f1 > 0 else float('inf')
            summary_lines.append(f"  {tod.capitalize()}: Phase 1 F1={p1_f1:.4f}, Phase 2 F1={p2_f1:.4f} (Change: {improvement:.2f}%)")
    
    if phase1_f1_all > 0:
        p2_all_f1 = best_f1['all'] if best_b['all'] is not None else 0.0
        all_improvement = ((p2_all_f1 - phase1_f1_all) / phase1_f1_all * 100)
        summary_lines.append(f"  ALL: Phase 1 F1={phase1_f1_all:.4f}, Phase 2 F1={p2_all_f1:.4f} (Change: {all_improvement:.2f}%)")
    
    # Save detailed optimal thresholds for the best b value
    if best_b['all'] is not None:
        best_overall_b = best_b['all']
        summary_lines.append(f"\nDetailed Optimal Thresholds for Best Overall b={best_overall_b:.1f}:")
        
        for tod_label in ['day', 'night']:
            if best_overall_b in distance_optimal_results[tod_label]:
                summary_lines.append(f"  {tod_label.capitalize()}:")
                for cat_id, thresh in distance_optimal_results[tod_label][best_overall_b].items():
                    summary_lines.append(f"    Class '{category_names.get(cat_id, f'Unknown Cat {cat_id}')}' (ID: {cat_id}): {thresh:.2f}")
    
    # Write the final summary
    summary_filename = os.path.join(output_dir, "optimal_thresholds_with_distance_factor_summary.txt")
    with open(summary_filename, 'w') as f:
        f.write("\n".join(summary_lines))
    print(f"Saved final summary to {summary_filename}")
    
    # Clean up temporary directory
    try:
        shutil.rmtree(main_temp_dir)
        print(f"Cleaned up temporary directory for intermediate files: {main_temp_dir}")
    except Exception as e:
        print(f"Error cleaning up temp directory {main_temp_dir}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize F1 scores with confidence thresholds and distance factors for fisheye images.")
    parser.add_argument("-g", "--gt_file", required=True, help="Path to the COCO format ground truth JSON file.")
    parser.add_argument("-p","--pred_file", required=True, help="Path to the COCO format predictions JSON file.")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to save plots and summary.")
    
    args = parser.parse_args()

    main(args.gt_file, args.pred_file, args.output_dir)