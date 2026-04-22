import json
import os
import numpy as np
import matplotlib.pyplot as plt
# Standard COCO import
from pycocotools.coco import COCO
# Import your modified COCOeval from the pycocotools package
from pycocotools.cocoeval_modified import COCOeval # <--- THIS IS THE KEY CHANGE
import shutil
import tempfile
import copy
import argparse # For command-line arguments

# --- Helper Functions --- (Same as before, no changes needed here)

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
                                    # Check your cocoeval_modified.py for the correct F1 index if you changed it.
                                    # Standard COCOeval has mAP, AP50, AP75, etc.
                                    # F1 is often derived from Precision/Recall arrays.
                                    # If coco_eval.stats[20] is your custom F1, this is fine.
                                    # Otherwise, you might need to calculate F1 from precision/recall arrays in coco_eval.eval
    return f1_score_val if f1_score_val != -1 else 0.0


# --- Main Script ---
def main(gt_json_file_path, pred_json_file_path, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # Create a temporary directory for intermediate files (still useful)
    main_temp_dir = tempfile.mkdtemp(prefix="f1_opt_temp_")
    print(f"Using main temporary directory for intermediate files: {main_temp_dir}")

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

    threshold_range = np.arange(0.2, 0.7, 0.05) # Start from 0.2 with 0.01 steps

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
                                if d['category_id'] == cat_id and d['image_id'] in current_tod_image_ids] # Ensure image_id is in current_tod_image_ids
            
            if not cat_tod_detections:
                print(f"    No detections for class {cat_name} in {tod_label} images. Setting optimal threshold to 0.05, F1=0.")
                optimal_thresholds[tod_label][cat_id] = 0.05 # Default low threshold
                all_f1_curves[tod_label][cat_id] = (threshold_range, [0.0] * len(threshold_range))
                continue

            # Create a GT object specific to this ToD and this single category
            single_cat_tod_gt_dict = save_coco_subset_annotations(original_gt_data, current_tod_image_ids, category_id_subset=cat_id)
            temp_single_cat_tod_gt_path = os.path.join(main_temp_dir, f"temp_gt_{tod_label}_cat{cat_id}.json")
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
                    # np.argmin(np.abs(thresholds - best_thresh_for_cat)) might not be robust if thresholds is not perfectly aligned
                    # It's better to use the stored best F1 if available or re-find it.
                    # The best_f1_idx from optimization is what we need, or find it again from f1_values.
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


        fig.suptitle(f"F1 Score Optimization ({tod_label.capitalize()})", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plot_filename = os.path.join(output_dir, f"f1_curves_{tod_label}.png")
        plt.savefig(plot_filename)
        print(f"Saved F1 curves plot to {plot_filename}")
        plt.close(fig)

    # 5. Print Optimal Thresholds and save to a file in output_dir
    summary_lines = ["--- Optimal Confidence Thresholds ---"]
    print("\n" + summary_lines[0])
    for tod_label in ['day', 'night']:
        line1 = f"  {tod_label.capitalize()}:"
        summary_lines.append(line1)
        print(line1)
        if not optimal_thresholds.get(tod_label): # Use .get for safety
            line2 = "    No thresholds optimized (e.g., no images for this ToD)."
            summary_lines.append(line2)
            print(line2)
            continue
        if not optimal_thresholds[tod_label]: # Check if the dict is empty
            line2 = "    No categories optimized for this ToD."
            summary_lines.append(line2)
            print(line2)
            continue
        for cat_id, thresh in optimal_thresholds[tod_label].items():
            line3 = f"    Class '{category_names.get(cat_id, f'Unknown Cat {cat_id}')}' (ID: {cat_id}): {thresh:.2f}"
            summary_lines.append(line3)
            print(line3)
    
    # 6. Evaluate with Optimal Thresholds (Overall F1 for Day/Night separately)
    summary_lines.append("\n--- Final Evaluation with Optimal Thresholds (Per ToD) ---")
    print("\n" + summary_lines[-1].split('\n')[-1]) # Print the title
    for tod_label, current_gt_coco_obj, current_tod_image_ids in [
        ('day', coco_gt_day_obj, day_image_ids),
        ('night', coco_gt_night_obj, night_image_ids)
    ]:
        if not current_gt_coco_obj or not current_tod_image_ids or not optimal_thresholds.get(tod_label):
            line = f"Skipping final evaluation for {tod_label} (no images, no GT, or no optimal thresholds for this ToD)."
            summary_lines.append("  " + line)
            print("  " + line)
            continue
        
        line_eval = f"  Evaluating {tod_label.upper()}:"
        summary_lines.append(line_eval)
        print(line_eval)
        
        final_filtered_detections = []
        # Detections relevant to the current ToD's images
        tod_specific_detections = [d for d in all_detections_data if d['image_id'] in current_tod_image_ids]

        for det in tod_specific_detections:
            cat_id = det['category_id']
            # Use the specific ToD's optimal thresholds for this category.
            # If category not in optimal_thresholds[tod_label] (e.g., class not present in GT for this ToD, so not optimized), use fallback.
            threshold_for_class = optimal_thresholds[tod_label].get(cat_id, DEFAULT_FALLBACK_THRESHOLD)
            if det['score'] >= threshold_for_class:
                final_filtered_detections.append(det)
        
        overall_f1 = get_f1_score(
            gt_coco_obj=current_gt_coco_obj, # GT object for this specific ToD
            dt_detections_list=final_filtered_detections,
            temp_dir_for_files=main_temp_dir,
            eval_params_template=base_eval_params,
            specific_cat_id_to_eval=None, # Overall F1 for all categories in this ToD GT
            img_ids_to_eval_explicitly=current_tod_image_ids # Images for this ToD
        )
        line_f1 = f"    Overall F1 Score for {tod_label.upper()} with optimal thresholds: {overall_f1:.4f}"
        summary_lines.append(line_f1)
        print(line_f1)

    # --- 7. Evaluate ALL IMAGES with ToD-Specific Optimal Thresholds ---
    summary_lines.append("\n--- Final Evaluation for ALL IMAGES with ToD-Specific Optimal Thresholds ---")
    print("\n" + summary_lines[-1].split('\n')[-1])

    final_filtered_detections_all_images = []
    # missing_threshold_info_logged = set() # Uncomment for debugging threshold choices

    for det in all_detections_data:
        img_id = det['image_id']
        cat_id = det['category_id']
        
        current_threshold_for_det = DEFAULT_FALLBACK_THRESHOLD
        tod_label_for_image = None

        if img_id in day_image_ids_set:
            tod_label_for_image = 'day'
        elif img_id in night_image_ids_set:
            tod_label_for_image = 'night'
        # else: image_id from detection is not in our segregated day/night sets.
        # This could happen if detections include images not in the GT, or GT images that didn't fit day/night criteria.
        # In such cases, DEFAULT_FALLBACK_THRESHOLD will be used.

        if tod_label_for_image:
            # Check if optimal thresholds were computed for this ToD
            if tod_label_for_image in optimal_thresholds and optimal_thresholds[tod_label_for_image]:
                cat_specific_thresh = optimal_thresholds[tod_label_for_image].get(cat_id)
                if cat_specific_thresh is not None:
                    current_threshold_for_det = cat_specific_thresh
                # else: Category was not optimized for this ToD (e.g. "person" in day, but not night GT)
                #       -> current_threshold_for_det remains DEFAULT_FALLBACK_THRESHOLD
                #       log_key = (tod_label_for_image, cat_id, 'cat_not_optimized_for_tod')
                #       if log_key not in missing_threshold_info_logged:
                #           print(f"Info: Cat {cat_id} not optimized for ToD '{tod_label_for_image}'. Using fallback {DEFAULT_FALLBACK_THRESHOLD} for img {img_id}.")
                #           missing_threshold_info_logged.add(log_key)
            # else: No thresholds were optimized for this ToD at all (e.g. no night images)
            #       -> current_threshold_for_det remains DEFAULT_FALLBACK_THRESHOLD
            #       log_key = (tod_label_for_image, 'no_thresholds_for_tod')
            #       if log_key not in missing_threshold_info_logged:
            #           print(f"Info: No thresholds for ToD '{tod_label_for_image}'. Using fallback {DEFAULT_FALLBACK_THRESHOLD} for img {img_id}, cat {cat_id}.")
            #           missing_threshold_info_logged.add(log_key)
        # else: Image not classified as day or night.
        #       -> current_threshold_for_det remains DEFAULT_FALLBACK_THRESHOLD
        #       log_key = (img_id, 'image_not_in_tod_sets')
        #       if log_key not in missing_threshold_info_logged:
        #           print(f"Info: Img {img_id} (cat {cat_id}) not in day/night sets. Using fallback {DEFAULT_FALLBACK_THRESHOLD}.")
        #           missing_threshold_info_logged.add(log_key)

        if det['score'] >= current_threshold_for_det:
            final_filtered_detections_all_images.append(det)
    
    if not coco_gt_original_obj: # Should exist if script ran this far
        line = "Skipping final evaluation for ALL IMAGES (original GT COCO object not available)."
        summary_lines.append("  " + line)
        print("  " + line)
    # No need to check final_filtered_detections_all_images here, get_f1_score handles empty list
    else:
        overall_f1_all_images = get_f1_score(
            gt_coco_obj=coco_gt_original_obj, # Use the original full GT COCO object
            dt_detections_list=final_filtered_detections_all_images,
            temp_dir_for_files=main_temp_dir,
            eval_params_template=base_eval_params,
            specific_cat_id_to_eval=None, # Overall F1 for all categories in the original GT
            img_ids_to_eval_explicitly=all_image_ids_original_gt # Evaluate on all image IDs from the original GT
        )
        line_f1_all = f"    Overall F1 Score for ALL IMAGES (using ToD-specific optimal thresholds): {overall_f1_all_images:.4f}"
        summary_lines.append(line_f1_all)
        print(line_f1_all)


    summary_filename = os.path.join(output_dir, "optimal_thresholds_summary.txt")
    with open(summary_filename, 'w') as f:
        f.write("\n".join(summary_lines))
    print(f"Saved summary to {summary_filename}")

    # Clean up temporary directory
    try:
        shutil.rmtree(main_temp_dir)
        print(f"Cleaned up temporary directory for intermediate files: {main_temp_dir}")
    except Exception as e:
        print(f"Error cleaning up temp directory {main_temp_dir}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize F1 scores by finding optimal confidence thresholds per class for day/night images.")
    parser.add_argument("-g", "--gt_file", required=True, help="Path to the COCO format ground truth JSON file.")
    parser.add_argument("-p","--pred_file", required=True, help="Path to the COCO format predictions JSON file.")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to save plots and summary.")
    
    args = parser.parse_args()

    main(args.gt_file, args.pred_file, args.output_dir)