#!/usr/bin/env python3
"""
Generate confusion matrix from COCO format object detection ground truth and prediction files.
Based on the reference implementation with proper IoU-based matching.
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import pandas as pd
import copy


def load_coco_annotations(file_path):
    """Load COCO format annotations from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def box_iou_calc(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Array[N, 4])
        boxes2 (Array[M, 4])
    Returns:
        iou (Array[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])
    
    area1 = box_area(boxes1.T)
    area2 = box_area(boxes2.T)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    inter = np.prod(np.clip(rb - lt, a_min=0, a_max=None), 2)
    return inter / (area1[:, None] + area2 - inter)


def coco_bbox_to_xyxy(bbox):
    """Convert COCO bbox format [x, y, w, h] to [x1, y1, x2, y2]"""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


class ConfusionMatrix:
    def __init__(self, num_classes, CONF_THRESHOLD=0.3, IOU_THRESHOLD=0.5):
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))
        self.num_classes = num_classes
        self.CONF_THRESHOLD = CONF_THRESHOLD
        self.IOU_THRESHOLD = IOU_THRESHOLD
    
    def process_batch(self, detections, labels):
        '''
        Process a batch of detections and labels to update confusion matrix.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        '''
        if len(detections) == 0 and len(labels) == 0:
            return
            
        # Filter detections by confidence threshold
        if len(detections) > 0:
            detections = detections[detections[:, 4] > self.CONF_THRESHOLD]
        
        if len(labels) == 0:
            # Only false positives
            for detection in detections:
                detection_class = int(detection[5])
                self.matrix[self.num_classes, detection_class] += 1
            return
            
        if len(detections) == 0:
            # Only false negatives
            for label in labels:
                gt_class = int(label[0])
                self.matrix[gt_class, self.num_classes] += 1
            return
        
        gt_classes = labels[:, 0].astype(np.int16)
        detection_classes = detections[:, 5].astype(np.int16)

        # Calculate IoU between all ground truth and detection boxes
        all_ious = box_iou_calc(labels[:, 1:], detections[:, :4])
        want_idx = np.where(all_ious > self.IOU_THRESHOLD)

        all_matches = []
        for i in range(want_idx[0].shape[0]):
            all_matches.append([want_idx[0][i], want_idx[1][i], all_ious[want_idx[0][i], want_idx[1][i]]])
        
        all_matches = np.array(all_matches)
        
        if all_matches.shape[0] > 0:  # if there are matches
            # Sort by IoU in descending order
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]
            
            # Remove duplicate detections (keep highest IoU)
            all_matches = all_matches[np.unique(all_matches[:, 1], return_index=True)[1]]
            
            # Sort by IoU again
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]
            
            # Remove duplicate ground truths (keep highest IoU)
            all_matches = all_matches[np.unique(all_matches[:, 0], return_index=True)[1]]

        # Process ground truth boxes
        for i, label in enumerate(labels):
            if all_matches.shape[0] > 0 and all_matches[all_matches[:, 0] == i].shape[0] == 1:
                # This ground truth has a match
                gt_class = gt_classes[i]
                detection_class = detection_classes[int(all_matches[all_matches[:, 0] == i, 1][0])]
                self.matrix[gt_class, detection_class] += 1
            else:
                # This ground truth has no match (false negative)
                gt_class = gt_classes[i]
                self.matrix[gt_class, self.num_classes] += 1
        
        # Process detection boxes for false positives
        for i, detection in enumerate(detections):
            if all_matches.shape[0] == 0 or all_matches[all_matches[:, 1] == i].shape[0] == 0:
                # This detection has no match (false positive)
                detection_class = detection_classes[i]
                self.matrix[self.num_classes, detection_class] += 1


def parse_ground_truth(gt_data):
    """Parse ground truth annotations."""
    # Handle both 0-based and 1-based category IDs
    categories = {cat['id']: cat['name'] for cat in gt_data['categories']}
    
    # Check if categories use 0-based or 1-based indexing
    min_cat_id = min(categories.keys())
    max_cat_id = max(categories.keys())
    
    print(f"Category IDs range from {min_cat_id} to {max_cat_id}")
    
    # Group annotations by image_id
    gt_by_image = defaultdict(list)
    for ann in gt_data['annotations']:
        bbox_xyxy = coco_bbox_to_xyxy(ann['bbox'])
        # Use category_id as-is (no conversion)
        gt_by_image[ann['image_id']].append([
            ann['category_id'],
            bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2], bbox_xyxy[3]
        ])
    
    return gt_by_image, categories


def parse_predictions(pred_data, score_threshold=0.5):
    """Parse prediction annotations."""
    pred_by_image = defaultdict(list)
    for pred in pred_data:
        if pred['score'] >= score_threshold:
            bbox_xyxy = coco_bbox_to_xyxy(pred['bbox'])
            pred_by_image[pred['image_id']].append([
                bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2], bbox_xyxy[3],
                pred['score'],
                pred['category_id']  # Use category_id as-is (no conversion)
            ])
    
    return pred_by_image


def plot_confusion_matrix(cm, class_names, normalize=True, show_text=True, show_fpfn=True, figsize=(15, 15)):
    '''
    Plot confusion matrix similar to the reference implementation.
    Parameters
    ----------
    cm : a nxn dim numpy array.
    class_names: a list of class names (str type)
    normalize: whether to normalize the values
    show_text: whether to show value in each block of the matrix
    show_fpfn: whether to show false positives and false negatives
    Returns
    -------
    fig: a plot of confusion matrix along with colorbar
    '''
    if show_fpfn:
        conf_mat = cm
        x_labels = copy.deepcopy(class_names)
        y_labels = copy.deepcopy(class_names)
        x_labels.append('FN')
        y_labels.append('FP')
    else:
        conf_mat = cm[0:cm.shape[0]-1, 0:cm.shape[0]-1]
        x_labels = class_names
        y_labels = class_names
    
    my_cmap = 'Greens'
    c_m = conf_mat.copy()
    
    if normalize:
        row_sums = c_m.sum(axis=1)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        c_m = c_m / row_sums[:, np.newaxis]
        c_m = np.round(c_m, 3)
    
    print('*' * 80)
    print('NOTE: In confusion_matrix the last column "FN" shows False Negatives (undetected objects)')
    print('      and the last row "FP" shows False Positives (wrong detections)')
    print('*' * 80)
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(c_m, cmap=my_cmap, vmin=0, vmax=1 if normalize else None)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(y_labels)))
    ax.set_yticks(np.arange(len(x_labels)))
    ax.set_xticklabels(y_labels)
    ax.set_yticklabels(x_labels)
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="center", rotation_mode="anchor")
    
    if show_text:
        for i in range(len(x_labels)):
            for j in range(len(y_labels)):
                text = ax.text(j, i, c_m[i, j], color="k", ha="center", va="center")
    
    ax.set_title("Normalized Confusion Matrix" if normalize else "Confusion Matrix")
    fig.tight_layout()
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(vmin=0, vmax=1 if normalize else c_m.max()))
    sm._A = []
    plt.colorbar(sm, ax=ax)
    
    return fig


def print_detection_metrics(cm, class_names):
    """Print detection-specific metrics from confusion matrix."""
    # Exclude FP/FN row/column for per-class metrics
    class_cm = cm[:-1, :-1]
    
    print(f"\n{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'TP':<6} {'FP':<6} {'FN':<6}")
    print("-" * 80)
    
    overall_tp = 0
    overall_fp = 0
    overall_fn = 0
    
    for i, class_name in enumerate(class_names):
        tp = class_cm[i, i]
        fp = cm[-1, i]  # False positives (last row)
        fn = cm[i, -1]  # False negatives (last column)
        
        overall_tp += tp
        overall_fp += fp
        overall_fn += fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{class_name:<20} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f} {int(tp):<6} {int(fp):<6} {int(fn):<6}")
    
    # Overall metrics
    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    print("-" * 80)
    print(f"{'Overall':<20} {overall_precision:<10.3f} {overall_recall:<10.3f} {overall_f1:<10.3f} {int(overall_tp):<6} {int(overall_fp):<6} {int(overall_fn):<6}")


def main():
    parser = argparse.ArgumentParser(description='Generate confusion matrix from COCO object detection results')
    parser.add_argument('--gt', required=True, help='Path to ground truth JSON file (COCO format)')
    parser.add_argument('--pred', required=True, help='Path to predictions JSON file (COCO format)')
    parser.add_argument('--score-threshold', type=float, default=0.3, 
                       help='Score threshold for predictions (default: 0.3)')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                       help='IoU threshold for matching detections (default: 0.5)')
    parser.add_argument('--normalize', action='store_true', 
                       help='Normalize confusion matrix by true class')
    parser.add_argument('--no-background', action='store_true',
                       help='Hide false positives and false negatives')
    parser.add_argument('--no-text', action='store_true',
                       help='Hide text values in confusion matrix')
    parser.add_argument('--output', help='Output path for confusion matrix plot (e.g., confusion_matrix.png)')
    parser.add_argument('--figsize', nargs=2, type=int, default=[15, 15],
                       help='Figure size for plot (width height)')
    parser.add_argument('--show-metrics', action='store_true',
                       help='Show detailed detection metrics')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading ground truth annotations...")
    gt_data = load_coco_annotations(args.gt)
    
    print("Loading predictions...")
    pred_data = load_coco_annotations(args.pred)
    
    # Parse data
    print("Parsing ground truth...")
    gt_by_image, categories = parse_ground_truth(gt_data)
    
    print("Parsing predictions...")
    pred_by_image = parse_predictions(pred_data, args.score_threshold)
    
    # Create class names list based on category IDs
    # Sort by category ID to ensure correct order
    sorted_categories = sorted(categories.items())
    class_names = [name for cat_id, name in sorted_categories]
    
    print(f"Found {len(class_names)} categories: {class_names}")
    print(f"Processing {len(gt_by_image)} images with ground truth")
    print(f"Found predictions for {len(pred_by_image)} images")
    
    # Initialize confusion matrix
    conf_mat = ConfusionMatrix(
        num_classes=len(class_names), 
        CONF_THRESHOLD=args.score_threshold, 
        IOU_THRESHOLD=args.iou_threshold
    )
    
    # Process all images
    print(f"Processing images with IoU threshold: {args.iou_threshold}")
    processed_images = 0
    
    all_image_ids = set(gt_by_image.keys()) | set(pred_by_image.keys())
    
    for img_id in all_image_ids:
        gt_boxes = np.array(gt_by_image.get(img_id, []))
        pred_boxes = np.array(pred_by_image.get(img_id, []))
        
        conf_mat.process_batch(pred_boxes, gt_boxes)
        processed_images += 1
    
    print(f"Processed {processed_images} images")
    
    # Plot confusion matrix
    fig = plot_confusion_matrix(
        conf_mat.matrix, 
        class_names, 
        normalize=args.normalize,
        show_text=not args.no_text,
        show_fpfn=not args.no_background,
        figsize=tuple(args.figsize)
    )
    
    if args.output:
        print(f"Saving confusion matrix to {args.output}")
        fig.savefig(args.output, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    # Print metrics if requested
    if args.show_metrics:
        print_detection_metrics(conf_mat.matrix, class_names)


if __name__ == '__main__':
    main()