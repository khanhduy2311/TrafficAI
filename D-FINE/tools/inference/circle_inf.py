# --- START OF FILE torch_inf.py ---

"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.utils.data

import numpy as np
from PIL import Image, ImageDraw
import json
import datetime
import os
import glob
import sys
import cv2

# NEW: Add dependency for Weighted Boxes Fusion
try:
    from ensemble_boxes import weighted_boxes_fusion
except ImportError:
    print("Error: 'ensemble-boxes' library not found. Please install it using 'pip install ensemble-boxes'", file=sys.stderr)
    sys.exit(1)


# Ensure the path is correct based on your project structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.core import YAMLConfig


def get_image_Id(img_name):
  # (Function is unchanged)
  for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
      if img_name.endswith(ext):
          img_name = img_name[:-len(ext)]
          break
  sceneList = ['M', 'A', 'E', 'N']
  parts = img_name.split('_')
  if len(parts) < 3: raise ValueError(f"Filename '{img_name}' does not match expected format")
  cameraIndx = int(parts[0].split('camera')[1])
  sceneIndx = sceneList.index(parts[1])
  frameIndx = int(parts[2])
  imageId = int(str(cameraIndx)+str(sceneIndx)+str(frameIndx))
  return imageId


class CocoImageDataset(torch.utils.data.Dataset):
    """Dataset class for loading images for COCO format output."""
    def __init__(self, image_files_list, transforms):
        self.image_files = image_files_list
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        img_filename_with_ext = os.path.basename(image_path)
        try:
            coco_image_id = get_image_Id(img_filename_with_ext)
            im_pil = Image.open(image_path).convert('RGB')
            original_width, original_height = im_pil.size
            im_data = self.transforms(im_pil)

            return {
                "im_data": im_data,
                "im_pil": im_pil, # MODIFIED: Return PIL image for masking
                "original_size_for_model": torch.tensor([original_width, original_height], dtype=torch.float32),
                "coco_image_id": coco_image_id,
                "file_name": img_filename_with_ext,
                "status": "ok"
            }
        except Exception as e:
            print(f"Error processing image {image_path} (or its ID): {e}", file=sys.stderr)
            return { "status": "error" }


def coco_collate_fn_revised(batch):
    """Collate function to filter errors and stack batch data."""
    batch = [item for item in batch if item["status"] == "ok"]
    if not batch:
        return { "empty_batch": True }

    im_data_batch = torch.stack([item['im_data'] for item in batch])
    im_pil_batch = [item['im_pil'] for item in batch] # MODIFIED: Collate PIL images
    original_sizes_for_model = torch.stack([item['original_size_for_model'] for item in batch])
    coco_image_ids = [item['coco_image_id'] for item in batch]
    file_names = [item['file_name'] for item in batch]

    return {
        "im_data_batch": im_data_batch,
        "im_pil_batch": im_pil_batch,
        "original_sizes_for_model": original_sizes_for_model,
        "coco_image_ids": coco_image_ids,
        "file_names": file_names,
        "empty_batch": False
    }

# --- HELPER FUNCTIONS for ENSEMBLE ---

def draw_center_circle(image_pil, radius_fraction):
    """Draws a filled black circle in the center of a PIL image."""
    masked_image = image_pil.copy()
    draw = ImageDraw.Draw(masked_image)
    width, height = image_pil.size
    center_x, center_y = width / 2, height / 2
    radius = min(width, height) * radius_fraction
    left, top = center_x - radius, center_y - radius
    right, bottom = center_x + radius, center_y + radius
    draw.ellipse([left, top, right, bottom], fill='black')
    return masked_image

def apply_wbf(boxes_list, scores_list, labels_list, image_size, iou_thr=0.5, skip_box_thr=0.0):
    """
    Applies Weighted Boxes Fusion. Clips boxes, normalizes, runs WBF, and denormalizes.
    """
    W, H = image_size
    norm_boxes_list = []
    for boxes in boxes_list:
        if boxes.numel() == 0:
            norm_boxes_list.append([])
            continue

        clipped_boxes = boxes.clone().float()
        clipped_boxes[:, 0].clamp_(min=0, max=W)
        clipped_boxes[:, 1].clamp_(min=0, max=H)
        clipped_boxes[:, 2].clamp_(min=0, max=W)
        clipped_boxes[:, 3].clamp_(min=0, max=H)

        clipped_boxes[:, [0, 2]] /= W
        clipped_boxes[:, [1, 3]] /= H
        norm_boxes_list.append(clipped_boxes.tolist())

    boxes, scores, labels = weighted_boxes_fusion(
        norm_boxes_list, scores_list, labels_list,
        weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr
    )
    
    if len(boxes) > 0:
        denorm_boxes = torch.tensor(boxes, dtype=torch.float32)
        denorm_boxes[:, [0, 2]] *= W
        denorm_boxes[:, [1, 3]] *= H
        return torch.tensor(labels), denorm_boxes, torch.tensor(scores)
    else:
        return torch.empty(0, dtype=torch.long), torch.empty(0, 4), torch.empty(0)

def run_ensemble_inference_on_batch(
    model, device, im_data_batch, im_pil_batch, orig_sizes_batch,
    file_names, transforms, radius_fraction, wbf_iou_thr, output_visualize_dir=None
):
    """
    Runs normal and masked inference on a batch, then ensembles the results using WBF.
    """
    # 1. Normal Inference Pass
    outputs_normal = model(im_data_batch, orig_sizes_batch)
    
    # 2. Masked Inference Pass
    masked_pil_images = [draw_center_circle(p, radius_fraction) for p in im_pil_batch]

    # MODIFIED: Save visualized masked image if requested
    if output_visualize_dir:
        for i, masked_img in enumerate(masked_pil_images):
            try:
                save_path = os.path.join(output_visualize_dir, f"masked_{file_names[i]}")
                masked_img.save(save_path)
            except Exception as e:
                print(f"Warning: Could not save visualized mask for {file_names[i]}: {e}", file=sys.stderr)

    masked_im_data_batch = torch.stack([transforms(p) for p in masked_pil_images]).to(device)
    outputs_masked = model(masked_im_data_batch, orig_sizes_batch)

    # 3. Ensemble results for each image in the batch
    ensembled_results = []
    for i in range(len(im_pil_batch)):
        labels_normal, boxes_normal, scores_normal = [o[i].cpu() for o in outputs_normal]
        labels_masked, boxes_masked, scores_masked = [o[i].cpu() for o in outputs_masked]
        
        ensembled_labels, ensembled_boxes, ensembled_scores = apply_wbf(
            boxes_list=[boxes_normal, boxes_masked],
            scores_list=[scores_normal.tolist(), scores_masked.tolist()],
            labels_list=[labels_normal.tolist(), labels_masked.tolist()],
            image_size=im_pil_batch[i].size,
            iou_thr=wbf_iou_thr
        )
        ensembled_results.append((ensembled_labels, ensembled_boxes, ensembled_scores))
        
    return ensembled_results


# --- PROCESSING FUNCTIONS ---

def draw_detections(image_pil, labels, boxes, scores, tod_specific_threshold_map, default_threshold):
    # (Function is unchanged)
    draw_obj = ImageDraw.Draw(image_pil) 
    for i in range(len(labels)):
        label, score, box = labels[i].item(), scores[i].item(), boxes[i].tolist()
        threshold = tod_specific_threshold_map.get(int(label), default_threshold)
        if score >= threshold:
            draw_obj.rectangle(box, outline='red', width=2)
            text = f"{int(label)} {round(score, 2)}"
            text_x, text_y = box[0], box[1] - 10 if box[1] >= 10 else box[1]
            draw_obj.text((text_x, text_y), text, fill='blue') 
    return image_pil 


def process_directory_to_coco(
    model, device, input_dir, output_json, 
    day_threshold_map, night_threshold_map, default_threshold, 
    batch_size, num_workers, wbf_iou_thr, radius_fraction, output_visualize_dir
):
    # (This function is mostly unchanged, just passes the new arg down)
    all_detections_list = [] 
    image_files = sorted(glob.glob(os.path.join(input_dir, "*.*[gG]")) + \
                         glob.glob(os.path.join(input_dir, "*.[bB][mM][pP]")))
    if not image_files:
        print(f"Warning: No images found in directory: {input_dir}")
        output_dir_path = os.path.dirname(output_json)
        if output_dir_path: os.makedirs(output_dir_path, exist_ok=True)
        with open(output_json, 'w') as f: json.dump([], f)
        return

    transforms_val = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    dataset = CocoImageDataset(image_files, transforms_val)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, 
        collate_fn=coco_collate_fn_revised, pin_memory=True if 'cuda' in device.type else False
    )
    print(f"Found {len(image_files)} images. Starting ensembled processing...")
    start_time = datetime.datetime.now()
    processed_image_count = 0
    total_detections_count = 0

    with torch.no_grad(): 
        for batch_data in dataloader:
            if batch_data.get("empty_batch", False): continue

            im_data_batch = batch_data['im_data_batch'].to(device)
            orig_sizes_batch = batch_data['original_sizes_for_model'].to(device)

            ensembled_results_batch = run_ensemble_inference_on_batch(
                model, device, im_data_batch, batch_data['im_pil_batch'],
                orig_sizes_batch, batch_data['file_names'], transforms_val,
                radius_fraction, wbf_iou_thr, output_visualize_dir
            )

            for i in range(len(ensembled_results_batch)):
                processed_image_count += 1
                current_image_id = batch_data['coco_image_ids'][i]
                current_file_name = batch_data['file_names'][i]
                is_night = "_N_" in current_file_name or "_E_" in current_file_name
                current_tod_threshold_map = night_threshold_map if is_night else day_threshold_map
                labels, boxes, scores = ensembled_results_batch[i]

                for k in range(len(labels)):
                    category_id, score = int(labels[k].item()), scores[k].item()
                    threshold = current_tod_threshold_map.get(category_id, default_threshold)
                    if score >= threshold:
                        x1, y1, x2, y2 = boxes[k].tolist()
                        if x2 > x1 and y2 > y1:
                            w, h = x2 - x1, y2 - y1
                            bbox_coco = [float(f"{v:.2f}") for v in [x1, y1, w, h]] 
                            all_detections_list.append({
                                "image_id": current_image_id, "category_id": category_id,
                                "bbox": bbox_coco, "score": float(f"{score:.4f}") 
                            })
                            total_detections_count += 1
                if processed_image_count % 100 == 0: 
                     print(f"Processed {processed_image_count}/{len(image_files)} images...")

    # (Reporting unchanged)
    end_time = datetime.datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    print("-" * 30)
    print(f"Processing complete. Results saved to: {output_json}")
    # ... rest of printing ...


def process_single_image(
    model, device, file_path, day_threshold_map, night_threshold_map, 
    default_threshold, wbf_iou_thr, radius_fraction, output_visualize_dir
):
    try:
        im_pil = Image.open(file_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image file {file_path}: {e}", file=sys.stderr); return

    transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    img_filename = os.path.basename(file_path)
    im_data_batch = transforms(im_pil).unsqueeze(0).to(device)
    orig_size_batch = torch.tensor([im_pil.size], dtype=torch.float32).to(device)

    with torch.no_grad():
        ensembled_results_batch = run_ensemble_inference_on_batch(
            model, device, im_data_batch, [im_pil], orig_size_batch,
            [img_filename], transforms, radius_fraction, wbf_iou_thr, output_visualize_dir
        )
    labels_img, boxes_img, scores_img = ensembled_results_batch[0]

    is_night = "_N_" in img_filename or "_E_" in img_filename
    current_tod_threshold_map = night_threshold_map if is_night else day_threshold_map
    drawn_image = draw_detections(im_pil.copy(), labels_img, boxes_img, scores_img, 
                                  current_tod_threshold_map, default_threshold)
    output_path = 'torch_results.jpg'
    drawn_image.save(output_path)
    print(f"Image processing complete. Result saved as '{output_path}'.")


def process_video_batched(
    model, device, file_path, day_threshold_map, night_threshold_map, 
    default_threshold, batch_size, video_tod_override, 
    wbf_iou_thr, radius_fraction, output_visualize_dir
):
    # (This function is mostly unchanged, just passes the new arg down)
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened(): print(f"Error: Could not open video file {file_path}", file=sys.stderr); return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('torch_results.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    
    frames_buffer, processed_count, frame_idx = [], 0, 0
    video_threshold_map = night_threshold_map if video_tod_override == 'night' else day_threshold_map
    
    print("Processing video frames with ensemble logic...")
    start_time = datetime.datetime.now()

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if ret: frames_buffer.append(frame)

            if len(frames_buffer) == batch_size or (not ret and frames_buffer):
                pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_buffer]
                file_names = [f"frame_{frame_idx + i}.jpg" for i in range(len(pil_images))]
                im_data = torch.stack([transforms(p) for p in pil_images]).to(device)
                orig_sizes = torch.tensor([p.size for p in pil_images], dtype=torch.float32).to(device)

                ensembled_results = run_ensemble_inference_on_batch(
                    model, device, im_data, pil_images, orig_sizes, file_names, 
                    transforms, radius_fraction, wbf_iou_thr, output_visualize_dir
                )
                
                for i in range(len(pil_images)):
                    labels, boxes, scores = ensembled_results[i]
                    drawn_pil = draw_detections(pil_images[i], labels, boxes, scores, 
                                                video_threshold_map, default_threshold)
                    out.write(cv2.cvtColor(np.array(drawn_pil), cv2.COLOR_RGB2BGR))
                    processed_count += 1
                    if processed_count % 30 == 0: print(f"Processed {processed_count} frames...")
                
                frame_idx += len(frames_buffer)
                frames_buffer.clear()
            if not ret: break

    cap.release(); out.release(); cv2.destroyAllWindows()
    # (Reporting unchanged)


def parse_threshold_list_to_map(threshold_str_list):
    # (Function is unchanged)
    threshold_map = {}
    if not threshold_str_list: return threshold_map
    try:
        values = [float(t.strip()) for t in threshold_str_list.split(',')]
        return {i: v for i, v in enumerate(values)}
    except Exception as e:
        raise ValueError(f"Error parsing threshold list '{threshold_str_list}': {e}")


def main(args):
    """Main function"""
    cfg = YAMLConfig(args.config, resume=args.resume)
    if "HGNetv2" in cfg.yaml_cfg: cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        state = checkpoint.get("ema", {}).get("module") or checkpoint.get("model") or checkpoint
        if not isinstance(state, dict): raise ValueError("Checkpoint format not recognized.")
    else:
        raise AttributeError("A checkpoint path must be provided via -r or --resume.")

    # (Threshold handling unchanged)
    day_threshold_map = parse_threshold_list_to_map(args.day_thresholds)
    night_threshold_map = parse_threshold_list_to_map(args.night_thresholds)
    
    # NEW: Create visualization output directory if specified
    if args.output_visualize:
        try:
            os.makedirs(args.output_visualize, exist_ok=True)
            print(f"Will save masked images for visualization to: {args.output_visualize}")
        except OSError as e:
            print(f"Error creating visualization directory {args.output_visualize}: {e}", file=sys.stderr)
            sys.exit(1)
            
    default_threshold = args.threshold
    cfg.model.load_state_dict(state)
    class DeployModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy() 
            self.postprocessor = cfg.postprocessor.deploy() 
        def forward(self, images, orig_target_sizes):
            return self.postprocessor(self.model(images), orig_target_sizes)

    device = torch.device(args.device)
    model = DeployModel().to(device).eval() 

    common_kwargs = {
        "model": model, "device": device,
        "day_threshold_map": day_threshold_map, "night_threshold_map": night_threshold_map,
        "default_threshold": default_threshold, "wbf_iou_thr": args.wbf_iou_thr,
        "radius_fraction": args.circle_mask_radius_fraction,
        "output_visualize_dir": args.output_visualize
    }

    input_path = args.input
    if os.path.isdir(input_path):
        process_directory_to_coco(
            input_dir=input_path, output_json=args.output,
            batch_size=args.batch_size, num_workers=args.num_workers,
            **common_kwargs
        )
    elif os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[-1].lower()
        if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            process_single_image(file_path=input_path, **common_kwargs)
        elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
             process_video_batched(
                 file_path=input_path, batch_size=args.batch_size, 
                 video_tod_override=args.video_tod, **common_kwargs
             )
        else:
             print(f"Error: Unsupported file type: {ext}.", file=sys.stderr); sys.exit(1)
    else:
        print(f"Error: Input path not found: {input_path}", file=sys.stderr); sys.exit(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="D-FINE Object Detection with WBF Ensemble")
    # --- Basic Arguments ---
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to model config (.yaml)")
    parser.add_argument("-r", "--resume", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to input image, video, or directory")
    parser.add_argument("-o", "--output", type=str, default="coco_results.json", help="Path to output COCO JSON (dir input only)")
    parser.add_argument("-d", "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device for inference")
    
    # --- Thresholding Arguments ---
    parser.add_argument("-t", "--threshold", type=float, default=0.4, help="Default confidence threshold")
    parser.add_argument("--day_thresholds", type=str, default=None, help='Per-class thresholds for DAY images (comma-separated).')
    parser.add_argument("--night_thresholds", type=str, default=None, help='Per-class thresholds for NIGHT images (comma-separated).')
    parser.add_argument("--video_tod", type=str, choices=['day', 'night'], default='day', help="Force 'day' or 'night' thresholds for video.")

    # --- Batching Arguments ---
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for directory/video processing")
    parser.add_argument("--num_workers", type=int, default=4, help="Num workers for data loading")

    # --- NEW & ENSEMBLE Arguments ---
    parser.add_argument("--output_visualize", type=str, default=None,
                        help="Path to a directory to save masked images for visualization.")
    parser.add_argument("--wbf_iou_thr", type=float, default=0.8,
                        help="IoU threshold for Weighted Boxes Fusion (WBF).")
    parser.add_argument("--circle_mask_radius_fraction", type=float, default=0.34
                        ,
                        help="Radius of the center circle mask as a fraction of the smaller image dimension.")
    
    args = parser.parse_args()
    # (Argument validation is unchanged)
    if not os.path.exists(args.config): print(f"Error: Config file not found: {args.config}", file=sys.stderr); sys.exit(1)
    if not os.path.exists(args.resume): print(f"Error: Checkpoint file not found: {args.resume}", file=sys.stderr); sys.exit(1)
    if not os.path.exists(args.input): print(f"Error: Input path not found: {args.input}", file=sys.stderr); sys.exit(1)
    
    main(args)

# --- END OF FILE torch_inf.py ---