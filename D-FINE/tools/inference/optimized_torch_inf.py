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

# Ensure the path is correct based on your project structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.core import YAMLConfig


def get_image_Id(img_name):
  img_name = img_name.split('.png')[0] # Assuming extension can also be .jpg etc. needs to be robust
  # Make get_image_Id more robust to different extensions by stripping any known extension
  for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
      if img_name.endswith(ext):
          img_name = img_name[:-len(ext)]
          break

  sceneList = ['M', 'A', 'E', 'N']
  parts = img_name.split('_')
  if len(parts) < 3:
      raise ValueError(f"Filename '{img_name}' does not match expected format 'cameraX_SCENE_frameY'")
  
  camera_part = parts[0]
  scene_part = parts[1]
  frame_part = parts[2]

  if not camera_part.startswith('camera'):
      raise ValueError(f"Camera part '{camera_part}' in '{img_name}' does not start with 'camera'")
  
  try:
      cameraIndx = int(camera_part.split('camera')[1])
  except (IndexError, ValueError) as e:
      raise ValueError(f"Could not parse camera index from '{camera_part}' in '{img_name}': {e}")
  
  if scene_part not in sceneList:
      raise ValueError(f"Scene part '{scene_part}' in '{img_name}' is not in {sceneList}")
  sceneIndx = sceneList.index(scene_part)
  
  try:
      frameIndx = int(frame_part)
  except ValueError as e:
      raise ValueError(f"Could not parse frame index from '{frame_part}' in '{img_name}': {e}")
      
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
        # img_filename_stem, _ = os.path.splitext(img_filename_with_ext) # get_image_Id will handle extension

        try:
            # Attempt to generate ID first using the provided function
            # Pass filename WITH extension to get_image_Id as it handles stripping now
            coco_image_id = get_image_Id(img_filename_with_ext)

            im_pil = Image.open(image_path).convert('RGB')
            original_width, original_height = im_pil.size
            im_data = self.transforms(im_pil) 

            return {
                "im_data": im_data,
                "original_size_for_model": torch.tensor([original_width, original_height], dtype=torch.float32),
                "coco_image_id": coco_image_id,
                "file_name": img_filename_with_ext, 
                "status": "ok"
            }
        except Exception as e:
            print(f"Error processing image {image_path} (or its ID): {e}", file=sys.stderr)
            return {
                "im_data": torch.zeros((3, 960, 960)), 
                "original_size_for_model": torch.tensor([0,0], dtype=torch.float32),
                "coco_image_id": -1, 
                "file_name": img_filename_with_ext,
                "status": "error"
            }


def coco_collate_fn_revised(batch):
    """Collate function to filter errors and stack batch data."""
    batch = [item for item in batch if item["status"] == "ok"]
    if not batch: 
        return {
            "im_data_batch": torch.empty(0, 3, 960, 960), 
            "original_sizes_for_model": torch.empty(0, 2),
            "coco_image_ids": [],
            "file_names": [],
            "empty_batch": True 
        }

    im_data_batch = torch.stack([item['im_data'] for item in batch])
    original_sizes_for_model = torch.stack([item['original_size_for_model'] for item in batch])
    coco_image_ids = [item['coco_image_id'] for item in batch]
    file_names = [item['file_name'] for item in batch] # Keep file_names for ToD determination

    return {
        "im_data_batch": im_data_batch,
        "original_sizes_for_model": original_sizes_for_model,
        "coco_image_ids": coco_image_ids,
        "file_names": file_names,
        "empty_batch": False
    }


def draw_detections(image_pil, labels, boxes, scores, tod_specific_threshold_map, default_threshold):
    """Draws filtered detections on a PIL image using ToD specific thresholds."""
    draw_obj = ImageDraw.Draw(image_pil) 

    for i in range(len(labels)):
        label = labels[i].item() 
        score = scores[i].item()
        box = boxes[i].tolist() 

        threshold = tod_specific_threshold_map.get(int(label), default_threshold)

        if score >= threshold:
            draw_obj.rectangle(box, outline='red', width=2)
            text = f"{int(label)} {round(score, 2)}"
            text_x = box[0]
            text_y = box[1] - 10 if box[1] >= 10 else box[1] 
            draw_obj.text((text_x, text_y), text, fill='blue') 
    return image_pil 


def process_directory_to_coco(model, device, input_dir, output_json, 
                              day_threshold_map, night_threshold_map, 
                              default_threshold, batch_size, num_workers):
    """
    Processes all images in a directory, applies ToD-specific per-class thresholds,
    and outputs results in COCO JSON format.
    """
    all_detections_list = [] 
    image_files = sorted(glob.glob(os.path.join(input_dir, "*.jpg")) + \
                         glob.glob(os.path.join(input_dir, "*.jpeg")) + \
                         glob.glob(os.path.join(input_dir, "*.png")) + \
                         glob.glob(os.path.join(input_dir, "*.bmp")))

    if not image_files:
        print(f"Warning: No images found in directory: {input_dir}")
        output_dir_path = os.path.dirname(output_json)
        if output_dir_path and not os.path.exists(output_dir_path):
            try:
                os.makedirs(output_dir_path)
            except OSError as e:
                print(f"Error creating output directory {output_dir_path}: {e}", file=sys.stderr)
        with open(output_json, 'w') as f:
            json.dump(all_detections_list, f, indent=2)
        print(f"Empty results list saved to {output_json}")
        return

    transforms_val = T.Compose([
        T.Resize((960, 960)), 
        T.ToTensor(),
    ])

    dataset = CocoImageDataset(image_files, transforms_val)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, collate_fn=coco_collate_fn_revised,
        pin_memory=True if 'cuda' in device.type else False
    )

    print(f"Found {len(image_files)} images. Starting processing for COCO JSON output...")
    start_time = datetime.datetime.now()
    processed_image_count = 0
    total_detections_count = 0
    failed_image_count = 0

    with torch.no_grad(): 
        for batch_data in dataloader:
            if batch_data.get("empty_batch", False):
                num_potential_failed = batch_size 
                failed_image_count += num_potential_failed
                print(f"Skipping an empty or fully failed batch (approx. {num_potential_failed} images).", file=sys.stderr)
                continue

            im_data_batch = batch_data['im_data_batch'].to(device)
            orig_sizes_batch = batch_data['original_sizes_for_model'].to(device)
            outputs = model(im_data_batch, orig_sizes_batch)
            labels_batch, boxes_batch, scores_batch = outputs

            for i in range(len(batch_data['coco_image_ids'])):
                processed_image_count += 1
                current_image_id = batch_data['coco_image_ids'][i]
                current_file_name = batch_data['file_names'][i] # Get filename for ToD check

                # Determine ToD and select appropriate threshold map
                is_night = "_N_" in current_file_name or "_E_" in current_file_name
                current_tod_threshold_map = night_threshold_map if is_night else day_threshold_map

                labels = labels_batch[i].detach().cpu() 
                boxes = boxes_batch[i].detach().cpu()   
                scores = scores_batch[i].detach().cpu() 

                for k in range(len(labels)):
                    category_id = int(labels[k].item()) 
                    score = scores[k].item()       
                    threshold = current_tod_threshold_map.get(category_id, default_threshold)

                    if score >= threshold:
                        x1, y1, x2, y2 = boxes[k].tolist()
                        if x2 > x1 and y2 > y1:
                            width = x2 - x1
                            height = y2 - y1
                            bbox_coco = [float(f"{val:.2f}") for val in [x1, y1, width, height]] 
                            detection_entry = {
                                "image_id": current_image_id,
                                "category_id": category_id,
                                "bbox": bbox_coco,
                                "score": float(f"{score:.4f}") 
                            }
                            all_detections_list.append(detection_entry)
                            total_detections_count += 1
                if processed_image_count % 100 == 0: 
                     print(f"Processed {processed_image_count}/{len(image_files)} images...")

    end_time = datetime.datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    output_dir_path = os.path.dirname(output_json)
    if output_dir_path and not os.path.exists(output_dir_path): 
        try:
            os.makedirs(output_dir_path, exist_ok=True)
        except OSError as e:
            print(f"Error creating output directory {output_dir_path}: {e}", file=sys.stderr)

    with open(output_json, 'w') as f:
        json.dump(all_detections_list, f, indent=2)

    print("-" * 30)
    print(f"Processing complete. Results saved to: {output_json}")
    print(f"Successfully processed images: {processed_image_count}")
    if failed_image_count > 0 or processed_image_count < len(image_files):
         actual_failed = len(image_files) - processed_image_count
         print(f"Warning: Failed to process {actual_failed} images.")
    print(f"Total detections meeting thresholds: {total_detections_count}")
    if elapsed_time > 0 and processed_image_count > 0:
        fps = processed_image_count / elapsed_time
        print(f"Time elapsed: {elapsed_time:.2f} seconds")
        print(f"Average processing speed: {fps:.2f} FPS")
    else:
        print(f"Time elapsed: {elapsed_time:.2f} seconds.")
    print("-" * 30)


def process_single_image(model, device, file_path, day_threshold_map, night_threshold_map, default_threshold):
    """Processes a single image and saves the result with drawn boxes."""
    try:
        im_pil = Image.open(file_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image file {file_path}: {e}", file=sys.stderr)
        return

    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]], dtype=torch.float32).to(device)
    transforms = T.Compose([T.Resize((960, 960)), T.ToTensor()])
    im_data = transforms(im_pil).unsqueeze(0).to(device)

    # Determine ToD for the single image
    img_filename = os.path.basename(file_path)
    is_night = "_N_" in img_filename or "_E_" in img_filename
    current_tod_threshold_map = night_threshold_map if is_night else day_threshold_map

    with torch.no_grad():
        outputs = model(im_data, orig_size)
        labels, boxes, scores = outputs
    labels_img, boxes_img, scores_img = labels[0].cpu(), boxes[0].cpu(), scores[0].cpu()

    drawn_image = draw_detections(im_pil.copy(), labels_img, boxes_img, scores_img, 
                                  current_tod_threshold_map, default_threshold)
    output_path = 'torch_results.jpg'
    drawn_image.save(output_path)
    print(f"Image processing complete. Result saved as '{output_path}'.")


def process_video_batched(model, device, file_path, day_threshold_map, night_threshold_map, 
                          default_threshold, batch_size, video_tod_override=None):
    """
    Processes a video using batching and saves the result with drawn boxes.
    Uses day_threshold_map by default for video frames, unless video_tod_override is 'night'.
    """
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {file_path}", file=sys.stderr)
        return

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video_path = 'torch_results.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_video_path, fourcc, fps_video, (orig_w, orig_h))
    transforms = T.Compose([T.Resize((960, 960)), T.ToTensor()])
    frames_buffer_cv2 = []
    processed_frame_count = 0

    # Determine which threshold map to use for the video
    if video_tod_override == 'night':
        video_threshold_map = night_threshold_map
        print("Applying NIGHT thresholds for video processing.")
    else:
        video_threshold_map = day_threshold_map # Default to day
        if video_tod_override is None:
            print("Applying DAY thresholds for video processing (default).")
        else: # e.g. video_tod_override == 'day'
            print("Applying DAY thresholds for video processing.")


    print("Processing video frames...")
    start_time = datetime.datetime.now()

    with torch.no_grad():
        while True:
            ret, frame_cv2 = cap.read()
            if ret: frames_buffer_cv2.append(frame_cv2)

            if len(frames_buffer_cv2) == batch_size or (not ret and len(frames_buffer_cv2) > 0):
                if not frames_buffer_cv2: break
                pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_buffer_cv2]
                original_sizes_for_model = torch.tensor(
                    [[p.size[0], p.size[1]] for p in pil_images], dtype=torch.float32
                ).to(device)
                im_data_batch = torch.stack([transforms(p) for p in pil_images]).to(device)
                outputs = model(im_data_batch, original_sizes_for_model)
                labels_batch, boxes_batch, scores_batch = outputs

                for i in range(len(pil_images)):
                    labels_img, boxes_img, scores_img = labels_batch[i].cpu(), boxes_batch[i].cpu(), scores_batch[i].cpu()
                    drawn_pil_image = draw_detections(pil_images[i], labels_img, boxes_img, scores_img, 
                                                      video_threshold_map, default_threshold) # Use video_threshold_map
                    processed_cv2_frame = cv2.cvtColor(np.array(drawn_pil_image), cv2.COLOR_RGB2BGR)
                    out_writer.write(processed_cv2_frame)
                    processed_frame_count += 1
                    if processed_frame_count % 30 == 0: print(f"Processed {processed_frame_count} frames...")
                frames_buffer_cv2.clear()
            if not ret: break

    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()
    end_time = datetime.datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    print("-" * 30)
    print(f"Video processing complete. Result saved as '{output_video_path}'.")
    print(f"Processed {processed_frame_count} frames.")
    if elapsed_time > 0 and processed_frame_count > 0:
        fps_proc = processed_frame_count / elapsed_time
        print(f"Time elapsed: {elapsed_time:.2f} seconds")
        print(f"Average processing speed: {fps_proc:.2f} FPS")
    else: print(f"Time elapsed: {elapsed_time:.2f} seconds.")
    print("-" * 30)


def parse_threshold_list_to_map(threshold_str_list):
    """ Parses a comma-separated string of threshold values into a class_id:threshold_value map. """
    threshold_map = {}
    if not threshold_str_list:
        return threshold_map
    try:
        threshold_values = [float(t.strip()) for t in threshold_str_list.split(',')]
        threshold_map = {i: threshold for i, threshold in enumerate(threshold_values)}
    except ValueError:
        raise ValueError(f"Invalid format for threshold list. Expected comma-separated floats. Got: {threshold_str_list}")
    except Exception as e:
        raise ValueError(f"Error processing threshold list '{threshold_str_list}': {e}")
    return threshold_map


def main(args):
    """Main function"""
    cfg = YAMLConfig(args.config, resume=args.resume)
    if "HGNetv2" in cfg.yaml_cfg: cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if "ema" in checkpoint and checkpoint["ema"]: state = checkpoint["ema"]["module"]
        elif "model" in checkpoint: state = checkpoint["model"]
        else: state = checkpoint
        if not isinstance(state, dict) or not any(k.endswith(('.weight', '.bias')) for k in state): # More robust check
             raise ValueError("Checkpoint format not recognized.")
    else:
        raise AttributeError("A checkpoint path must be provided via -r or --resume.")

    # --- ToD-Specific Per-Class Threshold Handling ---
    day_threshold_map = {}
    night_threshold_map = {}

    try:
        if args.day_thresholds:
            day_threshold_map = parse_threshold_list_to_map(args.day_thresholds)
            print(f"Using custom DAY thresholds (Class ID: Threshold): {day_threshold_map}")
        else:
            print("No specific DAY thresholds provided. Will use default threshold for day images.")

        if args.night_thresholds:
            night_threshold_map = parse_threshold_list_to_map(args.night_thresholds)
            print(f"Using custom NIGHT thresholds (Class ID: Threshold): {night_threshold_map}")
        else:
            print("No specific NIGHT thresholds provided. Will use default threshold for night images.")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    default_threshold = args.threshold 
    print(f"Using default confidence threshold: {default_threshold} (for classes not in specific ToD maps or if maps are not used)")

    cfg.model.load_state_dict(state)
    class DeployModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy() 
            self.postprocessor = cfg.postprocessor.deploy() 
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images) 
            return self.postprocessor(outputs, orig_target_sizes)

    device = torch.device(args.device)
    model = DeployModel().to(device)
    model.eval() 

    input_path = args.input
    if os.path.isdir(input_path):
        print(f"Input is a directory: {input_path}. Processing for COCO JSON output.")
        process_directory_to_coco(
            model=model, device=device, input_dir=input_path, output_json=args.output,
            day_threshold_map=day_threshold_map, night_threshold_map=night_threshold_map,
            default_threshold=default_threshold, batch_size=args.batch_size, num_workers=args.num_workers
        )
    elif os.path.isfile(input_path):
        print(f"Input is a file: {input_path}.")
        ext = os.path.splitext(input_path)[-1].lower()
        if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            process_single_image(
                model=model, device=device, file_path=input_path,
                day_threshold_map=day_threshold_map, night_threshold_map=night_threshold_map,
                default_threshold=default_threshold
            )
        elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
             process_video_batched(
                 model=model, device=device, file_path=input_path,
                 day_threshold_map=day_threshold_map, night_threshold_map=night_threshold_map,
                 default_threshold=default_threshold, batch_size=args.batch_size,
                 video_tod_override=args.video_tod # Pass the new argument
             )
        else:
             print(f"Error: Unsupported file type: {ext}.", file=sys.stderr); sys.exit(1)
    else:
        print(f"Error: Input path not found or invalid: {input_path}", file=sys.stderr); sys.exit(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="D-FINE Object Detection Inference with ToD Thresholds")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to model config (.yaml)")
    parser.add_argument("-r", "--resume", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to input image, video, or directory")
    parser.add_argument("-o", "--output", type=str, default="coco_results.json", help="Path to output COCO JSON (dir input only)")
    parser.add_argument("-d", "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device for inference")
    parser.add_argument("-t", "--threshold", type=float, default=0.4, help="Default confidence threshold")
    
    parser.add_argument("--day_thresholds", type=str, default=None,
                        help='Per-class thresholds for DAY images (comma-separated floats). Order corresponds to class IDs 0, 1, ... E.g., "0.5,0.4,0.6"')
    parser.add_argument("--night_thresholds", type=str, default=None,
                        help='Per-class thresholds for NIGHT images (comma-separated floats). Order corresponds to class IDs 0, 1, ... E.g., "0.4,0.3,0.5"')
    parser.add_argument("--video_tod", type=str, choices=['day', 'night'], default='day',
                        help="Specify if the video should be treated as 'day' or 'night' for thresholding (default: day).")

    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for directory/video processing")
    parser.add_argument("--num_workers", type=int, default=4, help="Num workers for data loading (directory input)")

    args = parser.parse_args()
    if not os.path.exists(args.config): print(f"Error: Config file not found: {args.config}", file=sys.stderr); sys.exit(1)
    if not os.path.exists(args.resume): print(f"Error: Checkpoint file not found: {args.resume}", file=sys.stderr); sys.exit(1)
    if not os.path.exists(args.input): print(f"Error: Input path not found: {args.input}", file=sys.stderr); sys.exit(1)
    if args.device != "cpu" and not torch.cuda.is_available():
        print(f"Warning: Device '{args.device}' unavailable, using CPU.", file=sys.stderr); args.device = "cpu"
    if args.threshold < 0 or args.threshold > 1: print(f"Warning: Default threshold {args.threshold} outside [0, 1].", file=sys.stderr)

    main(args)
# --- END OF FILE torch_inf.py ---