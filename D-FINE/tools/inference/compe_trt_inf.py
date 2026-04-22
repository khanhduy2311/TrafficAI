"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
Modified for AI City Challenge 2025 Track 4
"""

import collections
import contextlib
import os
import time
import json
import glob
from collections import OrderedDict
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch
import cv2  # Use OpenCV for fast image processing

# Removed PIL import as it's no longer needed

class TimeProfiler(contextlib.ContextDecorator):
    def __init__(self):
        self.total = 0
    def __enter__(self):
        self.start = self.time()
        return self
    def __exit__(self, type, value, traceback):
        self.total += self.time() - self.start
    def reset(self):
        self.total = 0
    def time(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()

class TRTInference(object):
    def __init__(self, engine_path, device="cuda:0", backend="torch", max_batch_size=1, verbose=False):
        self.engine_path = engine_path
        self.device = device
        self.backend = backend
        self.max_batch_size = max_batch_size
        self.logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.bindings = self.get_bindings(self.engine, self.context, self.max_batch_size, self.device)
        self.bindings_addr = OrderedDict((n, v.ptr) for n, v in self.bindings.items())
        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()
        self.time_profile = TimeProfiler()
    def load_engine(self, path):
        trt.init_libnvinfer_plugins(self.logger, "")
        with open(path, "rb") as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    def get_input_names(self):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                names.append(name)
        return names
    def get_output_names(self):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                names.append(name)
        return names
    def get_bindings(self, engine, context, max_batch_size=1, device=None) -> OrderedDict:
        Binding = collections.namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        bindings = OrderedDict()
        for i, name in enumerate(engine):
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            if shape[0] == -1:
                shape[0] = max_batch_size
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    context.set_input_shape(name, shape)
            data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            bindings[name] = Binding(name, dtype, shape, data, data.data_ptr())
        return bindings
    def run_torch(self, blob):
        for n in self.input_names:
            if blob[n].dtype is not self.bindings[n].data.dtype:
                blob[n] = blob[n].to(dtype=self.bindings[n].data.dtype)
            if self.bindings[n].shape != blob[n].shape:
                self.context.set_input_shape(n, blob[n].shape)
                self.bindings[n] = self.bindings[n]._replace(shape=blob[n].shape)
            assert self.bindings[n].data.dtype == blob[n].dtype, "{} dtype mismatch".format(n)
        self.bindings_addr.update({n: blob[n].data_ptr() for n in self.input_names})
        self.context.execute_v2(list(self.bindings_addr.values()))
        outputs = {n: self.bindings[n].data for n in self.output_names}
        return outputs
    def __call__(self, blob):
        if self.backend == "torch":
            return self.run_torch(blob)
        else:
            raise NotImplementedError("Only 'torch' backend is implemented.")
    def synchronize(self):
        if self.backend == "torch" and torch.cuda.is_available():
            torch.cuda.synchronize()

def warmup_gpu(model, device, warmup_iterations=10):
    """
    Warmup the GPU by running inference on dummy data.
    
    Args:
        model: The TensorRT model
        device: Device to use
        warmup_iterations: Number of warmup iterations
    """
    print(f"Warming up GPU with {warmup_iterations} iterations...")
    
    # Create dummy input data matching expected input shape
    dummy_img = torch.randn(1, 3, 960, 960).to(device)
    dummy_orig_size = torch.tensor([[960, 960]]).to(device)
    dummy_blob = {"images": dummy_img, "orig_target_sizes": dummy_orig_size}
    
    # Run warmup iterations
    warmup_start = time.time()
    for i in range(warmup_iterations):
        with torch.no_grad():
            _ = model(dummy_blob)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if (i + 1) % 20 == 0:
            print(f"Warmup iteration {i + 1}/{warmup_iterations}")
    
    warmup_end = time.time()
    print(f"GPU warmup completed in {warmup_end - warmup_start:.2f} seconds")
    print("=" * 50)

def get_image_id(img_name):
    img_name = img_name.split('.png')[0]
    scene_list = ['M', 'A', 'E', 'N']
    camera_idx = int(img_name.split('_')[0].split('camera')[1])
    scene_idx = scene_list.index(img_name.split('_')[1])
    frame_idx = int(img_name.split('_')[2])
    image_id = int(str(camera_idx) + str(scene_idx) + str(frame_idx))
    return image_id

def process_detections(output, image_name, score_threshold=0.4):
    detections = []
    boxes = output["boxes"][0].cpu().numpy()
    scores = output["scores"][0].cpu().numpy()
    labels = output["labels"][0].cpu().numpy()
    valid_indices = scores > score_threshold
    boxes = boxes[valid_indices]
    scores = scores[valid_indices]
    labels = labels[valid_indices]
    image_id = get_image_id(image_name)
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        width = x2 - x1
        height = y2 - y1
        detection = {"image_id": image_id, "category_id": int(labels[i]), "bbox": [float(x1), float(y1), float(width), float(height)], "score": float(scores[i])}
        detections.append(detection)
    return detections

def process_images_sequential(model, input_folder, output_file, device, score_threshold, skip_warmup=False, warmup_iterations=10):
    # GPU Warmup
    if not skip_warmup:
        warmup_gpu(model, device, warmup_iterations)
    
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
    image_files.sort()
    print(f"Found {len(image_files)} images to process")
    
    all_detections = []
    total_preprocess_time, total_inference_time, total_postprocess_time = 0, 0, 0
    processed_images = 0
    
    timer_start = time.time()
    for i, image_path in enumerate(image_files):
        image_name = os.path.basename(image_path)
        
        # Image loading (not timed)
        try:
            img_bgr = cv2.imread(image_path)
            if img_bgr is None: 
                raise IOError(f"Could not read image: {image_path}")
        except Exception as e:
            print(f"Error loading {image_name}: {e}")
            continue
        
        # Preprocessing (timed)
        preprocess_start = time.time()
        try:
            h, w, _ = img_bgr.shape
            orig_size = torch.tensor([w, h])[None].to(device)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (960, 960), interpolation=cv2.INTER_AREA)
            im_data = torch.from_numpy(img_resized).permute(2, 0, 1).float().div(255)
            im_data = im_data.unsqueeze(0).contiguous()
            blob = {"images": im_data.to(device), "orig_target_sizes": orig_size.to(device)}
        except Exception as e:
            print(f"Error preprocessing {image_name}: {e}")
            continue
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        preprocess_end = time.time()
        total_preprocess_time += (preprocess_end - preprocess_start)
        
        # Inference (timed)
        inference_start = time.time()
        with torch.no_grad():
            output = model(blob)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_end = time.time()
        total_inference_time += (inference_end - inference_start)
        
        # Postprocessing (timed)
        postprocess_start = time.time()
        detections = process_detections(output, image_name, score_threshold=score_threshold)
        all_detections.extend(detections)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        postprocess_end = time.time()
        total_postprocess_time += (postprocess_end - postprocess_start)
        
        processed_images += 1
        if (i + 1) % 100 == 0: 
            print(f"Processed {i + 1}/{len(image_files)} images")
    
    timer_end = time.time()
    
    # Save results
    with open(output_file, 'w') as f: 
        json.dump(all_detections, f, indent=2)
    
    # Calculate and display results
    elapsed_time = timer_end - timer_start
    total_processing_time = total_preprocess_time + total_inference_time + total_postprocess_time
    fps = processed_images / total_processing_time if total_processing_time > 0 else 0
    normalized_fps = min(fps / 25.0, 1.0)
    
    print(f"Processed {processed_images} images in {elapsed_time:.2f} seconds.")
    if processed_images > 0:
        print(f"Avg Image Preprocess Time      : {(total_preprocess_time / processed_images) * 1000:.2f} ms")
        print(f"Avg Inference Time       : {(total_inference_time / processed_images) * 1000:.2f} ms")
        print(f"Avg Postprocessing Time  : {(total_postprocess_time / processed_images) * 1000:.2f} ms")
        print(f"Avg Processing Time           : {(total_processing_time / processed_images) * 1000:.2f} ms")
    
    print(f"\n--- Evaluation Complete ---")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Total processing time: {total_processing_time:.2f} seconds")
    print(f"FPS: {fps:.2f}")
    print(f"Normalized FPS: {normalized_fps:.4f}")
    
    print(f"\nResults saved to: {output_file}")
    print(f"Total detections: {len(all_detections)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI City Challenge 2025 Track 4 Inference")
    parser.add_argument("-trt", "--trt", required=True, help="Path to TensorRT engine file")
    parser.add_argument("-i", "--input", required=True, help="Path to input folder containing test images")
    parser.add_argument("-o", "--output", default="submission.json", help="Output JSON file path")
    parser.add_argument("-d", "--device", default="cuda:0", help="Device to use for inference")
    parser.add_argument("--score-threshold", type=float, default=0.0001, help="Score threshold for filtering detections")
    parser.add_argument('--warmup_iterations', type=int, default=200, help='Number of GPU warmup iterations')
    parser.add_argument('--skip_warmup', action='store_true', help='Skip GPU warmup')
    args = parser.parse_args()
    
    if not os.path.exists(args.trt): 
        exit(f"Error: TensorRT engine file not found: {args.trt}")
    if not os.path.exists(args.input): 
        exit(f"Error: Input folder not found: {args.input}")
    
    print("Initializing TensorRT model...")
    model = TRTInference(args.trt, device=args.device, max_batch_size=1)
    print("Starting sequential image processing...")
    process_images_sequential(model, args.input, args.output, args.device,args.score_threshold, args.skip_warmup, args.warmup_iterations)