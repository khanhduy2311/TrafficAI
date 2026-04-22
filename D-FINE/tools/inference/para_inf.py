"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
Modified for AI City Challenge 2025 Track 4 - Parallel Inference
"""

import collections
import contextlib
import os
import time
import json
import glob
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch
import cv2
import pycuda.driver as cuda
import pycuda.autoinit

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
        return time.perf_counter()  # Using perf_counter for higher resolution timing

class ParallelTRTInference:
    def __init__(self, engine_path, device="cuda:0", max_batch_size=1, verbose=False, use_cuda_graph=False):
        """
        Initialize a TensorRT inference engine with two execution contexts for parallel inference.
        
        Args:
            engine_path: Path to the TensorRT engine file
            device: CUDA device to use
            max_batch_size: Maximum batch size (should be 1 for this use case)
            verbose: Whether to enable verbose logging
            use_cuda_graph: Whether to use CUDA Graph for potentially reduced launch overhead
        """
        self.engine_path = engine_path
        self.device = device
        self.max_batch_size = max_batch_size
        self.use_cuda_graph = use_cuda_graph
        
        # Create TensorRT logger
        self.logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)
        
        # Initialize TensorRT plugins and load engine (only once)
        trt.init_libnvinfer_plugins(self.logger, "")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        # Check TensorRT version
        self.trt_version = trt.__version__
        print(f"TensorRT version: {self.trt_version}")
        
        # Create two separate execution contexts sharing the same engine
        self.context1 = self.engine.create_execution_context()
        self.context2 = self.engine.create_execution_context()
        
        # Create separate CUDA streams for parallel execution
        self.stream1 = cuda.Stream()
        self.stream2 = cuda.Stream()
        
        # Get input and output names and binding indices
        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()
        self.binding_indices = {name: self.engine.get_binding_index(name) for name in self.input_names + self.output_names}
        
        # Check if we need to use the newer tensor address API
        # This is true for TensorRT 8.6+ typically
        self.use_tensor_address_api = hasattr(self.context1, 'set_tensor_address')
        print(f"Using tensor address API: {self.use_tensor_address_api}")
        
        # Create bindings and allocate buffers
        if self.use_tensor_address_api:
            # With newer API we just need the memory directly
            self.input_buffers1, self.output_buffers1 = self.create_io_buffers()
            self.input_buffers2, self.output_buffers2 = self.create_io_buffers()
            
            # Set shapes for dynamic inputs
            for name in self.input_names:
                shape = self.engine.get_tensor_shape(name)
                if shape[0] == -1:
                    shape[0] = max_batch_size
                self.context1.set_input_shape(name, shape)
                self.context2.set_input_shape(name, shape)
        else:
            # Older API needs bindings and buffers list
            self.bindings1, self.buffers1 = self.create_bindings(self.engine, self.context1, 
                                                                 self.max_batch_size, self.device)
            self.bindings2, self.buffers2 = self.create_bindings(self.engine, self.context2, 
                                                                self.max_batch_size, self.device)
        
        # Create CUDA graphs if requested
        self.cuda_graph1 = None
        self.cuda_graph2 = None
        if self.use_cuda_graph:
            self.prepare_cuda_graphs()
        
        # Profiling timers
        self.time_profile1 = TimeProfiler()
        self.time_profile2 = TimeProfiler()
    
    def get_input_names(self):
        """Get names of input tensors."""
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                names.append(name)
        return names
    
    def get_output_names(self):
        """Get names of output tensors."""
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                names.append(name)
        return names
    
    def create_io_buffers(self):
        """Create input and output buffers for newer TensorRT API."""
        input_buffers = {}
        output_buffers = {}
        
        # Allocate input buffers
        for name in self.input_names:
            shape = self.engine.get_tensor_shape(name)
            if shape[0] == -1:  # Dynamic batch dimension
                shape[0] = self.max_batch_size
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(self.device)
            input_buffers[name] = data
        
        # Allocate output buffers
        for name in self.output_names:
            shape = self.engine.get_tensor_shape(name)
            if shape[0] == -1:  # Dynamic batch dimension
                shape[0] = self.max_batch_size
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(self.device)
            output_buffers[name] = data
            
        return input_buffers, output_buffers
    
    def create_bindings(self, engine, context, max_batch_size=1, device=None):
        """Create bindings and GPU buffers for a specific context (for older TensorRT API)."""
        Binding = collections.namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        bindings = OrderedDict()
        buffers = []
        
        for i, name in enumerate(engine):
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            if shape[0] == -1:
                shape[0] = max_batch_size
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    context.set_input_shape(name, shape)
                    
            # Allocate device memory
            data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            bindings[name] = Binding(name, dtype, shape, data, data.data_ptr())
            buffers.append(int(data.data_ptr()))
            
        return bindings, buffers
    
    def prepare_cuda_graphs(self):
        """Prepare CUDA graphs for both contexts if enabled."""
        if not self.use_cuda_graph:
            return
        
        try:
            # Create dummy inputs
            dummy_img1 = torch.randn(1, 3, 960, 960).to(self.device)
            dummy_orig_size1 = torch.tensor([[960, 960]]).to(self.device)
            dummy_blob1 = {"images": dummy_img1, "orig_target_sizes": dummy_orig_size1}

            dummy_img2 = torch.randn(1, 3, 960, 960).to(self.device)
            dummy_orig_size2 = torch.tensor([[960, 960]]).to(self.device)
            dummy_blob2 = {"images": dummy_img2, "orig_target_sizes": dummy_orig_size2}
            
            # Perform a warmup inference to ensure everything is set up
            self._copy_inputs_to_device(dummy_blob1, context_id=1)
            self._copy_inputs_to_device(dummy_blob2, context_id=2)
            
            # Try to use the capture API appropriate for the CUDA version
            try:
                # Try newer capture API first
                cuda.Graph.begin_capture(self.stream1, cuda.Graph.CaptureMode.GLOBAL)
                self.execute_context(self.context1, context_id=1, stream=self.stream1)
                self.cuda_graph1 = cuda.Graph.end_capture()
                
                cuda.Graph.begin_capture(self.stream2, cuda.Graph.CaptureMode.GLOBAL)
                self.execute_context(self.context2, context_id=2, stream=self.stream2)
                self.cuda_graph2 = cuda.Graph.end_capture()
            except (TypeError, AttributeError) as e:
                print(f"Failed with newer CUDA graph API: {e}")
                print("Falling back to legacy capture API")
                
                # Fall back to older API
                cuda.Graph.capture_begin()
                self.execute_context(self.context1, context_id=1, stream=self.stream1)
                self.cuda_graph1 = cuda.Graph.capture_end()
                
                cuda.Graph.capture_begin()
                self.execute_context(self.context2, context_id=2, stream=self.stream2)
                self.cuda_graph2 = cuda.Graph.capture_end()
            
            print("CUDA graphs successfully created for both contexts")
        except Exception as e:
            print(f"Failed to create CUDA graphs: {e}")
            self.cuda_graph1 = None
            self.cuda_graph2 = None
            self.use_cuda_graph = False
    
    def execute_context(self, context, context_id, stream):
        """Execute a context with the given stream.
        
        Args:
            context: The TensorRT execution context
            context_id: Either 1 or 2, indicating which context is being executed
            stream: The CUDA stream to use for execution
        """
        if self.use_tensor_address_api:
            # Newer TensorRT API (8.6+)
            # Execute with tensor addresses already set
            context.execute_async_v3(stream_handle=stream.handle)
        else:
            # Older TensorRT API
            buffers = self.buffers1 if context_id == 1 else self.buffers2
            try:
                # Try execute_async_v2 first
                context.execute_async_v2(buffers, stream.handle)
            except AttributeError:
                # Fall back to execute_async
                context.execute_async(bindings=buffers, stream_handle=stream.handle)
    
    def _copy_inputs_to_device(self, blob, context_id):
        """Copy input tensors from blob to device memory and set tensor addresses if needed.
        
        Args:
            blob: Dictionary with input tensors
            context_id: Either 1 or 2, indicating which context to use
        """
        context = self.context1 if context_id == 1 else self.context2
        
        if self.use_tensor_address_api:
            # For newer TensorRT API
            input_buffers = self.input_buffers1 if context_id == 1 else self.input_buffers2
            output_buffers = self.output_buffers1 if context_id == 1 else self.output_buffers2
            
            for name in self.input_names:
                if blob[name].dtype != input_buffers[name].dtype:
                    blob[name] = blob[name].to(dtype=input_buffers[name].dtype)
                
                # Handle dynamic shapes if needed
                if input_buffers[name].shape != blob[name].shape:
                    context.set_input_shape(name, blob[name].shape)
                    # Reallocate buffer with new shape
                    input_buffers[name] = blob[name].clone()
                else:
                    input_buffers[name].copy_(blob[name])
                
                # Set input tensor address
                context.set_tensor_address(name, int(input_buffers[name].data_ptr()))
            
            # Set output tensor addresses
            for name in self.output_names:
                context.set_tensor_address(name, int(output_buffers[name].data_ptr()))
        else:
            # For older TensorRT API
            bindings = self.bindings1 if context_id == 1 else self.bindings2
            
            for name in self.input_names:
                if blob[name].dtype != bindings[name].data.dtype:
                    blob[name] = blob[name].to(dtype=bindings[name].data.dtype)
                
                if bindings[name].shape != blob[name].shape:
                    # Update binding shape if needed
                    try:
                        context.set_input_shape(name, blob[name].shape)
                    except AttributeError:
                        # Some versions might have different API
                        context.set_binding_shape(self.engine.get_binding_index(name), blob[name].shape)
                    bindings[name] = bindings[name]._replace(shape=blob[name].shape)
                    
                # Copy data to preallocated buffer
                bindings[name].data.copy_(blob[name])
    
    def _copy_outputs_from_device(self, context_id):
        """Copy output tensors from device memory to host.
        
        Args:
            context_id: Either 1 or 2, indicating which context was used
        
        Returns:
            Dictionary of output tensors
        """
        if self.use_tensor_address_api:
            # For newer TensorRT API
            output_buffers = self.output_buffers1 if context_id == 1 else self.output_buffers2
            outputs = {n: output_buffers[n].clone() for n in self.output_names}
        else:
            # For older TensorRT API
            bindings = self.bindings1 if context_id == 1 else self.bindings2
            outputs = {n: bindings[n].data.clone() for n in self.output_names}
            
        return outputs
    
    def run_inference(self, blob1, blob2):
        """
        Run inference on two inputs in parallel
        
        Args:
            blob1: Dictionary with input tensors for first context
            blob2: Dictionary with input tensors for second context
            
        Returns:
            Tuple of output dictionaries
        """
        # Copy inputs to device memory and set tensor addresses
        self._copy_inputs_to_device(blob1, context_id=1)
        self._copy_inputs_to_device(blob2, context_id=2)
        
        # Execute inference in parallel on separate streams
        if self.use_cuda_graph and self.cuda_graph1 and self.cuda_graph2:
            self.cuda_graph1.launch(self.stream1)
            self.cuda_graph2.launch(self.stream2)
        else:
            # Use the version-agnostic execute method
            self.execute_context(self.context1, context_id=1, stream=self.stream1)
            self.execute_context(self.context2, context_id=2, stream=self.stream2)
        
        # Synchronize streams to ensure inference completion
        self.stream1.synchronize()
        self.stream2.synchronize()
        
        # Copy outputs from device to host
        outputs1 = self._copy_outputs_from_device(context_id=1)
        outputs2 = self._copy_outputs_from_device(context_id=2)
        
        return outputs1, outputs2
    
    def __call__(self, blob1, blob2):
        """Shorthand for running inference on two inputs in parallel."""
        return self.run_inference(blob1, blob2)

def get_image_id(img_name):
    """Extract image ID from filename using the specified format."""
    img_name = img_name.split('.png')[0]
    scene_list = ['M', 'A', 'E', 'N']
    camera_idx = int(img_name.split('_')[0].split('camera')[1])
    scene_idx = scene_list.index(img_name.split('_')[1])
    frame_idx = int(img_name.split('_')[2])
    image_id = int(str(camera_idx) + str(scene_idx) + str(frame_idx))
    return image_id

def process_detections(output, image_name, score_threshold=0.4):
    """Process model output and format detections."""
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
        detection = {
            "image_id": image_id, 
            "category_id": int(labels[i]), 
            "bbox": [float(x1), float(y1), float(width), float(height)], 
            "score": float(scores[i])
        }
        detections.append(detection)
    return detections

def preprocess_image(image_path, device):
    """Preprocess an image for model input."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise IOError(f"Could not read image: {image_path}")
    
    h, w, _ = img_bgr.shape
    orig_size = torch.tensor([w, h])[None].to(device)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (960, 960), interpolation=cv2.INTER_AREA)
    im_data = torch.from_numpy(img_resized).permute(2, 0, 1).float().div(255)
    im_data = im_data.unsqueeze(0).contiguous()
    blob = {"images": im_data.to(device), "orig_target_sizes": orig_size.to(device)}
    return blob

def warmup_gpu(model, device, warmup_iterations=10):
    """Warmup the GPU by running parallel inference on dummy data."""
    print(f"Warming up GPU with {warmup_iterations} iterations...")
    
    # Create dummy input data matching expected input shape
    dummy_img1 = torch.randn(1, 3, 960, 960).to(device)
    dummy_orig_size1 = torch.tensor([[960, 960]]).to(device)
    dummy_blob1 = {"images": dummy_img1, "orig_target_sizes": dummy_orig_size1}
    
    dummy_img2 = torch.randn(1, 3, 960, 960).to(device)
    dummy_orig_size2 = torch.tensor([[960, 960]]).to(device)
    dummy_blob2 = {"images": dummy_img2, "orig_target_sizes": dummy_orig_size2}
    
    # Run warmup iterations
    warmup_start = time.time()
    for i in range(warmup_iterations):
        try:
            with torch.no_grad():
                _ = model(dummy_blob1, dummy_blob2)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if (i + 1) % 20 == 0:
                print(f"Warmup iteration {i + 1}/{warmup_iterations}")
        except Exception as e:
            print(f"Warning: Warmup iteration {i} failed: {e}")
            # Continue with next iteration
    
    warmup_end = time.time()
    print(f"GPU warmup completed in {warmup_end - warmup_start:.2f} seconds")
    print("=" * 50)

def process_images_parallel(model, input_folder, output_file, device, score_threshold, skip_warmup=False, warmup_iterations=10):
    """Process images in parallel using two model instances."""
    # GPU Warmup
    if not skip_warmup:
        warmup_gpu(model, device, warmup_iterations)
    
    # Find all image files
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
    image_files.sort()
    print(f"Found {len(image_files)} images to process")
    
    # Process images in pairs
    all_detections = []
    total_preprocess_time = 0
    total_inference_time = 0
    total_postprocess_time = 0
    processed_pairs = 0
    
    timer_start = time.time()
    
    # Process images in pairs
    i = 0
    while i < len(image_files):
        # Get a pair of images (or single image if at the end)
        image_path1 = image_files[i]
        image_name1 = os.path.basename(image_path1)
        
        if i + 1 < len(image_files):
            image_path2 = image_files[i + 1]
            image_name2 = os.path.basename(image_path2)
            pair_size = 2
        else:
            # Handle the case of an odd number of images
            image_path2 = image_path1  # Just duplicate the last image
            image_name2 = image_name1
            pair_size = 1
        
        # Preprocessing (timed)
        preprocess_start = time.perf_counter()
        try:
            blob1 = preprocess_image(image_path1, device)
            blob2 = preprocess_image(image_path2, device)
        except Exception as e:
            print(f"Error preprocessing images: {e}")
            i += pair_size
            continue
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        preprocess_end = time.perf_counter()
        total_preprocess_time += (preprocess_end - preprocess_start)
        
        # Inference (timed) - This is the critical section for parallel execution
        inference_start = time.perf_counter()
        try:
            with torch.no_grad():
                output1, output2 = model(blob1, blob2)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_end = time.perf_counter()
            total_inference_time += (inference_end - inference_start)
            
            # Postprocessing (timed)
            postprocess_start = time.perf_counter()
            detections1 = process_detections(output1, image_name1, score_threshold=score_threshold)
            all_detections.extend(detections1)
            
            # Only process the second output if it's a real image (not a duplicate)
            if pair_size == 2:
                detections2 = process_detections(output2, image_name2, score_threshold=score_threshold)
                all_detections.extend(detections2)
                
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            postprocess_end = time.perf_counter()
            total_postprocess_time += (postprocess_end - postprocess_start)
            
            processed_pairs += 1
        except Exception as e:
            print(f"Error during inference/postprocessing: {e}")
            
        i += pair_size
        
        if processed_pairs % 50 == 0 and processed_pairs > 0:
            print(f"Processed {i}/{len(image_files)} images ({processed_pairs} pairs)")
            # Print intermediate statistics
            print(f"  Avg Inference Time (pair): {(total_inference_time / processed_pairs) * 1000:.2f} ms")
            current_fps = (i / total_inference_time) if total_inference_time > 0 else 0
            print(f"  Current FPS: {current_fps:.2f}")
    
    timer_end = time.time()
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(all_detections, f, indent=2)
    
    # Calculate and display results
    elapsed_time = timer_end - timer_start
    total_processing_time = total_preprocess_time + total_inference_time + total_postprocess_time
    processed_images = i if pair_size == 2 else i - 1  # Adjust for odd number case
    
    # Calculate true parallel FPS (images/second)
    parallel_fps = processed_images / total_inference_time if total_inference_time > 0 else 0
    sequential_equivalent_fps = processed_images / (2 * total_inference_time) if total_inference_time > 0 else 0
    speedup_factor = parallel_fps / sequential_equivalent_fps if sequential_equivalent_fps > 0 else 0
    
    print("\n" + "="*60)
    print("PARALLEL INFERENCE PERFORMANCE REPORT")
    print("="*60)
    print(f"Processed {processed_images} images in {elapsed_time:.2f} seconds.")
    print(f"Total pairs processed: {processed_pairs}")
    
    if processed_pairs > 0:
        print(f"\nTIMING BREAKDOWN (per pair):")
        print(f"  Preprocessing:   {(total_preprocess_time / processed_pairs) * 1000:.2f} ms")
        print(f"  Parallel Inference: {(total_inference_time / processed_pairs) * 1000:.2f} ms")
        print(f"  Postprocessing:  {(total_postprocess_time / processed_pairs) * 1000:.2f} ms")
        print(f"  Total Processing:    {(total_processing_time / processed_pairs) * 1000:.2f} ms")
    
    print(f"\nPERFORMANCE METRICS:")
    print(f"  Parallel FPS:         {parallel_fps:.2f} images/sec")
    print(f"  Sequential Equivalent: {sequential_equivalent_fps:.2f} images/sec")
    print(f"  Parallelization Speedup: {speedup_factor:.2f}x")
    
    print(f"\nRESULTS:")
    print(f"  Output saved to: {output_file}")
    print(f"  Total detections: {len(all_detections)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI City Challenge 2025 Track 4 Parallel Inference")
    parser.add_argument("-trt", "--trt", required=True, help="Path to TensorRT engine file")
    parser.add_argument("-i", "--input", required=True, help="Path to input folder containing test images")
    parser.add_argument("-o", "--output", default="submission.json", help="Output JSON file path")
    parser.add_argument("-d", "--device", default="cuda:0", help="Device to use for inference")
    parser.add_argument("--score-threshold", type=float, default=0.0001, help="Score threshold for filtering detections")
    parser.add_argument('--warmup_iterations', type=int, default=200, help='Number of GPU warmup iterations')
    parser.add_argument('--skip_warmup', action='store_true', help='Skip GPU warmup')
    parser.add_argument('--use_cuda_graph', action='store_true', help='Use CUDA Graph for inference (if supported)')
    args = parser.parse_args()
    
    if not os.path.exists(args.trt):
        exit(f"Error: TensorRT engine file not found: {args.trt}")
    if not os.path.exists(args.input):
        exit(f"Error: Input folder not found: {args.input}")
    
    print("Initializing Parallel TensorRT model...")
    model = ParallelTRTInference(args.trt, device=args.device, max_batch_size=1, use_cuda_graph=args.use_cuda_graph)
    print("Starting parallel image processing...")
    process_images_parallel(model, args.input, args.output, args.device, args.score_threshold, 
                         args.skip_warmup, args.warmup_iterations)