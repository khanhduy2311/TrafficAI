import tensorrt as trt
import numpy as np
import os
import pycuda.driver as cuda 
import pycuda.autoinit # Important: Initializes CUDA context

# Calibrator for INT8.
class BasicINT8Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_image_paths_or_dummy, batch_size, input_name, input_resolution_hw=(640, 640)):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = "int8_calibration.cache" # For TensorRT to store calibration scales
        self.batch_size = batch_size
        self.input_name = input_name 
        self.input_resolution_hw = input_resolution_hw # H, W
        self.current_index = 0
        self.device_input_ptr = None # Will hold PyCUDA device allocation pointer

        # Determine if we are using dummy data or real data paths
        self.use_dummy_data = not calibration_image_paths_or_dummy or not os.path.exists(calibration_image_paths_or_dummy[0])

        if self.use_dummy_data:
            print("BasicINT8Calibrator: Using DUMMY random data for calibration.")
            self.num_calibration_batches = 10 
            self.image_data_np = [] # List to hold numpy arrays if needed for dummy data
            for _ in range(self.num_calibration_batches):
                batch_data = np.random.rand(self.batch_size, 3, self.input_resolution_hw[0], self.input_resolution_hw[1]).astype(np.float32)
                self.image_data_np.append(batch_data)
        else:
            print(f"BasicINT8Calibrator: Using REAL image paths for calibration. Count: {len(calibration_image_paths_or_dummy)}")
            self.image_paths = calibration_image_paths_or_dummy
            self.num_calibration_batches = (len(self.image_paths) + self.batch_size - 1) // self.batch_size
            if not self.image_paths:
                 raise ValueError("No calibration images provided, and not in dummy mode.")
            # Preprocessing for real images would happen in get_batch

        # Allocate GPU memory for one batch
        # Size in bytes: batch_size * C * H * W * sizeof(float32)
        self.bytes_per_batch = self.batch_size * 3 * self.input_resolution_hw[0] * self.input_resolution_hw[1] * np.dtype(np.float32).itemsize
        self.device_input_ptr = cuda.mem_alloc(self.bytes_per_batch)


    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names): # names is a list of binding names (usually just one, the input_name)
        if self.current_index < self.num_calibration_batches:
            if self.use_dummy_data:
                current_batch_np = self.image_data_np[self.current_index]
                # print(f"Calibrator: Providing DUMMY batch {self.current_index + 1}/{self.num_calibration_batches}")
            else:
                # --- REAL IMAGE LOADING AND PREPROCESSING WOULD GO HERE ---
                # 1. Get `self.batch_size` image paths from `self.image_paths`
                # 2. Load them (e.g., using PIL or OpenCV)
                # 3. Preprocess: resize to self.input_resolution_hw, normalize to [0,1], CHW, float32
                # 4. Stack them into a numpy array: `current_batch_np`
                # Example placeholder for real data:
                print(f"Calibrator: Providing REAL batch {self.current_index + 1}/{self.num_calibration_batches}. Implement image loading!")
                # For now, to make it runnable, use random data if real path is chosen but loading not impl.
                current_batch_np = np.random.rand(self.batch_size, 3, self.input_resolution_hw[0], self.input_resolution_hw[1]).astype(np.float32)
                # --- END OF REAL IMAGE LOADING PLACEHOLDER ---

            # Copy data from host (NumPy array) to the pre-allocated GPU buffer
            cuda.memcpy_htod(self.device_input_ptr, current_batch_np)
            
            self.current_index += 1
            # Return a list containing the integer address of the GPU buffer
            return [int(self.device_input_ptr)] 
        else:
            # No more batches
            if self.device_input_ptr: # Free GPU memory when done (optional here, TRT might manage)
                # self.device_input_ptr.free() # Be careful with freeing if TRT expects it to persist
                # print("Calibrator: Freed device input buffer.")
                pass 
            return None 

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f"Calibrator: Reading calibration cache from {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        print("Calibrator: Calibration cache not found.")
        return None

    def write_calibration_cache(self, cache):
        print(f"Calibrator: Writing calibration cache to {self.cache_file}")
        with open(self.cache_file, "wb") as f:
            f.write(cache)

    # It's good practice to free GPU memory if you allocated it, 
    # though the calibrator object might be short-lived.
    def __del__(self):
        if self.device_input_ptr:
            try:
                self.device_input_ptr.free()
                # print("BasicINT8Calibrator: Freed GPU memory in __del__")
            except Exception as e:
                # print(f"BasicINT8Calibrator: Error freeing GPU memory in __del__: {e}")
                pass # Might already be freed or context gone

def convert_onnx_to_trt(onnx_path, engine_path, batch_size=1, precision="fp16", 
                        calibration_image_dir=None, image_input_name="images"):
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE if precision == "int8" else trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    
    EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)
    
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
        print("Successfully parsed ONNX model")

    profile = builder.create_optimization_profile()
    profile.set_shape(
        image_input_name,
        min=(1, 3, 640, 640),
        opt=(batch_size, 3, 640, 640), # Use `batch_size` for opt here for INT8 calibration consistency
        max=(batch_size, 3, 640, 640)  # Or a larger max if needed for inference
    )
    profile.set_shape(
        "orig_target_sizes",
        min=(1, 2),
        opt=(batch_size, 2),
        max=(batch_size, 2)
    )

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1GB workspace (reduced from 2GB for Jetson)
    if not config.add_optimization_profile(profile):
        print("ERROR: Failed to add optimization profile to builder config.")
        return None
    print("Successfully added optimization profile.")


    if precision == "fp16":
        print("Using FP16 precision.")
        if not builder.platform_has_fast_fp16:
            print("Warning: Platform does not have fast FP16. Performance may be suboptimal.")
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        print("Using INT8 precision.")
        if not builder.platform_has_fast_int8:
            print("ERROR: Platform does not support fast INT8.")
            return None
        config.set_flag(trt.BuilderFlag.INT8)
        
        calibration_paths_for_calibrator = []
        if calibration_image_dir and os.path.isdir(calibration_image_dir):
            calibration_paths_for_calibrator = [os.path.join(calibration_image_dir, f) for f in os.listdir(calibration_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not calibration_paths_for_calibrator:
                print(f"Warning: No images found in calibration directory: {calibration_image_dir}. Using DUMMY calibrator.")
            else:
                print(f"Found {len(calibration_paths_for_calibrator)} images for INT8 calibration from {calibration_image_dir}")
        else:
            print("Warning: INT8 precision: calibration_image_dir not provided or invalid. Using DUMMY calibrator.")
            
        # Pass the actual input name of the image tensor from your ONNX model
        calibrator = BasicINT8Calibrator(calibration_paths_for_calibrator, batch_size, image_input_name, input_resolution_hw=(640,640))
        config.int8_calibrator = calibrator
        # It's good practice to set the calibration profile if you have multiple profiles
        # and want to calibrate for a specific one. If only one, it's often inferred.
        # config.set_calibration_profile(profile) 
    else:
        print("Using FP32 precision (default).")


    print("Building TensorRT engine... This may take a while.")
    try:
        # serialized_engine = builder.build_engine(network, config) # Old API
        serialized_engine = builder.build_serialized_network(network, config) # New API
        if serialized_engine is None:
            print("ERROR: Failed to build serialized engine. build_serialized_network returned None.")
            TRT_LOGGER.log(trt.Logger.Severity.ERROR, "Engine serialization failed.")
            return None
    except Exception as e:
        print(f"ERROR: Exception during engine build: {e}")
        return None

    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    print(f"Successfully converted model to TensorRT engine: {engine_path}")
    return engine_path


if __name__ == "__main__":
    onnx_file = "model.onnx" 
    
    if not os.path.exists(onnx_file):
        print(f"ERROR: ONNX file not found: {onnx_file}. Please run export_onnx.py first.")
    else:
        print("\n--- Converting to FP16 Engine ---")
        fp16_engine_path = onnx_file.replace(".onnx", "_fp16.engine")
        convert_onnx_to_trt(
            onnx_path=onnx_file,
            engine_path=fp16_engine_path,
            batch_size=1, 
            precision="fp16",
            image_input_name="images" 
        )

        print("\n--- Converting to INT8 Engine (with DUMMY calibrator or real if path provided) ---")
        int8_engine_path = onnx_file.replace(".onnx", "_int8.engine")
        
        # --- IMPORTANT: SET YOUR CALIBRATION IMAGE DIRECTORY HERE ---
        # Create a folder, e.g., "calibration_images" and put ~100-500 representative images there.
        # If None, it will use the dummy random data calibrator.
        my_calibration_image_directory = None 
        # my_calibration_image_directory = "/workspace/datasets/calibration_subset" # Example path

        convert_onnx_to_trt(
            onnx_path=onnx_file,
            engine_path=int8_engine_path,
            batch_size=1, # Calibrator can use a different batch size if needed, but engine profile uses 1.
            precision="int8",
            calibration_image_dir=my_calibration_image_directory, 
            image_input_name="images" 
        )