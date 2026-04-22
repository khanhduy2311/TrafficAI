"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
Modified for AI City Challenge 2025 Track 4

FINAL ARCHITECTURE: Fully C++ accelerated, config-driven, multi-threaded pipeline with Cross-Class NMS.
- Loads a JSON config file to drive per-class, per-time-of-day ensembling.
- Python orchestrates the logic, including a final Cross-Class NMS step.
- C++ extensions for pre-processing and WBF act as high-speed workhorses.
- Model ordering for parameters is made explicit with constants to ensure correctness.
"""
import argparse
import collections
import os
import time
import json
import glob
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

if not hasattr(np, "bool"): np.bool = bool
import tensorrt as trt
import cv2
from ultralytics import YOLO

try:
    from preproc import dfine_preproc_cpp
except ImportError:
    print("\n--- FATAL ERROR: Could not import 'dfine_preproc_cpp'. Build it first. ---")
    exit(1)
try:
    from fusion import wbf_cpp
except ImportError:
    print("\n--- FATAL ERROR: Could not import 'wbf_cpp'. Build it first. ---")
    exit(1)

try:
    from utils import f1_score
except ImportError:
    print("\n--- WARNING: 'utils.py' not found. F1 score calculation will be skipped. ---")
    f1_score = None

# --- Explicitly define the model order for all parameter lists ---
# This order MUST match the order of data in the `all_boxes`, `all_scores`, etc. lists below.
DFINE_IDX = 0
YOLO_IDX = 1
# ----------------------------------------------------------------------

class TRTInference:
    def __init__(self, engine_path, device="cuda:0", max_batch_size=1):
        self.device = torch.device(device)
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime: self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        bindings = OrderedDict()
        Binding = collections.namedtuple("Binding", ("name", "dtype", "shape", "data"))
        for i, name in enumerate(self.engine):
            shape = list(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            if shape and shape[0] == -1: shape[0] = max_batch_size
            data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(self.device)
            bindings[name] = Binding(name, dtype, shape, data)
        self.bindings = bindings
        self.bindings_addr = OrderedDict((n, v.data.data_ptr()) for n, v in self.bindings.items())
        self.input_names = [n for n in self.engine if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.output_names = [n for n in self.engine if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
    def __call__(self, blob: dict):
        for name in self.input_names: self.bindings[name].data.copy_(blob[name])
        self.context.execute_v2(list(self.bindings_addr.values())); return {name: self.bindings[name].data.clone() for name in self.output_names}

def warmup_gpu(dfine_model, yolo_model, device, warmup_iterations=50):
    print(f"Warming up GPU with {warmup_iterations} iterations..."); ds=str(device); di=np.random.randint(0,255,size=(960,960,3),dtype=np.uint8); ws=time.time()
    for _ in range(warmup_iterations):
        with torch.no_grad():
            db=dfine_preproc_cpp.preprocess(di,ds);_=dfine_model(db);_=yolo_model.predict(di,device=device,verbose=False)
    torch.cuda.synchronize();we=time.time();print(f"GPU warmup completed in {we-ws:.2f} seconds");print("="*50)

def dfine_cpp_preprocess_thread_func(img_bgr, device_str, result_container): result_container['dfine_blob']=dfine_preproc_cpp.preprocess(img_bgr,device_str)
def get_image_id(img_name): n=os.path.basename(img_name).split('.png')[0];s=['M','A','E','N'];p=n.split('_');c=int(p[0].replace('camera',''));si=s.index(p[1]);f=int(p[2]);return int(f"{c}{si}{f}")
def clip_boxes(b,w,h):
    if b.size==0:return b
    b[:,[0,2]]=np.clip(b[:,[0,2]],0,w);b[:,[1,3]]=np.clip(b[:,[1,3]],0,h);return b
def extract_predictions(o,w,h,is_yolo=True):
    if is_yolo:r=o[0].boxes;b,s,l=r.xyxy.cpu().numpy(),r.conf.cpu().numpy(),r.cls.cpu().numpy()
    else:b,s,l=o["boxes"][0].cpu().numpy(),o["scores"][0].cpu().numpy(),o["labels"][0].cpu().numpy()
    return clip_boxes(b,w,h),s,l
def format_fused_detections_fast(b,s,l,img_id):
    w=b[:,2]-b[:,0];h=b[:,3]-b[:,1]
    return[{"image_id":img_id,"category_id":int(la),"bbox":[float(bo[0]),float(bo[1]),float(wi),float(he)],"score":float(sc)} for bo,sc,la,wi,he in zip(b,s,l,w,h)]
def cross_class_nms(dets,iou_thresh=0.8):
    if not dets:return[]
    dets.sort(key=lambda x:x['score'],reverse=True);k=[True]*len(dets)
    for i in range(len(dets)):
        if not k[i]:continue
        b1=dets[i]['bbox'];b1x2,b1y2=b1[0]+b1[2],b1[1]+b1[3]
        for j in range(i+1,len(dets)):
            if not k[j] or dets[i]['category_id']==dets[j]['category_id']:continue
            b2=dets[j]['bbox'];b2x2,b2y2=b2[0]+b2[2],b2[1]+b2[3]
            ix1=max(b1[0],b2[0]);iy1=max(b1[1],b2[1]);ix2=min(b1x2,b2x2);iy2=min(b1y2,b2y2)
            if ix2>ix1 and iy2>iy1:
                ia=(ix2-ix1)*(iy2-iy1);ua=b1[2]*b1[3]+b2[2]*b2[3]-ia
                if ia/(ua+1e-9)>iou_thresh:k[j]=False
    return[d for i,d in enumerate(dets) if k[i]]

def process_images(dfine_model, yolo_model, input_folder, output_file, device, args, params_config):
    if not args.skip_warmup: warmup_gpu(dfine_model, yolo_model, device, args.warmup_iterations)
    image_files = sorted(glob.glob(os.path.join(input_folder, '*.png'))); print(f"Found {len(image_files)} images to process")
    all_detections, total_imread_time, total_parallel_time, total_dfine_time, total_postprocess_time = [], 0, 0, 0, 0
    processed_images = 0
    overall_timer_start = time.time()

    with ThreadPoolExecutor(max_workers=1) as executor:
        for i, image_path in enumerate(image_files):
            imread_start = time.time(); img_bgr = cv2.imread(image_path)
            if img_bgr is None: continue
            h, w, _ = img_bgr.shape; total_imread_time += (time.time() - imread_start)

            parallel_start = time.time(); thread_result = {}; future = executor.submit(dfine_cpp_preprocess_thread_func, img_bgr, str(device), thread_result)
            yolo_output = yolo_model.predict(img_bgr, device=device, verbose=False, imgsz=960); future.result()
            dfine_blob = thread_result['dfine_blob']; torch.cuda.synchronize(); total_parallel_time += (time.time() - parallel_start)

            dfine_start = time.time(); dfine_output = dfine_model(dfine_blob); torch.cuda.synchronize(); total_dfine_time += (time.time() - dfine_start)

            postprocess_start = time.time()
            yolo_boxes_raw, yolo_scores_raw, yolo_labels_raw = extract_predictions(yolo_output, w, h, is_yolo=True)
            dfine_boxes_raw, dfine_scores_raw, dfine_labels_raw = extract_predictions(dfine_output, w, h, is_yolo=False)

            filename = os.path.basename(image_path)
            time_of_day = "night" if ("_N_" in filename or "_E_" in filename) else "day"
            tod_params_config = params_config[time_of_day]
            per_class_params = tod_params_config['per_class_params']
            cross_class_nms_iou = tod_params_config.get('cross_class_nms_iou', 0.8)

            image_detections_before_nms = []
            inv_dims = np.array([1/w, 1/h, 1/w, 1/h], dtype=np.float64)

            for class_id_str, class_params_data in per_class_params.items():
                class_id = int(class_id_str)
                params = class_params_data['params']
                
                # --- THIS IS THE NEW, CORRECT LOGIC ---
                # 1. Filter each model's raw detections
                dfine_conf_thr = params['model_confidences'][DFINE_IDX]
                yolo_conf_thr = params['model_confidences'][YOLO_IDX]
                
                dfine_mask = (dfine_labels_raw == class_id) & (dfine_scores_raw >= dfine_conf_thr)
                yolo_mask = (yolo_labels_raw == class_id) & (yolo_scores_raw >= yolo_conf_thr)

                dfine_boxes_filt, dfine_scores_filt = dfine_boxes_raw[dfine_mask], dfine_scores_raw[dfine_mask]
                yolo_boxes_filt, yolo_scores_filt = yolo_boxes_raw[yolo_mask], yolo_scores_raw[yolo_mask]

                # 2. Create the flattened list of internal box representations
                # This mirrors the logic of `prefilter_boxes`
                flat_box_list = []
                # Add D-FINE boxes (Model 0)
                for box, score in zip(dfine_boxes_filt, dfine_scores_filt):
                    norm_box = box * inv_dims
                    flat_box_list.append((class_id, score * params['weights'][DFINE_IDX], params['weights'][DFINE_IDX], DFINE_IDX, norm_box[0], norm_box[1], norm_box[2], norm_box[3]))
                # Add YOLO boxes (Model 1)
                for box, score in zip(yolo_boxes_filt, yolo_scores_filt):
                    norm_box = box * inv_dims
                    flat_box_list.append((class_id, score * params['weights'][YOLO_IDX], params['weights'][YOLO_IDX], YOLO_IDX, norm_box[0], norm_box[1], norm_box[2], norm_box[3]))

                if not flat_box_list:
                    continue

                # 3. Sort the flattened list by score, exactly like the original script
                flat_box_list.sort(key=lambda x: x[1], reverse=True)

                # 4. Call the new, simpler C++ function
                fused_boxes, fused_scores, fused_labels = wbf_cpp.fusion(
                    box_list=flat_box_list,
                    iou_thr=params['iou_thr'],
                    weights=params['weights']
                )
                # --- END NEW LOGIC ---

                if fused_boxes.size > 0:
                    dims = np.array([w, h, w, h])
                    fused_boxes_pixel = fused_boxes * dims
                    image_id = get_image_id(image_path)
                    detections_for_class = format_fused_detections_fast(fused_boxes_pixel, fused_scores, fused_labels, image_id)
                    image_detections_before_nms.extend(detections_for_class)
            
            final_image_detections = cross_class_nms(image_detections_before_nms, cross_class_nms_iou)
            all_detections.extend(final_image_detections)

            total_postprocess_time += (time.time() - postprocess_start)
            processed_images += 1
            if (i + 1) % 100 == 0: print(f"Processed {processed_images}/{len(image_files)} images")

    overall_timer_end = time.time(); elapsed_time = overall_timer_end - overall_timer_start
    with open(output_file, 'w') as f: json.dump(all_detections, f); print(f"\nResults saved to: {output_file}")
    print("\n" + "="*50); print("           PERFORMANCE METRICS"); print("="*50); print(f"Processed {processed_images} images.")
    if processed_images > 0:
        total_processing_time = total_parallel_time + total_dfine_time + total_postprocess_time
        wall_clock_fps = processed_images / elapsed_time if elapsed_time > 0 else float('inf')
        processing_fps = processed_images / total_processing_time if total_processing_time > 0 else float('inf')
        print(f"Total Wall-Clock Time        : {elapsed_time:.2f} seconds"); print(f"Total Core Processing Time   : {total_processing_time:.2f} seconds")
        print(f"Wall-Clock FPS (incl. I/O)   : {wall_clock_fps:.2f}"); print(f"Processing FPS (for scoring) : {processing_fps:.2f}"); print("-" * 50)
        print(f"Avg Image Load Time          : {(total_imread_time / processed_images) * 1000:.2f} ms")
        print(f"Avg Parallel Block           : {(total_parallel_time / processed_images) * 1000:.2f} ms")
        print(f"Avg D-FINE Inference Time    : {(total_dfine_time / processed_images) * 1000:.2f} ms")
        print(f"Avg Post-processing          : {(total_postprocess_time / processed_images) * 1000:.2f} ms")
    print("\n" + "="*50); print("           OFFICIAL METRICS"); print("="*50)
    normfps = min(processing_fps, args.max_fps) / args.max_fps; print(f"Normalized FPS (max_fps={args.max_fps}): {normfps:.4f}")
    if args.ground_truths_path and f1_score:
        if os.path.exists(args.ground_truths_path):
            f1 = f1_score(output_file, args.ground_truths_path)
            harmonic_mean = 2 * f1 * normfps / (f1 + normfps + 1e-9)
            print(f"F1-score                     : {f1:.4f}"); print("-" * 50); print(f"Final Metric (Harmonic Mean) : {harmonic_mean:.4f}")
        else: print(f"F1-score                     : SKIPPED (Ground truth file not found at '{args.ground_truths_path}')")
    else: print("F1-score                     : SKIPPED (No ground truth path provided or utils.py missing)")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI City Challenge 2025 C++ Accelerated Inference")
    parser.add_argument('--params-config', type=str, default='./ensemble_configs.json', help='Path to the JSON file with parameters.')
    parser.add_argument("--dfine-model", default="./models/D-FINE-L.engine")
    parser.add_argument("--yolo-model", default="./models/11m_pseudo_1_3_cross_nms.engine")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", default="submission.json")
    parser.add_argument("-d", "--device", default="cuda")
    parser.add_argument('--warmup-iterations', type=int, default=100)
    parser.add_argument('--skip-warmup', action='store_true')
    parser.add_argument('--ground-truths-path', type=str, default=None, help='Path to ground truths JSON file for F1 score calculation.')
    parser.add_argument('--max-fps', type=float, default=25.0, help='Maximum FPS for normalization in the final metric.')
    args = parser.parse_args()

    try:
        with open(args.params_config, 'r') as f:
            params_config = json.load(f)['per_class_optimal_parameters']
        print(f"Successfully loaded ensemble parameters from '{args.params_config}'")
    except Exception as e:
        print(f"--- FATAL ERROR: Could not load or parse the params config file. Error: {e} ---"); exit(1)

    print("Initializing models..."); dfine_model = TRTInference(args.dfine_model, device=args.device); yolo_model = YOLO(args.yolo_model, task='detect')
    print("Starting processing with full C++ acceleration and per-class fusion..."); print("=" * 50)
    process_images(dfine_model, yolo_model, args.input, args.output, args.device, args, params_config)
