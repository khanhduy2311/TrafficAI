"""
A self-contained script for visualizing the output of the final, config-driven
ensemble pipeline on a single image for debugging and analysis.

THIS SCRIPT IS A PERFECT LOGICAL TWIN OF `final_inference.py`.

It uses PURE PYTHON implementations for pre-processing and ensembling
to mirror the logic of the main pipeline without relying on the C++ extensions,
making it a portable and reliable debugging tool.
"""
import argparse
import collections
import json
import os
from collections import OrderedDict

import cv2
import numpy as np
import tensorrt as trt
import torch
from ensemble_boxes import weighted_boxes_fusion # Used as the Python equivalent of our C++ WBF
from ultralytics import YOLO

# --- Configuration for Visualization ---
CLASS_NAMES = ["Bus", "Bike", "Car", "Pedestrian", "Truck"]
COLORS = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100)]
DFINE_IDX = 0
YOLO_IDX = 1

# --- Helper Classes and Functions (Copied from main pipeline) ---
if not hasattr(np, "bool"): np.bool = bool

class TRTInference:
    def __init__(self, engine_path, device="cuda:0", max_batch_size=1):
        self.device=torch.device(device);self.logger=trt.Logger(trt.Logger.WARNING);trt.init_libnvinfer_plugins(self.logger,"");
        with open(engine_path,"rb") as f, trt.Runtime(self.logger) as runtime: self.engine=runtime.deserialize_cuda_engine(f.read())
        self.context=self.engine.create_execution_context();bindings=OrderedDict();Binding=collections.namedtuple("Binding",("name","dtype","shape","data"))
        for i,name in enumerate(self.engine):
            shape=list(self.engine.get_tensor_shape(name));dtype=trt.nptype(self.engine.get_tensor_dtype(name))
            if shape and shape[0]==-1: shape[0]=max_batch_size
            data=torch.from_numpy(np.empty(shape,dtype=dtype)).to(self.device);bindings[name]=Binding(name,dtype,shape,data)
        self.bindings=bindings;self.bindings_addr=OrderedDict((n,v.data.data_ptr()) for n,v in self.bindings.items())
        self.input_names=[n for n in self.engine if self.engine.get_tensor_mode(n)==trt.TensorIOMode.INPUT]
        self.output_names=[n for n in self.engine if self.engine.get_tensor_mode(n)==trt.TensorIOMode.OUTPUT]
    def __call__(self,blob:dict):
        for name in self.input_names: self.bindings[name].data.copy_(blob[name])
        self.context.execute_v2(list(self.bindings_addr.values()));return {name:self.bindings[name].data.clone() for name in self.output_names}

def dfine_preprocess_python(img_bgr, device_str="cuda:0"):
    h, w, _ = img_bgr.shape; device = torch.device(device_str)
    orig_size = torch.tensor([[w, h]], device=device)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (960, 960), interpolation=cv2.INTER_AREA)
    im_tensor = torch.from_numpy(img_resized).to(device).float().permute(2,0,1).div(255.0).unsqueeze(0)
    return {"images": im_tensor, "orig_target_sizes": orig_size}

def clip_boxes(b, w, h):
    if b.size==0: return b
    b[:,[0,2]] = np.clip(b[:,[0,2]], 0, w); b[:,[1,3]] = np.clip(b[:,[1,3]], 0, h); return b

def extract_predictions(o, w, h, is_yolo=True):
    if is_yolo: r=o[0].boxes; b,s,l=r.xyxy.cpu().numpy(),r.conf.cpu().numpy(),r.cls.cpu().numpy()
    else: b,s,l=o["boxes"][0].cpu().numpy(),o["scores"][0].cpu().numpy(),o["labels"][0].cpu().numpy()
    return clip_boxes(b,w,h), s, l

def cross_class_nms(dets, iou_thresh=0.8):
    if not dets: return []
    dets.sort(key=lambda x:x['score'], reverse=True); k=[True]*len(dets)
    for i in range(len(dets)):
        if not k[i]: continue
        b1=dets[i]['bbox']; b1x2,b1y2 = b1[0]+b1[2], b1[1]+b1[3]
        for j in range(i+1, len(dets)):
            if not k[j] or dets[i]['category_id']==dets[j]['category_id']: continue
            b2=dets[j]['bbox']; b2x2,b2y2 = b2[0]+b2[2], b2[1]+b2[3]
            ix1=max(b1[0],b2[0]); iy1=max(b1[1],b2[1]); ix2=min(b1x2,b2x2); iy2=min(b1y2,b2y2)
            if ix2>ix1 and iy2>iy1:
                ia=(ix2-ix1)*(iy2-iy1); ua=b1[2]*b1[3]+b2[2]*b2[3]-ia
                if ia/(ua+1e-9) > iou_thresh: k[j]=False
    return [d for i,d in enumerate(dets) if k[i]]

def draw_detections(image, boxes, scores, labels):
    for box, score, label in zip(boxes, scores, labels):
        label = int(label); color = COLORS[label % len(COLORS)]
        class_name = CLASS_NAMES[label] if 0 <= label < len(CLASS_NAMES) else f"L{label}"
        x1, y1, x2, y2 = map(int, box); cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text = f"{class_name} {score:.2f}"
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x1, y1 - th - bl), (x1 + tw, y1), color, -1)
        cv2.putText(image, text, (x1, y1 - bl), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return image

def visualize_image(dfine_model, yolo_model, image_path, output_path, params_config, device):
    print("Loading image..."); img_bgr = cv2.imread(image_path)
    if img_bgr is None: print(f"Error: Could not load image from {image_path}"); return
    h, w, _ = img_bgr.shape
    
    # --- 1. Run Inference ---
    print("Running D-FINE inference...")
    dfine_blob = dfine_preprocess_python(img_bgr, device)
    dfine_output = dfine_model(dfine_blob)
    dfine_boxes_raw, dfine_scores_raw, dfine_labels_raw = extract_predictions(dfine_output, w, h, is_yolo=False)

    print("Running YOLO inference...");
    yolo_output = yolo_model.predict(img_bgr, device=device, verbose=False, imgsz=960)
    yolo_boxes_raw, yolo_scores_raw, yolo_labels_raw = extract_predictions(yolo_output, w, h, is_yolo=True)
    
    # --- 2. Perform Config-driven Ensemble (LOGIC IDENTICAL TO final_inference.py) ---
    print("Performing per-class ensemble...")
    filename = os.path.basename(image_path)
    time_of_day = "night" if ("_N_" in filename or "_E_" in filename) else "day"
    tod_params = params_config[time_of_day]
    per_class_params = tod_params['per_class_params']
    cross_class_nms_iou = tod_params.get('cross_class_nms_iou', 0.8)

    image_detections_before_nms = []
    inv_dims = np.array([1/w, 1/h, 1/w, 1/h], dtype=np.float64)

    for class_id_str, class_params_data in per_class_params.items():
        class_id = int(class_id_str)
        params = class_params_data['params']
        
        dfine_conf_thr = params['model_confidences'][DFINE_IDX]
        yolo_conf_thr = params['model_confidences'][YOLO_IDX]
        
        dfine_mask = (dfine_labels_raw == class_id) & (dfine_scores_raw >= dfine_conf_thr)
        yolo_mask = (yolo_labels_raw == class_id) & (yolo_scores_raw >= yolo_conf_thr)

        dfine_boxes_filt, dfine_scores_filt = dfine_boxes_raw[dfine_mask], dfine_scores_raw[dfine_mask]
        yolo_boxes_filt, yolo_scores_filt = yolo_boxes_raw[yolo_mask], yolo_scores_raw[yolo_mask]

        # In this Python version, we must construct the lists that the `ensemble-boxes` library expects
        boxes_list = [dfine_boxes_filt * inv_dims, yolo_boxes_filt * inv_dims]
        scores_list = [dfine_scores_filt, yolo_scores_filt]
        labels_list = [np.full_like(dfine_scores_filt, class_id), np.full_like(yolo_scores_filt, class_id)]
        
        # Skip if no boxes for this class from either model
        if dfine_boxes_filt.size == 0 and yolo_boxes_filt.size == 0:
            continue

        # Call the standard, validated Python/Numba fusion function
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            iou_thr=params['iou_thr'],
            skip_box_thr=0.0, # Filtering is already done
            weights=params['weights']
        )
        
        if fused_boxes.size > 0:
            dims = np.array([w, h, w, h])
            fused_boxes_pixel = fused_boxes * dims
            image_detections_before_nms.append((fused_boxes_pixel, fused_scores, fused_labels))

    # --- 3. Combine and Apply Cross-Class NMS ---
    # This step is now IDENTICAL to the main script
    if image_detections_before_nms:
        all_boxes_before_nms = np.vstack([d[0] for d in image_detections_before_nms])
        all_scores_before_nms = np.hstack([d[1] for d in image_detections_before_nms])
        all_labels_before_nms = np.hstack([d[2] for d in image_detections_before_nms])
        
        # We need to convert to the list-of-dicts format for the cross_class_nms function
        temp_dets_for_nms = []
        for box, score, label in zip(all_boxes_before_nms, all_scores_before_nms, all_labels_before_nms):
            width, height = box[2] - box[0], box[3] - box[1]
            temp_dets_for_nms.append({
                "score": score, "category_id": int(label),
                "bbox": [box[0], box[1], width, height]
            })

        print(f"Applying Cross-Class NMS to {len(temp_dets_for_nms)} detections...")
        final_detections_dict = cross_class_nms(temp_dets_for_nms, cross_class_nms_iou)

        # Convert back from dicts to numpy arrays for drawing
        final_boxes = np.array([[d['bbox'][0], d['bbox'][1], d['bbox'][0]+d['bbox'][2], d['bbox'][1]+d['bbox'][3]] for d in final_detections_dict])
        final_scores = np.array([d['score'] for d in final_detections_dict])
        final_labels = np.array([d['category_id'] for d in final_detections_dict])
    else:
        final_boxes, final_scores, final_labels = np.array([]), np.array([]), np.array([])
        
    print(f"Found {len(final_boxes)} final detections after ensemble and NMS.")
    
    # --- 4. Visualize and Save ---
    print("Drawing final detections on the image...")
    output_image = draw_detections(img_bgr.copy(), final_boxes, final_scores, final_labels)
    
    cv2.imwrite(output_path, output_image)
    print(f"\n✅ Success! Visualization saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize ensemble predictions on a single image.")
    parser.add_argument('--input-image', '-i', type=str, required=True)
    parser.add_argument('--output-image', '-o', type=str, required=True)
    parser.add_argument('--params-config', type=str, required=True)
    parser.add_argument("--dfine-model", required=True)
    parser.add_argument("--yolo-model", required=True)
    parser.add_argument("-d", "--device", default="cuda")
    args = parser.parse_args()

    try:
        with open(args.params_config, 'r') as f:
            params_config = json.load(f)['per_class_optimal_parameters']
        print(f"Successfully loaded ensemble parameters from '{args.params_config}'")
    except Exception as e:
        print(f"--- FATAL ERROR: Could not load or parse the params config file. Error: {e} ---"); exit(1)

    print("Initializing models...");
    dfine_model = TRTInference(args.dfine_model, device=args.device)
    yolo_model = YOLO(args.yolo_model, task='detect')
    
    print("=" * 50)
    visualize_image(dfine_model, yolo_model, args.input_image, args.output_image, params_config, args.device)