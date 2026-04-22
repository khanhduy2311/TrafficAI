"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.

This script is modified to export D-FINE models to ONNX, combining features from
the original D-FINE and DEIM-D-FINE export scripts.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import torch
import torch.nn as nn

from src.core import YAMLConfig


def main(args):
    """main"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]
        cfg.model.load_state_dict(state)
        print(f"Successfully loaded model weights from {args.resume}")
    else:
        # This script requires a checkpoint to export a trained model.
        raise ValueError("--resume argument is required to specify the model checkpoint.")

    # MODIFICATION: Updated Model wrapper for D-FINE's multi-output postprocessor.
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs_raw = self.model(images)
            # The postprocessor returns three separate tensors.
            labels, boxes, scores = self.postprocessor(outputs_raw, orig_target_sizes)
            # Cast labels to int32 for ONNX compatibility.
            labels = labels.to(torch.int32)
            return labels, boxes, scores

    model = Model()
    model.eval()  # Important: set model to evaluation mode

    # MODIFICATION: Updated dynamic_axes for the D-FINE model structure.
    # We define axes for dynamic batch size on inputs. Outputs may have dynamic
    # numbers of detections, which is handled implicitly by ONNX.
    dynamic_axes = {
        "images": {0: "N"},
        "orig_target_sizes": {0: "N"},
    }

    # If not exporting with a dynamic batch, set axes to None.
    if not args.dynamic_batch:
        dynamic_axes = None

    # MODIFICATION: Cleaner and more descriptive output filename logic.
    base_name = os.path.splitext(os.path.basename(args.resume))[0]
    fp16_suffix = "_fp16" if args.fp16 else ""
    batch_suffix = "_dynamic_batch" if args.dynamic_batch else ""
    output_file = f"{base_name}{batch_suffix}{fp16_suffix}.onnx"
    print(f"ONNX model will be saved to: {output_file}")


    # --- Refactored and Simplified Export Logic ---
    device = "cuda" if args.fp16 else "cpu"
    if args.fp16 and not torch.cuda.is_available():
        raise RuntimeError("FP16 export requires a CUDA-enabled GPU.")
    
    model.to(device)

    h, w = args.size
    # Create dummy data on the correct device.
    # The 'size' tensor needs to match the second input of the forward pass.
    data = torch.randn(1, 3, h, w, device=device)
    size = torch.tensor([[h, w]], dtype=torch.int32, device=device)

    # Perform a dry run with no_grad to ensure the model runs.
    with torch.no_grad():
        _ = model(data, size)

    def export_model():
        print(f"Starting ONNX export (Opset 17)...")
        # MODIFICATION: Updated export call with correct inputs/outputs.
        torch.onnx.export(
            model,
            (data, size),
            output_file,
            input_names=["images", "orig_target_sizes"],
            output_names=["labels", "boxes", "scores"],
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
        )
        print("ONNX export completed successfully.")

    if args.fp16:
        with torch.autocast("cuda", dtype=torch.float16):
            export_model()
    else:
        export_model()

    if args.check:
        import onnx
        print("Checking ONNX model...")
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print("ONNX model check passed.")

    if args.simplify:
        # MODIFICATION: Using a more robust simplification method with test inputs.
        try:
            import onnx
            import onnxsim

            print(f"Simplifying ONNX model: {output_file}")
            # Provide input shapes for more reliable simplification.
            input_shapes = {"images": data.shape, "orig_target_sizes": size.shape}
            
            onnx_model_simplify, check = onnxsim.simplify(
                output_file,
                test_input_shapes=input_shapes
            )
            if check:
                onnx.save(onnx_model_simplify, output_file)
                print(f"Successfully simplified ONNX model.")
            else:
                print("ONNX simplification failed or was not needed. Using original model.")
        except Exception as e:
            print(f"An error occurred during ONNX simplification: {e}. Using original model.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export D-FINE model to ONNX format.")
    # MODIFICATION: Updated arguments to match the D-FINE model's needs.
    parser.add_argument(
        "--config", "-c", 
        default="configs/dfine/dfine_hgnetv2_l_coco.yml", 
        type=str,
        help="Path to the model config file."
    )
    parser.add_argument(
        "--resume", "-r", 
        type=str, 
        required=True,
        help="Path to the model checkpoint (.pth) to export."
    )
    parser.add_argument(
        "--size", "-s", 
        nargs=2, 
        default=[960, 960], 
        type=int,
        metavar=('HEIGHT', 'WIDTH'),
        help="Input image size for the export dummy input."
    )
    parser.add_argument(
        "--dynamic_batch", 
        action="store_true",
        help="Export the model with a dynamic batch size."
    )
    parser.add_argument(
        "--fp16", "-f", 
        action="store_true",
        help="Export the model in FP16 precision (requires CUDA)."
    )
    parser.add_argument(
        "--check", 
        action="store_true", 
        default=True,
        help="Run ONNX checker on the exported model."
    )
    parser.add_argument(
        "--simplify", 
        action="store_true", 
        default=True,
        help="Run onnx-simplifier on the exported model."
    )
    args = parser.parse_args()
    main(args)
