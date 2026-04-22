import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def convert_onnx_to_trt(onnx_path, engine_path, batch_size=1, precision="fp16"):
    # Initialize TensorRT stuff
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # Parse ONNX model
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, 'rb') as model:
        parser.parse(model.read())
        
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))

    # Create optimization profile
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "images",
        min=(1, 3, 640, 640),
        opt=(batch_size, 3, 640, 640),
        max=(batch_size, 3, 640, 640)
    )
    profile.set_shape(
        "orig_target_sizes",
        min=(1, 2),
        opt=(batch_size, 2),
        max=(batch_size, 2)
    )

    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30) # 1 MiB
    config.add_optimization_profile(profile)
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
        config.clear_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        for layer_idx in range(network.num_layers):
            layer = network[layer_idx]
            if layer.type == trt.LayerType.NORMALIZATION:
                layer.precision = trt.float32
                layer.set_output_type(0, trt.float32)


    # Build and save engine
    serialized_engine = builder.build_serialized_network(network, config)
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)


    print(f"Successfully converted model to TensorRT engine: {engine_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert an ONNX model to a TensorRT engine.")
    parser.add_argument(
        "--onnx-path",
        type=str,
        required=True,
        help="Path to the input ONNX model file."
    )
    parser.add_argument(
        "--engine-path",
        type=str,
        required=True,
        help="Path to save the output TensorRT engine file."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for the optimization profile (default: 1)."
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp16", "fp32"],
        help="Precision for the engine: 'fp16' or 'fp32' (default: fp16)."
    )

    args = parser.parse_args()
    
    convert_onnx_to_trt(
        onnx_path=args.onnx_path,
        engine_path=args.engine_path,
        batch_size=args.batch_size,
        precision=args.precision
    )
