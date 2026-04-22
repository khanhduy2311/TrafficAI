"""
A script to fuse two identical ONNX models for ensembled inference.

This script loads two ONNX models, renames their internal nodes to prevent
conflicts, and combines them into a single graph. The new graph takes one
set of inputs, feeds it to both models in parallel, and concatenates their
outputs.

Usage:
    python fuse_models.py --model1 model_a.onnx --model2 model_b.onnx --output fused_model.onnx
"""
import onnx
import argparse
from onnx import helper, checker

def fuse_models(model1_path, model2_path, output_path):
    """
    Fuses two ONNX models with identical architecture and inputs/outputs.
    """
    print(f"Loading model 1 from: {model1_path}")
    model1 = onnx.load(model1_path)
    print(f"Loading model 2 from: {model2_path}")
    model2 = onnx.load(model2_path)
    
    # --- The IR version of the runtime is older, let's find a compatible one ---
    # We will downgrade the final model to match the runtime's capabilities.
    # From your error, model1 is likely IR v11, let's target IR v10.
    target_ir_version = 10 
    print(f"Original model IR version: {model1.ir_version}. Targetting IR version {target_ir_version} for compatibility.")


    prefix1 = "model1_"
    prefix2 = "model2_"

    # (The prefixing logic remains the same as the last correct version)
    for node in model1.graph.node:
        node.name = prefix1 + node.name
        for i in range(len(node.input)):
            if node.input[i]: node.input[i] = prefix1 + node.input[i]
        for i in range(len(node.output)):
            if node.output[i]: node.output[i] = prefix1 + node.output[i]
    for initializer in model1.graph.initializer:
        if initializer.name: initializer.name = prefix1 + initializer.name
    for vi in model1.graph.value_info:
        if vi.name: vi.name = prefix1 + vi.name

    for node in model2.graph.node:
        node.name = prefix2 + node.name
        for i in range(len(node.input)):
            if node.input[i]: node.input[i] = prefix2 + node.input[i]
        for i in range(len(node.output)):
            if node.output[i]: node.output[i] = prefix2 + node.output[i]
    for initializer in model2.graph.initializer:
        if initializer.name: initializer.name = prefix2 + initializer.name
    for vi in model2.graph.value_info:
        if vi.name: vi.name = prefix2 + vi.name

    fused_graph_nodes = list(model1.graph.node) + list(model2.graph.node)
    fused_graph_initializers = list(model1.graph.initializer) + list(model2.graph.initializer)
    fused_graph_value_info = list(model1.graph.value_info) + list(model2.graph.value_info)

    original_input_names = [inp.name for inp in model1.graph.input]
    for name in original_input_names:
        for node in fused_graph_nodes:
            for i, node_input in enumerate(node.input):
                if node_input == prefix1 + name: node.input[i] = name
                if node_input == prefix2 + name: node.input[i] = name

    output_names = [out.name for out in model1.graph.output]
    fused_outputs = []
    
    for name in output_names:
        fused_output_name = "fused_" + name
        concat_node = helper.make_node('Concat', inputs=[prefix1 + name, prefix2 + name], outputs=[fused_output_name], name="concat_" + name, axis=1)
        fused_graph_nodes.append(concat_node)
        
        original_output_info = next((o for o in model1.graph.output if o.name == name), None)
        if original_output_info:
            fused_output_info = onnx.ValueInfoProto()
            fused_output_info.CopyFrom(original_output_info)
            fused_output_info.name = fused_output_name
            if len(fused_output_info.type.tensor_type.shape.dim) > 1:
                fused_output_info.type.tensor_type.shape.dim[1].ClearField('dim_value')
                fused_output_info.type.tensor_type.shape.dim[1].dim_param = "num_fused_detections"
            fused_outputs.append(fused_output_info)

    fused_graph = helper.make_graph(
        nodes=fused_graph_nodes,
        name='fused-dfine-model-graph',
        inputs=model1.graph.input,
        outputs=fused_outputs,
        initializer=fused_graph_initializers,
        value_info=fused_graph_value_info
    )

    fused_model = helper.make_model(fused_graph, producer_name='D-FINE-fuse-script')
    fused_model.opset_import[0].version = model1.opset_import[0].version

    # =========================================================================
    # === THE FIX: Manually set the IR version to an older, compatible one. ===
    # =========================================================================
    fused_model.ir_version = target_ir_version

    print("Checking the fused model...")
    checker.check_model(fused_model)
    print("Fused model is valid.")
    
    onnx.save(fused_model, output_path)
    print(f"Successfully saved fused model to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuse two identical ONNX models.")
    parser.add_argument("--model1", type=str, required=True, help="Path to the first ONNX model file.")
    parser.add_argument("--model2", type=str, required=True, help="Path to the second ONNX model file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the fused ONNX model.")
    args = parser.parse_args()
    fuse_models(args.model1, args.model2, args.output)