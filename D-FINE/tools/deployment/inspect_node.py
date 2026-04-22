import onnx
import sys

# Path to your ONNX model
onnx_path = "best_stg2_dynamic_batch_fp16.onnx"

# Load the model
model = onnx.load(onnx_path)
nodes = model.graph.node

idx = 182
if idx >= len(nodes):
    print(f"Model only has {len(nodes)} nodes, cannot inspect #{idx}")
    sys.exit(1)

n = nodes[idx]
print(f"Node #{idx}:")
print("  name:    ", n.name or "(no name)")
print("  op_type: ", n.op_type)
print("  domain:  ", n.domain or "ai.onnx")
print("  inputs:  ", list(n.input))
print("  outputs: ", list(n.output))
if n.attribute:
    print("  attributes:")
    for attr in n.attribute:
        v = onnx.helper.get_attribute_value(attr)
        print(f"    - {attr.name}: {v!r}")
else:
    print("  (no attributes)")
