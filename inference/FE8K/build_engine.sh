#!/bin/bash
set -e  # Exit if any command fails

echo "==========================="
echo " Building TensorRT Engines"
echo "==========================="

# Go to models folder
cd ./models

# Export ONNX models
echo "-> Running get_model.py..."
python3 get_model.py

# Build D-FINE-L engine
echo "-> Building TensorRT engine for D-FINE-L..."
/usr/src/tensorrt/bin/trtexec \
  --onnx="D-FINE-L.onnx" \
  --saveEngine="D-FINE-L.engine" \
  --fp16 \
  --builderOptimizationLevel=5 \
  --useCudaGraph

# Go back to main directory
cd ..

echo "✅ Engine build complete."

