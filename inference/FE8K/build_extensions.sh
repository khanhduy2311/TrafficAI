#!/bin/bash
set -e  # Exit on error

echo "==========================="
echo " Building C++ Extensions..."
echo "==========================="

# Build fusion/wbf_cpp
echo "-> Building fusion/wbf_cpp..."
cd fusion
python3 setup_wbf.py build_ext --inplace

# Build preproc module
echo "-> Building preproc module..."
cd ../preproc
python3 setup.py build_ext --inplace

echo "✅ All extensions built successfully."

