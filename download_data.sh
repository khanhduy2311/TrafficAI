#!/usr/bin/env bash

# ========================
# Setup Kaggle API
# ========================
echo -e "\n\033[1;34m[INFO]\033[0m Setting up Kaggle credentials..."
mkdir -p ~/.kaggle
cp /workspace/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
echo -e "\033[1;32m[DONE]\033[0m Kaggle credentials setup complete."

# ========================
# Download YOLO Dataset
# ========================
echo -e "\n\033[1;34m[INFO]\033[0m Starting YOLO dataset download..."
mkdir -p /workspace/datasets/yolo/
cd /workspace/datasets/yolo/
kaggle datasets download whonoac/carlafisheyecubidangcapso1vietname --unzip
echo -e "\033[1;32m[DONE]\033[0m YOLO dataset downloaded to /workspace/datasets/yolo/"

# ========================
# Download DFINE Batches
# ========================
echo -e "\n\033[1;34m[INFO]\033[0m Starting DFINE dataset download..."
mkdir -p /workspace/datasets/dfine/
cd /workspace/datasets/dfine/

echo -e "\033[1;36m[Downloading]\033[0m Batch 1..."
kaggle datasets download -d newnguyen/dfine-batch-1 --unzip
echo -e "\033[1;32m[DONE]\033[0m Dfine-Batch-1 downloaded."

echo -e "\033[1;36m[Downloading]\033[0m Batch 2..."
kaggle datasets download -d newnguyen/dfine-batch-2 --unzip 
echo -e "\033[1;32m[DONE]\033[0m Dfine-Batch-2 downloaded."

echo -e "\033[1;36m[Downloading]\033[0m Validation set..."
kaggle datasets download -d newnguyen/dfine-val --unzip 
echo -e "\033[1;32m[DONE]\033[0m Dfine-Val downloaded."

# ========================
# Download Stage 2 Dataset
# ========================
echo -e "\n\033[1;34m[INFO]\033[0m Starting download of Stage 2 dataset..."
cd /workspace/datasets/
kaggle datasets download -d newnguyen/yolo-stage-2 --unzip
echo -e "\033[1;32m[DONE]\033[0m Stage 2 dataset downloaded."

# ========================
# Copy Stage 2 to DFINE
# ========================
echo -e "\n\033[1;34m[INFO]\033[0m Copying Stage 2 images to dfine_stage_2 folder..."
mkdir -p /workspace/datasets/dfine_stage_2/train/
cp -r /workspace/datasets/yolo_stage_2/train/images /workspace/datasets/dfine_stage_2/train/
echo -e "\033[1;32m[DONE]\033[0m Images copied to dfine_stage_2/train/"

# # ========================
# # Merge DFINE Batches
# # ========================
echo -e "\n\033[1;34m[INFO]\033[0m Merging Dfine Batch 1 and 2..."
python /workspace/src/merge_dfine.py --batch1 /workspace/datasets/dfine/batch_1 --batch2 /workspace/datasets/dfine/batch_2 --output /workspace/datasets/dfine/train/ --workers 25
echo -e "\033[1;32m[DONE]\033[0m Merge complete."

# ========================
# Convert YOLO to COCO Format
# ========================
pip install opencv-python
echo -e "\n\033[1;34m[INFO]\033[0m Converting YOLO to COCO format..."

echo -e "\033[1;36m[Processing]\033[0m Training set..."
python /workspace/src/yolo2coco.py --images_dir /workspace/datasets/dfine/train/images --labels_dir /workspace/datasets/dfine/train/labels --output_json /workspace/datasets/dfine/train/train.json 

echo -e "\033[1;36m[Processing]\033[0m Validation set..."
python /workspace/src/yolo2coco.py --images_dir /workspace/datasets/dfine/val/images --labels_dir /workspace/datasets/dfine/val/labels --output_json /workspace/datasets/dfine/val/val.json 

echo -e "\033[1;32m[DONE]\033[0m COCO conversion complete."

# ========================
# All done!
# ========================

mv /workspace/dfine_stage_2.json /workspace/datasets/dfine_stage_2/train/

echo -e "\n\033[1;35m[ALL TASKS COMPLETED SUCCESSFULLY]\033[0m 🎉"
