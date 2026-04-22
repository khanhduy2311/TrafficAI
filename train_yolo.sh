echo "Starting Training Yolo..."
. /root/miniconda3/etc/profile.d/conda.sh
conda activate yolo
cd Yolo

# # Stage 1
python train_st_1.py
echo "Finished Stage 1!"

# Stage 2
python train_st_2.py
echo "Finished Stage 2!"
echo "Fishished Training Yolo..."