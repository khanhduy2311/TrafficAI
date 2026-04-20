#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate cv

# Đoạn cấu hình GPU để chạy Model: thay đổi số 0 thành số hiệu GPU bạn muốn (ví dụ: 1 hoặc 0,1)
export CUDA_VISIBLE_DEVICES=0 

pkill -f 'python run.py' || true
nohup python run.py > webapp_backend.log 2>&1 &
echo "✓ Backend started on port 5000 in background. Log at webapp_backend.log"
