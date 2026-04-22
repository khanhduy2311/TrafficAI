#!/bin/bash

. /root/miniconda3/etc/profile.d/conda.sh
conda activate dfine

echo "Starting Training DFine..."
echo "Stage 1..."

# Finetune Stage 1
cd D-FINE
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 train.py -c configs/dfine/custom/objects365/finetune_stage_1_dfine_l.yml --use-amp --seed=0 -t ../models/dfine_l_obj365_e25.pth

# Finetune Stage 2
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 train.py -c configs/dfine/custom/objects365/finetune_stage_2_dfine_l.yml --use-amp --seed=0 -t output/finetune_stage_1_dfine_l/best_stg2.pth