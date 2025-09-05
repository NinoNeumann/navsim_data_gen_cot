#!/bin/bash
set -e

# distribution setting
WORLD_SIZE=1
RANK_IDX=0

# Path to your Python script
PY_SCRIPT="qwen_cot_generate.py"

export NUPLAN_MAPS_ROOT=./navsim/maps
# export ASCEND_LAUNCH_BLOCKING=1

# Common args for all ranks
NAV_ROOT="obs://yw-2030-extern/Partner_Zhu/navsim/navsim-data/"
OUTPUT_DIR="./"
MODEL_PATH="../Qwen2.5-VL-7B-Instruct"

DATA_TYPE="mini"
OBS_ROOT="obs://yw-2030-extern/Partner_Zhu/navsim/navsim-data/sensor_blobs/$DATA_TYPE/"
NUM_HIST_TRAJ=4
NUM_FUT_TRAJ=24
FRAME_INTERVAL=10

MAX_NEW_TOKENS=1280

# TODO 加入参数
MIN_PIXELS=175616
MAX_PIXELS=401408

VIS_PATH="./visual"
IF_VIS=Ture


# Loop over ranks (GPUs) and launch processes
cmd="python $PY_SCRIPT \
    --model_path $MODEL_PATH \
    --nav_root $NAV_ROOT \
    --out_dir $OUTPUT_DIR \
    --obs_root $OBS_ROOT \
    --data_type $DATA_TYPE \
    --num_hist_traj $NUM_HIST_TRAJ \
    --num_hist_frames 1 \
    --num_fut_traj $NUM_FUT_TRAJ \
    --max_new_tokens $MAX_NEW_TOKENS \
    --vis_path $VIS_PATH \
    --frame_interval $FRAME_INTERVAL \
    --rank_idx $RANK_IDX \
    --world_size $WORLD_SIZE  \
    --max_scenes 1 
"
echo "$cmd"
eval "$cmd" &

wait
echo "All ranks finished."
