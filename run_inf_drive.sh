#!/bin/bash
set -e

# export S3_ENDPOINT=https://obs.cn-southwest-2.huaweicloud.com
# export S3_USE_HHTPS=0
# export ACCESS_KEY_ID=HPUAUMBABND5R21BA8CR
# export SECRET_ACCESS_KEY=GPs3Ag6ahEpm]rEZZmb9bOUlWaCHBVVLYR1rONSV

# Path to your Python script
PY_SCRIPT="qwen_cot_generate.py"

export NUPLAN_MAPS_ROOT=/home/ma-user/work/navsim/maps
export ASCEND_LAUNCH_BLOCKING=1

# Common args for all ranks
NAV_ROOT="obs://yw-2030-extern/Partner_Zhu/navsim/navsim-data/"
OUTPUT_JSON="converted_output"
MODEL_PATH="/home/ma-user/work/Qwen2.5-VL-7B-Instruct"

# Additional parameters matching qwen_cot_generate.py
IMAGE_ROOT="./images"

DATA_TYPE="mini"
OBS_ROOT="obs://yw-2030-extern/Partner_Zhu/navsim/navsim-data/sensor_blobs/$DATA_TYPE/"
NUM_HIST_TRAJ=4
NUM_FUT_TRAJ=6

MAX_NEW_TOKENS=1280
MIN_PIXELS=64
MAX_PIXELS=1280
VIS_PATH="./visual"
IF_VIS=Ture

# Loop over ranks (GPUs) and launch processes
cmd="python $PY_SCRIPT \
    --model_path $MODEL_PATH \
    --nav_root $NAV_ROOT \
    --out_dir $OUTPUT_JSON \
    --image_root $IMAGE_ROOT \
    --obs_root $OBS_ROOT \
    --data_type $DATA_TYPE \
    --num_hist_traj $NUM_HIST_TRAJ \
    --num_fut_traj $NUM_FUT_TRAJ \
    --max_new_tokens $MAX_NEW_TOKENS \
    --min_pixels $MIN_PIXELS \
    --max_pixels $MAX_PIXELS \
    --vis_path $VIS_PATH \
    --rank_idx 0 \
    --world_size 1 "
echo "$cmd"
eval "$cmd" &

wait
echo "All ranks finished."
