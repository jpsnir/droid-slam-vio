#!/usr/bin/bash

if [ $# -ne 2 ]; then
   echo "Usage : ./run_droid_underwater.sh <datadirectory> <output_filename>"
   exit 0
fi

data_dir=$1
output_filename=$2
vins_workspace="/home/$USER/workspaces/NEUFR/vins"
results_folder=$(basename "$(dirname '$data_dir')")
calibration_file="$vins_workspace/config_files/vslam_configs/droid_slam/intrinsics_uw.txt"
model_weights_file="$vins_workspace/config_files/vslam_configs/droid_slam/droid.pth"
if [ -e "$results_folder" ]; then
   echo "Results folder exists"
else
   echo " results folder does not exist"
   exit 1
fi

if [ -e "$vins_workspace" ]; then
   echo "vins workspace folder exists"
else
   echo "vins workspace does not exist"
   exit 1
fi

python demo.py \
        --imagedir="$data_dir" \
        --calib="$calibration_file" \
        --stride=1 \
        --weights="$model_weights_file" \
        --buffer=1024 \
        --reconstruction_path="$results_folder/$output_filename"
 
exit 0
