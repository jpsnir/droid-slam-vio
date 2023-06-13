#!/usr/bin/bash

if [ $# -ne 1 ]; then
   echo "Usage : ./run_droid_underwater.sh <datadirectory>"
   exit 0
fi

data_dir=$1

python demo.py \
        --imagedir="$data_dir" \
        --calib=/media/goku/data/droid_slam/dataset_2_uw.txt \
        --stride=1 \
        --weights=/media/goku/data/droid_slam/droid.pth \
        --buffer=1024
 
