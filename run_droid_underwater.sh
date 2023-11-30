#!/usr/bin/bash

show_help() {
      echo "Usage : ./run_droid_underwater.sh -d <datadirectory> -v(optional)"
      echo "d : data directory, v : visualization on. "
}


viz=false

if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
  echo "Conda environment is active: $CONDA_DEFAULT_ENV"
else
  echo "No Conda environment is active"
  exit 1
fi

# parse options
while getopts "d:v" opt; do
  case "$opt" in
    d)
      if [ -n "$OPTARG" ];then 
         data_dir="$OPTARG"
	 output_folder=$(basename "$data_dir")
         results_folder=$(dirname $data_dir)

         echo "data directory: $data_dir"
         echo "results folder name: $results_folder"
	 echo "output folder name: $output_folder"
      else
	 echo " Missing directory name"
	 exit 1
      fi 
      ;;
    v)
      viz=true
      echo "Visualization : $viz"
      ;;
    h)
      show_help
      ;;
    \?)
      # Handle unrecognized options here
      echo "Invalid option: -$OPTARG"
      show_help
      exit 1
      ;;
  esac
done

# Required options are not provided.
if [ -z "$data_dir" ]; then
  echo "Data directory is required"
  show_help
  exit 1
fi

# define specific paths
vins_workspace="/home/$USER/workspaces/NEUFR/vins"
calibration_file="$vins_workspace/config_files/vslam_configs/droid_slam/intrinsics_uw.txt"
model_weights_file="$vins_workspace/config_files/vslam_configs/droid_slam/droid.pth"

if [ -e "$vins_workspace" ]; then
   echo "vins workspace folder exists"
else
   echo "vins workspace does not exist"
   exit 1
fi

# run droid slam
if [ "$viz" = false ]; then
	python demo.py \
		--imagedir="$data_dir" \
		--calib="$calibration_file" \
		--stride=1 \
		--weights="$model_weights_file" \
		--buffer=2048 \
		--reconstruction_path="$results_folder"\
		--disable_vis\
	        --factor_graph_save_format="pkl"
else 
	python demo.py \
		--imagedir="$data_dir" \
		--calib="$calibration_file" \
		--stride=1 \
		--weights="$model_weights_file" \
		--buffer=2048 

fi
exit 0
