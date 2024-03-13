#!/bin/bash


EUROC_PATH="/data/jagat/euroc/"
recon_path="/data/jagat/processed/"
evalset=(
   MH_01_easy
   MH_02_easy
   MH_03_medium
   MH_04_difficult
    MH_05_difficult
    V1_01_easy
    V1_02_medium
    V1_03_difficult
    V2_01_easy
    V2_02_medium
    V2_03_difficult
)
declare -A start_images
start_images["MH_01_easy"]="1403636625713555456.png"
start_images["MH_02_easy"]="1403636897801666560.png"
start_images["MH_03_medium"]="1403637149588318976.png"
start_images["MH_04_difficult"]="1403638147695097088.png"
start_images["MH_05_difficult"]="1403638538727829504.png"
start_images["V1_01_easy"]="1403715278012143104.png"
start_images["V1_02_medium"]="1403715528012143104.png"
start_images["V1_03_difficult"]="1403715893234057984.png"
start_images["V2_01_easy"]="1413393217155760384.png"
start_images["V2_02_medium"]="1413393889555760384.png"
start_images["V2_03_difficult"]="1413394887105760512.png"


for seq in ${evalset[@]}; do
    python evaluation_scripts/test_euroc.py --datapath=$EUROC_PATH/$seq --gt=data/euroc_groundtruth/$seq.txt\
     --weights=droid.pth --max_age=120 --stride=2 --max_factors=20000 --start_image_name=${start_images[${seq}]} --max_images=400 --reconstruction_path=$recon_path $@ 
done

# for seq in ${evalset[@]}; do
#     python evaluation_scripts/test_euroc.py --datapath=$EUROC_PATH/$seq --gt=data/euroc_groundtruth/$seq.txt --weights=droid.pth --disable_vis --reconstruction_path=$recon_path $@
# done

# for seq in ${evalset[@]}; do
#     python evaluation_scripts/test_euroc.py --datapath=$EUROC_PATH/$seq --gt=data/euroc_groundtruth/$seq.txt --weights=droid.pth --disable_vis  --global_ba  --reconstruction_path=$recon_path $@
# done


# for seq in ${evalset[@]}; do
#     python evaluation_scripts/test_euroc.py --datapath=$EUROC_PATH/$seq --gt=data/euroc_groundtruth/$seq.txt --stereo --weights=droid.pth --disable_vis --reconstruction_path=$recon_path $@
# done

# for seq in ${evalset[@]}; do
#     python evaluation_scripts/test_euroc.py --datapath=$EUROC_PATH/$seq --gt=data/euroc_groundtruth/$seq.txt --stereo --weights=droid.pth --disable_vis --global_ba --reconstruction_path=$recon_path $@
# done
