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

evalset=(
   V1_01_easy
)

for seq in ${evalset[@]}; do
    python evaluation_scripts/test_euroc.py --datapath=$EUROC_PATH/$seq --gt=data/euroc_groundtruth/$seq.txt\
     --weights=droid.pth --max_age=120 --stride=2 --max_factors=20000  --max_images=400 --reconstruction_path=$recon_path $@ 
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
