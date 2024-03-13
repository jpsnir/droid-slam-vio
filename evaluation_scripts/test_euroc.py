import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse

from torch.multiprocessing import Process
from droid import Droid
from datetime import datetime
import torch.nn.functional as F
from pathlib import Path
import logging
logging.basicConfig(
     level=logging.DEBUG,
     format='%(asctime)s:%(levelname)s:%(funcName)s:%(lineno)s:%(message)s'
)
import csv
import pandas

NANOSEC_TO_SEC = 1e-9
SEC_TO_NANOSEC = 1e9

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def imu_stream(data_path, stride):
        "generator for IMU time stamps"
        
        timestamps = []
        with open(Path(data_path).joinpath("mav/imu0/data.csv"), "r") as f:
            data = csv.reader(f, delimiter=',')
            for row in data:
                timestamps.append(row[0])
        ts = timestamps[::stride]
        for t in ts:
            yield t

def image_stream(datapath, image_size=[320, 512], stereo=False, stride=1, start_image_name:str = None, max_images:int = -1):
    """ image generator """

    K_l = np.array([458.654, 0.0, 367.215, 0.0, 457.296, 248.375, 0.0, 0.0, 1.0]).reshape(3,3)
    d_l = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.0])
    R_l = np.array([
         0.999966347530033, -0.001422739138722922, 0.008079580483432283, 
         0.001365741834644127, 0.9999741760894847, 0.007055629199258132, 
        -0.008089410156878961, -0.007044357138835809, 0.9999424675829176
    ]).reshape(3,3)
    
    P_l = np.array([435.2046959714599, 0, 367.4517211914062, 0,  0, 435.2046959714599, 252.2008514404297, 0,  0, 0, 1, 0]).reshape(3,4)
    map_l = cv2.initUndistortRectifyMap(K_l, d_l, R_l, P_l[:3,:3], (752, 480), cv2.CV_32F)
    
    K_r = np.array([457.587, 0.0, 379.999, 0.0, 456.134, 255.238, 0.0, 0.0, 1]).reshape(3,3)
    d_r = np.array([-0.28368365, 0.07451284, -0.00010473, -3.555907e-05, 0.0]).reshape(5)
    R_r = np.array([
         0.9999633526194376, -0.003625811871560086, 0.007755443660172947, 
         0.003680398547259526, 0.9999684752771629, -0.007035845251224894, 
        -0.007729688520722713, 0.007064130529506649, 0.999945173484644
    ]).reshape(3,3)
    
    P_r = np.array([435.2046959714599, 0, 367.4517211914062, -47.90639384423901, 0, 435.2046959714599, 252.2008514404297, 0, 0, 0, 1, 0]).reshape(3,4)
    map_r = cv2.initUndistortRectifyMap(K_r, d_r, R_r, P_r[:3,:3], (752, 480), cv2.CV_32F)

    intrinsics_vec = [435.2046959714599, 435.2046959714599, 367.4517211914062, 252.2008514404297]
    ht0, wd0 = [480, 752]

    # read all png images in folder
    images_left = sorted(glob.glob(os.path.join(datapath, 'mav0/cam0/data/*.png')))
    
    if start_image_name is None:
        start_image = 0
        last_image = len(images_left)
    else:
        for id, f in enumerate(images_left):
            filename = f.split('/')[-1]
            if start_image_name == filename:
                start_image = id
                last_image = id + max_images
                break
    logging.info(f"First image chosen: {start_image}, max_images: {max_images}, stride - {stride}, stereo - {stereo}") 
    time.sleep(3)
    images_left = images_left[::stride]
    images_right = [x.replace('cam0', 'cam1') for x in images_left]
    if max_images > 0:
        images_left = images_left[start_image:last_image]
        images_right = images_right[start_image:last_image]
    #breakpoint()
    for t, (imgL, imgR) in enumerate(zip(images_left, images_right)):
        if stereo and not os.path.isfile(imgR):
            continue
        tstamp = float(imgL.split('/')[-1][:-4])        
        images = [cv2.remap(cv2.imread(imgL), map_l[0], map_l[1], interpolation=cv2.INTER_LINEAR)]
        if stereo:
            images += [cv2.remap(cv2.imread(imgR), map_r[0], map_r[1], interpolation=cv2.INTER_LINEAR)]
        
        images = torch.from_numpy(np.stack(images, 0))
        images = images.permute(0, 3, 1, 2).to("cuda:0", dtype=torch.float32)
        images = F.interpolate(images, image_size, mode="bilinear", align_corners=False)
        
        intrinsics = torch.as_tensor(intrinsics_vec).cuda()
        intrinsics[0] *= image_size[1] / wd0
        intrinsics[1] *= image_size[0] / ht0
        intrinsics[2] *= image_size[1] / wd0
        intrinsics[3] *= image_size[0] / ht0

        yield stride*t, images, intrinsics

def save_reconstruction(droid, args):
    from pathlib import Path
    import random
    import string

    date_time = args.date_time
    reconstruction_path = args.reconstruction_path
    dataset_name = args.dataset_name
    camera_name = args.camera_name

    t = droid.video.counter.value
    images_list = sorted(glob.glob(os.path.join(args.datapath, 'mav0/cam0/data/*.png')))
    tstamps = [float(x.split('/')[-1][:-4]) for x in images_list]
    image_ids = droid.video.tstamp[:t].cpu().numpy()
    tstamps = np.take(tstamps, image_ids.astype(int).tolist())
    images = droid.video.images[:t].cpu().numpy()
    disps = droid.video.disps_up[:t].cpu().numpy()
    poses = droid.video.poses[:t].cpu().numpy()
    intrinsics = droid.video.intrinsics[:t].cpu().numpy()
    print(f'poses size before interpolation: {poses.shape}')
    
    recon_save_path = Path(f"{reconstruction_path}").joinpath(
         dataset_name,"reconstructions",camera_name,args.ba_tag)
    recon_save_path.mkdir(parents=True, exist_ok=True)
    np.save(recon_save_path.joinpath("image_ids.npy").as_posix(), image_ids)
    np.save(recon_save_path.joinpath("tstamps.npy").as_posix(), tstamps)
    np.save(recon_save_path.joinpath("images.npy").as_posix(), images)
    np.save(recon_save_path.joinpath("disps.npy").as_posix(), disps)
    np.save(recon_save_path.joinpath("poses.npy").as_posix(), poses)
    np.save(recon_save_path.joinpath("intrinsics.npy").as_posix(), intrinsics)
    with open(recon_save_path.joinpath("notes.txt"), "w") as f:
         f.write("Timestamps: nanoseconds \n")
         f.write("Pose: T_cw - camera wrt world/ camera to world \n")
         f.write("World frame : first camera pose is identity")
         f.write("Pose format: tx, ty, tz, qx, qy, qz, qw")
         f.write("To evaluate with ground truth compute T_wc")
         f.write("disparity is in camera frame")
    

    logging.info(f"Saved reconstruction at {recon_save_path.as_posix()}")

def evaluate_trajectory(timestamps:dict, poses: dict, args) -> None:
    """ evaluation of trajectory and saving the results"""
    ### run evaluation ###
    import evo
    from evo.core.trajectory import PoseTrajectory3D
    from evo.tools import file_interface
    from evo.core import sync
    import evo.main_ape as main_ape
    from evo.core.metrics import PoseRelation
    traj_wc_nip = poses['nip']
    traj_est_wc_ip = poses['ip']
    tstamps_nip_s = timestamps['t_nip_s']
    tstamps_ip_ns = timestamps['t_ip_ns']


    ####################### FRAME TRANSFORMATIONS OF DROID SLAM ###########
    # NOTE: Droid SLAM computes the pose as T_cw (world to camera). 
    # Therefore inverse is taken to get T_wc. T_wc -> camera wrt world transformations.  
    # Droid SLAM trajectories are relative to camera's first frame 
    # Camera coordinate system : z-> forward, x -> right, y -> down.

    # Non - interpolated trajectories, time stamps in seconds and nanoseconds
    # NOTE: evo considers quaternions in q - wxyz format, droid has it in xyzw format
    # therefore np.roll is used.
    evo_traj_est_wc_nip_s = PoseTrajectory3D(
         positions_xyz = traj_wc_nip[:, :3],
         orientations_quat_wxyz = np.roll(traj_wc_nip[:, 3:], 1, axis=1),
         timestamps = tstamps_nip_s,
    )

    evo_traj_est_wc_nip_ns = PoseTrajectory3D(
        positions_xyz = traj_wc_nip[:,:3],
        orientations_quat_wxyz=np.roll(traj_wc_nip[:,3:], 1, axis=1),
        timestamps=tstamps_nip_s*SEC_TO_NANOSEC)    
    
    # Interpolated trajectories, timestamps in seconds and nanoseconds. 
    evo_traj_est_wc_ip_ns = PoseTrajectory3D(
        positions_xyz = traj_est_wc_ip[:,:3],
        orientations_quat_wxyz=np.roll(traj_est_wc_ip[:,3:], 1, axis=1),
        timestamps=np.array(tstamps_ip_ns))
    
    evo_traj_est_wc_ip_s = PoseTrajectory3D(
        positions_xyz = traj_est_wc_ip[:,:3],
        orientations_quat_wxyz=np.roll(traj_est_wc_ip[:,3:], 1, axis=1),
        timestamps=np.array(tstamps_ip_ns)*NANOSEC_TO_SEC)
    
    

    # Reference trajectory from ground truth 
    # NOTE : TUM format of input file is tx, ty, tz, qx, qy, qz, qw.
    # evo package uses q = (qw, qx, qy, qz) - converts it while reading the TUM format files.  
    # measured from world coordinate system of the EUROC - (uses ROS convention - xyz -> Forward, Left, Up)
    # This reference trajectory is T_wb -> body frame wrt to world frame. 
    # world frame will be estimated by aligning the trajectory. 

    traj_ref = file_interface.read_tum_trajectory_file(args.gt)
    
    result = {}
    if args.stereo:
        logging.info("Stereo trajectory alignment and ape error computations - only with translation part")
        traj_ref = file_interface.read_tum_trajectory_file(args.gt)
        traj_ref, traj_est_stereo_ip = sync.associate_trajectories(traj_ref, evo_traj_est_wc_ip_ns)
        result['stereo_ip'] = main_ape.ape(traj_ref, traj_est_stereo_ip, est_name='traj', 
            pose_relation=PoseRelation.translation_part, align=True, correct_scale=False)
        
        logging.debug("APE error for non-interpolated stereo trajectory")
        traj_ref = file_interface.read_tum_trajectory_file(args.gt)
        traj_ref, traj_est_stereo_nip = sync.associate_trajectories(traj_ref, evo_traj_est_wc_nip_ns)
        result['stereo_nip'] = main_ape.ape(traj_ref, traj_est_stereo_nip, est_name='traj', 
            pose_relation=PoseRelation.translation_part, align=True, correct_scale=False)
        
        logging.info(f"EVO alignment and trajectory evaluation \n STEREO - INTERPOLATED \n {result['stereo_ip']}")
        logging.info(f"EVO alignment and trajectory evaluation \n STEREO - NOT INTERPOLATED \n {result['stereo_nip']}")
    else:
        logging.info("Mono Trajectory alignment and ape error computation - only with translation part")
        traj_ref = file_interface.read_tum_trajectory_file(args.gt)
        traj_ref, traj_est_mono_ip = sync.associate_trajectories(traj_ref, evo_traj_est_wc_ip_ns)
        result['mono_ip'] = main_ape.ape(traj_ref, traj_est_mono_ip, est_name='traj', 
            pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)
        
        logging.debug("Computing APE error for noninterpolated and not prescaled trajectory")
        traj_ref = file_interface.read_tum_trajectory_file(args.gt)
        traj_ref, traj_est_mono_nip = sync.associate_trajectories(traj_ref, evo_traj_est_wc_nip_ns)
        result['mono_nip'] = main_ape.ape(traj_ref, traj_est_mono_nip, est_name='traj', 
            pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)
        
        logging.info(f"EVO alignment and trajectory evaluation \n  MONO - INTERPOLATED : \n {result['mono_ip']}")
        logging.info(f"EVO alignment and trajectory evaluation \n  MONO -  NOT INTERPOLATED : \n {result['mono_nip']}")
    
    # Save trajectories:
    p = Path(args.reconstruction_path).joinpath("evo", args.dataset_name, args.camera_name,args.ba_tag)
    p.mkdir(parents=True, exist_ok=True)
    file_interface.write_tum_trajectory_file(p.joinpath(f"traj_ref__ns_{args.ba_tag}.txt").as_posix(), traj_ref)
    file_interface.write_tum_trajectory_file(p.joinpath(f"T_wc_nip_s_{args.ba_tag}.txt").as_posix(), evo_traj_est_wc_nip_s)
    file_interface.write_tum_trajectory_file(p.joinpath(f"T_wc_ip_s_{args.ba_tag}.txt").as_posix(), evo_traj_est_wc_ip_s)
    
    if args.stereo:
        file_interface.write_tum_trajectory_file(p.joinpath(f"T_wc_ip_ns_{args.ba_tag}.txt").as_posix(), traj_est_stereo_ip)
        file_interface.write_tum_trajectory_file(p.joinpath(f"T_wc_nip_ns_{args.ba_tag}.txt").as_posix(), traj_est_stereo_nip)
        with open(p.joinpath(f"ape_stereo_ip_{args.ba_tag}.txt").as_posix(), 'w') as f:
            f.write(str(result['stereo_ip']))
        with open(p.joinpath(f"ape_stereo_nip_{args.ba_tag}.txt").as_posix(), 'w') as f:
            f.write(str(result['stereo_nip']))
    else:     
        file_interface.write_tum_trajectory_file(p.joinpath(f"T_wc_ip_ns_{args.ba_tag}.txt").as_posix(), traj_est_mono_ip)
        file_interface.write_tum_trajectory_file(p.joinpath(f"T_wc_ip_ns_{args.ba_tag}.txt").as_posix(), traj_est_mono_nip)
        with open(p.joinpath(f"ape_mono_ip_{args.ba_tag}.txt").as_posix(), 'w') as f:
            f.write(str(result['mono_ip']))
        
        with open(p.joinpath(f"ape_mono_nip_{args.ba_tag}.txt").as_posix(), 'w') as f:
            f.write(str(result['mono_nip']))
    
    # save umeyama alignment results.
    logging.info("saving Umeyama alignment results")
    traj_ref, traj_est_ip = sync.associate_trajectories(traj_ref, evo_traj_est_wc_ip_ns)
    r_a_um, t_a_um, s_um = traj_est_ip.align(traj_ref, correct_scale = True)
    with open(p.joinpath(f"umeyama_alignment_ip_{args.ba_tag}.txt").as_posix(), 'w') as f:
        f.write(f"Rotation : \n {r_a_um}\n")
        f.write(f"Translation : \n {t_a_um}\n")
        f.write(f"Scale : \n {s_um}\n")
    
    traj_ref, traj_est_nip = sync.associate_trajectories(traj_ref, evo_traj_est_wc_nip_ns)
    r_a_um, t_a_um, s_um = traj_est_nip.align(traj_ref, correct_scale = True)
    with open(p.joinpath(f"umeyama_alignment_no_ip_{args.ba_tag}.txt").as_posix(), 'w') as f:
        f.write(f"Rotation : \n {r_a_um}\n")
        f.write(f"Translation : \n {t_a_um}\n")
        f.write(f"Scale : \n {s_um}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", help="path to euroc sequence")
    parser.add_argument("--gt", help="path to gt file")
    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[320,512])
    parser.add_argument("--disable_vis", action="store_true")
    parser.add_argument("--stereo", action="store_true")
    parser.add_argument("--stride", default=2, type=int, help="skip number of images")
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=2.4)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--keyframe_thresh", type=float, default=3.5)
    parser.add_argument("--frontend_thresh", type=float, default=17.5)
    parser.add_argument("--frontend_window", type=int, default=20, 
                        help="The number of frames which will be considered for proximity")
    parser.add_argument("--frontend_radius", type=int, default=2)
    parser.add_argument("--frontend_nms", type=int, default=1)
    parser.add_argument("--max_age",type=int, default=25, 
                        help="The number of frames for marginalization in factor graph")
    parser.add_argument("--max_factors", type=int, default=48, 
                        help="max number of factors/edges in the factor graph")
    parser.add_argument("--max_images", type=int, default=-1,
                        help="max number of images to process")

    parser.add_argument("--backend_thresh", type=float, default=24.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=2)
    parser.add_argument("--start_image_name", default = None, type=str, help="image start name for getting the trajectory")
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--reconstruction_path", help="path to saved reconstruction", default=".")
    parser.add_argument("--global_ba", action="store_true", help="flag to perform global BA at the end.")
    parser.add_argument(
        "-fg_fmt",
        "--factor_graph_save_format",
        type=str,
        default="pkl",
        help="format to save factor graph data",
    )

    args = parser.parse_args()
    args.date_time = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    args.dataset_name = os.path.basename(args.datapath)
    if args.stereo:
            args.camera_name = 'stereo'
    else:
            args.camera_name = 'mono'
    if args.global_ba:
         args.ba_tag = "gba"
    else:
         args.ba_tag = "lba"
    
    # processing starts
    logging.info(f"Running evaluation on {args.datapath}")
    torch.multiprocessing.set_start_method('spawn')
    droid = Droid(args)
    time.sleep(5)
    

    for (t, image, intrinsics) in tqdm(
         image_stream(args.datapath, stereo=args.stereo, stride=args.stride,  
                      start_image_name=args.start_image_name, max_images=args.max_images)):
        droid.track(t, image, intrinsics=intrinsics)

    save_reconstruction(droid, args)
    
    images_list = sorted(glob.glob(os.path.join(args.datapath, 'mav0/cam0/data/*.png')))
    # these timestamps are in nanoseconds
    tstamps_ns = [float(x.split('/')[-1][:-4]) for x in images_list]

    # data for trajectory without interpolation
    n = droid.video.counter.value
    image_ids = droid.video.tstamp[:n].cpu().numpy()
    tstamps_no_ip_seconds = np.take(
         tstamps_ns, image_ids.astype(int).tolist())*NANOSEC_TO_SEC
    poses_se3 = lietorch.SE3(droid.video.poses[:n])
    poses_no_interpolation_inverted = poses_se3.inv().data.cpu().numpy()
    poses_no_interpolation = poses_se3.data.cpu().numpy()

    # terminate fills the trajectory by interpolating at intermediate timestamps.
    traj_est = droid.terminate(image_stream(args.datapath, stride=1))
    logging.debug(f'poses shape after interpolation : {traj_est.shape}')
    poses = {'nip': poses_no_interpolation_inverted,
             'ip':traj_est
    }
    timestamps = {"t_nip_s": tstamps_no_ip_seconds,
                  "t_ip_ns": tstamps_ns
    }
    evaluate_trajectory(timestamps, poses, args)

    
    


