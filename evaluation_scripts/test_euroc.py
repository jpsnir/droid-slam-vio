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
NANOSEC_TO_SEC = 1e-9
SEC_TO_NANOSEC = 1e9

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def image_stream(datapath, image_size=[320, 512], stereo=False, stride=1):
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
    images_left = sorted(glob.glob(os.path.join(datapath, 'mav0/cam0/data/*.png')))[::stride]
    images_right = [x.replace('cam0', 'cam1') for x in images_left]

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
         dataset_name,"reconstructions",camera_name)
    recon_save_path.mkdir(parents=True, exist_ok=True)
    np.save(recon_save_path.joinpath("image_ids.npy").as_posix(), image_ids)
    np.save(recon_save_path.joinpath("tstamps.npy").as_posix(), tstamps)
    np.save(recon_save_path.joinpath("images.npy").as_posix(), images)
    np.save(recon_save_path.joinpath("disps.npy").as_posix(), disps)
    np.save(recon_save_path.joinpath("poses.npy").as_posix(), poses)
    np.save(recon_save_path.joinpath("intrinsics.npy").as_posix(), intrinsics)
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
    poses_no_interpolation = poses['no_ip']
    poses_no_interpolation_inverted = poses['no_ip_inv']
    traj_est = poses['ip']
    tstamps_no_ip_seconds = timestamps['t_no_ip_s']
    tstamps_ns = timestamps['t_ip_ns']

    traj_est_no_interpolation = PoseTrajectory3D(
         positions_xyz = poses_no_interpolation[:, :3],
         orientations_quat_wxyz = poses_no_interpolation[:, 3:],
         timestamps = tstamps_no_ip_seconds,
    )
    traj_est_no_interpolation_inverted = PoseTrajectory3D(
         positions_xyz = poses_no_interpolation_inverted[:, :3],
         orientations_quat_wxyz = poses_no_interpolation_inverted[:, 3:],
         timestamps = tstamps_no_ip_seconds,
    )    
    traj_est_no_interpolation_inverted = PoseTrajectory3D(
         positions_xyz = poses_no_interpolation_inverted[:, :3],
         orientations_quat_wxyz = poses_no_interpolation_inverted[:, 3:],
         timestamps = tstamps_no_ip_seconds,
    )
    traj_est_prescaled = PoseTrajectory3D(
        positions_xyz=1.10 * traj_est[:,:3],
        orientations_quat_wxyz=traj_est[:,3:],
        timestamps=np.array(tstamps_ns))
    
    traj_est_unscaled_ip = PoseTrajectory3D(
        positions_xyz = traj_est[:,:3],
        orientations_quat_wxyz=traj_est[:,3:],
        timestamps=np.array(tstamps_ns))
    
    traj_est_ip_unscaled_seconds = PoseTrajectory3D(
        positions_xyz = traj_est[:,:3],
        orientations_quat_wxyz=traj_est[:,3:],
        timestamps=np.array(tstamps_ns)*NANOSEC_TO_SEC)
    
    traj_est_no_ip_unscaled_ns = PoseTrajectory3D(
        positions_xyz = poses_no_interpolation_inverted[:,:3],
        orientations_quat_wxyz=poses_no_interpolation_inverted[:,3:],
        timestamps=tstamps_no_ip_seconds*SEC_TO_NANOSEC)

    # reading filestraj_est_unscaled_ip
    traj_ref = file_interface.read_tum_trajectory_file(args.gt)
    
    if args.stereo:
        logging.info("Stereo trajectory alignment and ape error computations")
        logging.debug("APE error for unscaled stereo trajectory")
        traj_ref = file_interface.read_tum_trajectory_file(args.gt)
        traj_ref, traj_est_unscaled = sync.associate_trajectories(traj_ref, traj_est_unscaled_ip)
        result_ip = main_ape.ape(traj_ref, traj_est_unscaled, est_name='traj', 
            pose_relation=PoseRelation.translation_part, align=True, correct_scale=False)
        
        logging.debug("APE error for unscaled and not - interpolated stereo trajectory")
        traj_ref = file_interface.read_tum_trajectory_file(args.gt)
        traj_ref, traj_est_no_ip_unscaled = sync.associate_trajectories(traj_ref, traj_est_no_ip_unscaled_ns)
        result_no_ip = main_ape.ape(traj_ref, traj_est_no_ip_unscaled, est_name='traj', 
            pose_relation=PoseRelation.translation_part, align=True, correct_scale=False)
        
        logging.info(f"EVO alignment and trajectory evaluation \n STEREO - UNSCALED INTERPOLATED \n {result_ip}")
        logging.info(f"EVO alignment and trajectory evaluation \n STEREO - UNSCALED NOT INTERPOLATED \n {result_no_ip}")
    else:
        logging.info("Mono Trajectory alignment and ape error computation")
        logging.debug("Computing APE error for prescaled trajectory")
        traj_ref = file_interface.read_tum_trajectory_file(args.gt)
        traj_ref, traj_est_prescaled = sync.associate_trajectories(traj_ref, traj_est_prescaled)
        result_scaled_before = main_ape.ape(traj_ref, traj_est_prescaled, est_name='traj', 
            pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)
        
        # unscaled trajectory
        logging.debug("Computing APE error for unscaled trajectory")
        traj_ref = file_interface.read_tum_trajectory_file(args.gt)
        traj_ref, traj_est_unscaled = sync.associate_trajectories(traj_ref, traj_est_unscaled_ip)
        result_unscaled_before = main_ape.ape(traj_ref, traj_est_unscaled, est_name='traj', 
            pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)
        
        # not interpolated and unscaled
        logging.debug("Computing APE error for not interpolated and not prescaled trajectory")
        traj_ref = file_interface.read_tum_trajectory_file(args.gt)
        traj_ref, traj_est_no_ip_unscaled = sync.associate_trajectories(traj_ref, traj_est_no_ip_unscaled_ns)
        result_no_ip_unscaled = main_ape.ape(traj_ref, traj_est_no_ip_unscaled, est_name='traj', 
            pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)
        
        
        logging.info(f'EVO alignment and trajectory evaluation \n  MONO - INTERPOLATED PRESCALED: \n {result_scaled_before}')
        logging.info(f'EVO alignment and trajectory evaluation \n  MONO - INTERPOLATED UNSCALED : \n {result_unscaled_before}')
        logging.info(f"EVO alignment and trajectory evaluation \n  MONO -  NO INTERPOLATION - UNSCALED: \n {result_no_ip_unscaled}")

    p = Path(args.reconstruction_path).joinpath(args.dataset_name,"evo", args.camera_name)
    p.mkdir(parents=True, exist_ok=True)
    file_interface.write_tum_trajectory_file(p.joinpath(f"traj_ref_{args.ba_tag}.txt").as_posix(), traj_ref)
    file_interface.write_tum_trajectory_file(p.joinpath(f"traj_est_no_ip_inv_{args.ba_tag}.txt").as_posix(), traj_est_no_interpolation_inverted)
    file_interface.write_tum_trajectory_file(p.joinpath(f"traj_est_no_ip_{args.ba_tag}.txt").as_posix(), traj_est_no_interpolation)
    file_interface.write_tum_trajectory_file(p.joinpath(f"traj_est_ip_no_prescale_{args.ba_tag}.txt").as_posix(), traj_est_ip_unscaled_seconds)
    
    if args.stereo:
        file_interface.write_tum_trajectory_file(p.joinpath(f"traj_est_{args.ba_tag}.txt").as_posix(), traj_est_unscaled)
        with open(p.joinpath(f"ape_result_ip_no_prescale_{args.ba_tag}.txt").as_posix(), 'w') as f:
            f.write(str(result_ip))
        with open(p.joinpath(f"ape_result_no_ip_no_prescale_{args.ba_tag}.txt").as_posix(), 'w') as f:
            f.write(str(result_no_ip))
    else:     
        file_interface.write_tum_trajectory_file(p.joinpath(f"traj_est_prescale_{args.ba_tag}.txt").as_posix(), traj_est_prescaled)
        file_interface.write_tum_trajectory_file(p.joinpath(f"traj_est_no_prescale_{args.ba_tag}.txt").as_posix(), traj_est_unscaled)
        with open(p.joinpath(f"ape_result_prescale_{args.ba_tag}.txt").as_posix(), 'w') as f:
            f.write(str(result_scaled_before))
        with open(p.joinpath(f"ape_result_no_prescale_{args.ba_tag}.txt").as_posix(), 'w') as f:
            f.write(str(result_unscaled_before))
        
        with open(p.joinpath(f"ape_result_no_ip_no_prescale_{args.ba_tag}.txt").as_posix(), 'w') as f:
            f.write(str(result_no_ip_unscaled))
    
    # save umeyama alignment results.
    logging.info("saving Umeyama alignment results")
    traj_ref, traj_est_unscaled = sync.associate_trajectories(traj_ref, traj_est_unscaled_ip)
    r_a_um, t_a_um, s_um = traj_est_unscaled.align(traj_ref, correct_scale = True)
    with open(p.joinpath(f"umeyama_alignment_ip_{args.ba_tag}.txt").as_posix(), 'w') as f:
        f.write(f"Rotation : \n {r_a_um}\n")
        f.write(f"Translation : \n {t_a_um}\n")
        f.write(f"Scale : \n {s_um}\n")
    
    traj_ref, traj_est_no_ip_unscaled = sync.associate_trajectories(traj_ref, traj_est_no_ip_unscaled_ns)
    r_a_um, t_a_um, s_um = traj_est_no_ip_unscaled.align(traj_ref, correct_scale = True)
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

    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=2.4)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--keyframe_thresh", type=float, default=3.5)
    parser.add_argument("--frontend_thresh", type=float, default=17.5)
    parser.add_argument("--frontend_window", type=int, default=20)
    parser.add_argument("--frontend_radius", type=int, default=2)
    parser.add_argument("--frontend_nms", type=int, default=1)

    parser.add_argument("--backend_thresh", type=float, default=24.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=2)
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
    

    for (t, image, intrinsics) in tqdm(image_stream(args.datapath, stereo=args.stereo, stride=2)):
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
    poses = {'no_ip': poses_no_interpolation,
             'no_ip_inv': poses_no_interpolation_inverted,
             'ip':traj_est
    }
    timestamps = {"t_no_ip_s": tstamps_no_ip_seconds,
                  "t_ip_ns": tstamps_ns
    }
    evaluate_trajectory(timestamps, poses, args)

    
    


