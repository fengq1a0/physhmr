import os
import sys
sys.path.append(os.getcwd())

import torch 
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import joblib
from tqdm import tqdm
import argparse
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
from smplx import SMPL



def patch_numpy_core_alias():
    """
    Make NumPy 1.x able to import numpy._core (NumPy 2.x internal path)
    so that pickles created under NumPy 2.x can be unpickled.
    """
    if "numpy._core" in sys.modules:
        return

    import numpy.core as core

    # Alias numpy._core -> numpy.core
    sys.modules["numpy._core"] = core

    # Common submodules referenced by pickles / numpy internals
    submods = [
        "multiarray",
        "_multiarray_umath",
        "umath",
        "numeric",
        "numerictypes",
        "_exceptions",
    ]
    for name in submods:
        old = f"numpy.core.{name}"
        new = f"numpy._core.{name}"
        if old in sys.modules:
            sys.modules[new] = sys.modules[old]
        else:
            try:
                __import__(old)
                sys.modules[new] = sys.modules[old]
            except Exception:
                # Not all submodules exist in all numpy builds; ignore.
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess GVHMR and TRAM outputs for PhysHMR')
    parser.add_argument('--gvhmr_root', type=str, required=True,
                        help='Path to GVHMR output directory')
    parser.add_argument('--tram_root', type=str, required=True,
                        help='Path to TRAM output directory')
    parser.add_argument('--out_root', type=str, required=True,
                        help='Output directory for processed data')
    parser.add_argument('--video_name', type=str, default='demo',
                        help='Name for the output sequence')
    parser.add_argument('--use_vitpose', action='store_true',
                        help='Use ViTPose keypoints from GVHMR instead of TRAM keypoints')
    args = parser.parse_args()

    patch_numpy_core_alias()

    # Create output directory
    os.makedirs(args.out_root, exist_ok=True)
    
    print(f"Processing GVHMR output from: {args.gvhmr_root}")
    print(f"Processing TRAM output from: {args.tram_root}")
    print(f"Output will be saved to: {args.out_root}")
    print("---------------------------------------------------")
    
    # ===== Load GVHMR outputs =====
    gvhmr_results_path = os.path.join(args.gvhmr_root, "hmr4d_results.pt")
    if not os.path.exists(gvhmr_results_path):
        raise FileNotFoundError(f"GVHMR results not found at: {gvhmr_results_path}")
    
    print(f"Loading GVHMR results from: {gvhmr_results_path}")
    gvhmr_output = torch.load(gvhmr_results_path, weights_only=True)
    image_feature = gvhmr_output["net_outputs"]["model_output"]["pred_context"][0].numpy()
    smpl_params = gvhmr_output['smpl_params_global']
    
    transl = smpl_params['transl'].numpy().astype(np.float64)
    body_pose_tmp = smpl_params['body_pose'].numpy().astype(np.float64)
    body_pose = np.zeros((body_pose_tmp.shape[0], 23*3), dtype=np.float64)
    body_pose[:, :-6] = body_pose_tmp
    global_orient = smpl_params['global_orient'].numpy().astype(np.float64)
    betas = smpl_params['betas'].numpy().astype(np.float64)
    betas = betas.mean(axis=0)[None]
    
    # ===== Load TRAM outputs =====
    tram_camera_path = os.path.join(args.tram_root, "camera.npy")
    tram_hps_path = os.path.join(args.tram_root, "hps", "hps_track_0.npy")
    
    if not os.path.exists(tram_camera_path):
        raise FileNotFoundError(f"TRAM camera not found at: {tram_camera_path}")
    if not os.path.exists(tram_hps_path):
        raise FileNotFoundError(f"TRAM HPS not found at: {tram_hps_path}")
    
    print(f"Loading TRAM camera from: {tram_camera_path}")
    print(f"Loading TRAM HPS from: {tram_hps_path}")
    
    tram_cam = np.load(tram_camera_path, allow_pickle=True).item()
    tram_hps = np.load(tram_hps_path, allow_pickle=True).item()
    
    # ===== Load 2D keypoints =====
    if args.use_vitpose:
        # Load ViTPose keypoints from GVHMR
        vitpose_path = os.path.join(args.gvhmr_root, "preprocess", "vitpose.pt")
        if not os.path.exists(vitpose_path):
            raise FileNotFoundError(f"ViTPose not found at: {vitpose_path}")
        
        print(f"Loading ViTPose keypoints from: {vitpose_path}")
        vitpose_data = torch.load(vitpose_path, weights_only=True)
        # ViTPose format: [N, num_joints, 3] with (x, y, confidence)
        kp2d = vitpose_data.numpy().astype(np.float64)
        # Add dummy z coordinate to match expected format [N, num_joints, 4]
        if kp2d.shape[-1] == 3:
            kp2d_4d = np.zeros((kp2d.shape[0], kp2d.shape[1], 4), dtype=np.float64)
            kp2d_4d[:, :, :3] = kp2d
            kp2d = kp2d_4d

            # TODO: This part maybe not True
    else:
        # Extract 2D keypoints from TRAM HPS
        print(f"Using TRAM keypoints from HPS")
        if 'keypoints_2d' in tram_hps:
            kp2d = tram_hps['keypoints_2d'].astype(np.float64)  # [N, num_joints, 3 or 4]
            # Ensure 4D format
            if kp2d.shape[-1] == 3:
                kp2d_4d = np.zeros((kp2d.shape[0], kp2d.shape[1], 4), dtype=np.float64)
                kp2d_4d[:, :, :3] = kp2d
                kp2d = kp2d_4d
        else:
            print("Warning: keypoints_2d not found in TRAM HPS, creating dummy keypoints")
            kp2d = np.zeros((transl.shape[0], 10, 4), dtype=np.float64)
    
    
    camera = np.zeros((tram_cam['world_cam_R'].shape[0],4,4), dtype=np.float64)
    camera[:,:3,:3] = tram_cam['world_cam_R'].astype(np.float64)
    camera[:,:3, 3] = tram_cam['world_cam_T'].astype(np.float64)
    camera[:, 3, 3] = 1.0

    print(f"Loaded {transl.shape[0]} frames from GVHMR")
    print(f"Loaded {camera.shape[0]} frames from TRAM")
    
    # Align sequence lengths
    N = min(transl.shape[0], camera.shape[0])
    transl = transl[:N]
    body_pose = body_pose[:N]
    global_orient = global_orient[:N]
    image_feature = image_feature[:N]
    camera = camera[:N]
    if kp2d.shape[0] > N:
        kp2d = kp2d[:N]
    
    #-----------------------------------------------------------------------
    upright_start = True
    robot_cfg = {
            "mesh": False,
            "rel_joint_lm": True,
            "upright_start": upright_start,
            "remove_toe": False,
            "real_weight": True,
            "real_weight_porpotion_capsules": True,
            "real_weight_porpotion_boxes": True, 
            "replace_feet": True,
            "masterfoot": False,
            "big_ankle": True,
            "freeze_hand": False, 
            "box_body": False,
            "master_range": 50,
            "body_params": {},
            "joint_params": {},
            "geom_params": {},
            "actuator_params": {},
            "model": "smpl",
        }
    smpl_local_robot = LocalRobot(robot_cfg,)
    smpl = SMPL(model_path="data/smpl", gender="neutral")
    smpl_2_mujoco = [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES]
    
    #-----------------------------------------------------------------------
    # Normalize scale using SMPL shape
    print("Computing scale normalization...")
    with torch.no_grad():
        joints = smpl(
            betas=torch.from_numpy(betas).float(),
            global_orient=torch.zeros((betas.shape[0], 3)),
            body_pose=torch.zeros((betas.shape[0], 23*3))
        ).joints.numpy()
        
        jjjj = smpl(
            betas=torch.zeros(betas.shape).float(),
            global_orient=torch.zeros((betas.shape[0], 3)),
            body_pose=torch.zeros((betas.shape[0], 23*3))
        ).joints.numpy()
        
        j0 = jjjj[:, 0]
        joints = joints[0]
        jjjj = jjjj[0]
        pc1_centered = joints - np.mean(joints, axis=0)
        pc2_centered = jjjj - np.mean(jjjj, axis=0)
        scale_rms = np.linalg.norm(pc1_centered) / np.linalg.norm(pc2_centered)
        scale = 1 / scale_rms
    
    betas = np.zeros_like(betas)
    
    # Apply scale normalization
    print(f"Applying scale: {scale:.4f}")
    transl = transl / scale
    if kp2d.shape[-1] >= 3:
        kp2d[:, :, :3] = kp2d[:, :, :3] / scale
    camera = camera / scale
    
    # Compute j0 for all frames
    with torch.no_grad():
        j0 = smpl(
            global_orient=torch.zeros((body_pose.shape[0], 3)),
            body_pose=torch.zeros((body_pose.shape[0], 23*3))
        ).joints[:, 0].numpy()
    
    #-----------------------------------------------------------------------
    # Coordinate transformation: SMPL (Y-up) -> Isaac Gym (Z-up)
    print("Applying coordinate transformations...")
    RRRR = np.array([[ 1.0,  0.0,  0.0, 0],
                     [ 0.0,  0.0, -1.0, 0], 
                     [ 0.0,  1.0,  0.0, 0],
                     [   0,    0,    0, 1]])
    
    # Transform for Isaac Gym
    poses_isaac = np.zeros((body_pose.shape[0], 24*3), dtype=np.float64)
    poses_isaac[:, 3:] = body_pose
    root_rot = sRot.from_rotvec(global_orient).as_matrix()
    poses_isaac[:, :3] = sRot.from_matrix(RRRR[:3, :3] @ root_rot).as_rotvec()
    
    after_j0 = (j0 @ RRRR[:3, :3].T)
    trans_isaac = (transl + j0) @ RRRR[:3, :3].T + RRRR[:3, 3] - after_j0
    
    # Transform camera and keypoints
    camera = RRRR @ camera
    if kp2d.shape[-1] >= 3:
        kp2d[:, :, :3] = kp2d[:, :, :3] @ RRRR[:3, :3].T + RRRR[:3, 3]
    
    # Keep original SMPL parameters (for reference)
    poses_g = np.zeros((body_pose.shape[0], 24*3), dtype=np.float64)
    poses_g[:, 3:] = body_pose
    poses_g[:, :3] = global_orient
    trans_g = transl
    
    #-----------------------------------------------------------------------
    # Convert to quaternions for physics simulation
    print("Converting to quaternion representation...")
    N = poses_isaac.shape[0]
    pose_aa_mj = poses_isaac.reshape(N, 24, 3)[:, smpl_2_mujoco]
    pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(N, 24, 4)
    
    # Build skeleton tree
    smpl_local_robot.load_from_skeleton(betas=torch.zeros(1, 10), gender=[0], objs_info=None)
    smpl_local_robot.write_xml(f"phc/data/assets/mjcf/{robot_cfg['model']}_humanoid.xml")
    skeleton_tree = SkeletonTree.from_mjcf(f"phc/data/assets/mjcf/{robot_cfg['model']}_humanoid.xml")
    root_trans_offset = torch.from_numpy(trans_isaac) + after_j0
    
    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        torch.from_numpy(pose_quat),
        root_trans_offset,
        is_local=True)
    
    if robot_cfg['upright_start']:
        pose_quat_global = (sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) * 
                           sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(N, -1, 4)
        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree, torch.from_numpy(pose_quat_global), root_trans_offset, is_local=False)
        pose_quat = new_sk_state.local_rotation.numpy()
    
    pose_quat_global = new_sk_state.global_rotation.numpy()
    pose_quat = new_sk_state.local_rotation.numpy()
    
    #-----------------------------------------------------------------------
    # Package output data
    print(f"Packaging output for {N} frames...")
    motion_dict = {}
    new_motion_out = {
        'pose_quat_global': pose_quat_global,
        'pose_quat': pose_quat,
        'root_trans_offset': root_trans_offset.numpy(),
        'pose_aa': poses_isaac,
        'img_feat': image_feature,
        'camera': camera.astype(np.float32),
        'kp2d': kp2d.astype(np.float32),
        'fps': 30,
        'gender': 'neutral',
        'scale': scale,
        'poses_g': poses_g,
        'trans_g': trans_g,
    }
    motion_dict[args.video_name] = new_motion_out
    
    # Save output
    output_path = os.path.join(args.out_root, f"{args.video_name}.pkl")
    joblib.dump(motion_dict, output_path, compress=True)
    print(f"✓ Successfully saved processed data to: {output_path}")
    print(f"  - Frames: {N}")
    print(f"  - Scale: {scale:.4f}")
    print(f"  - FPS: 30")
