"""
Generate GT SMPL params from AIST++ original motion + camera data.

Pipeline per clip:
  1. SMPL decode: verts = (SMPL(pose) - j0) * scale + trans  [world, cm]
  2. Camera transform: v_cam = R_cam @ v_world + T_cam        [camera, cm]
  3. YZ flip: v_flip = v_cam * [1, -1, -1]                    [Y-up, cm]
  4. Gravity correction: v_grav = R_grav @ v_flip              [gravity-aligned, cm]
  5. Convert cm -> m

Steps 2-4 are folded into R_total applied to global_orient and trans.

Output npz: poses [T,72], trans [T,3], scale (scalar)
Decode: verts = (SMPL(poses) - j0) * scale + trans

Usage:
    python scripts/gen_gt_from_aist.py
"""
import os
import json
import pickle
import argparse
import numpy as np
import joblib
from scipy.spatial.transform import Rotation as sRot
from tqdm import tqdm


def load_mapping(camera_dir):
    mapping = {}
    with open(os.path.join(camera_dir, "mapping.txt")) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                mapping[parts[0]] = parts[1]
    return mapping


def load_cameras(camera_dir):
    cameras = {}
    for fname in os.listdir(camera_dir):
        if not fname.endswith('.json'): continue
        setting = fname.replace('.json', '')
        for cam in json.load(open(os.path.join(camera_dir, fname))):
            R = sRot.from_rotvec(np.array(cam['rotation'])).as_matrix()
            T = np.array(cam['translation'])
            cameras[(setting, cam['name'])] = (R, T)
    return cameras


def compute_transform(R_cam, T_cam):
    """Compute R_total = R_grav @ F @ R_cam and T_offset = R_grav @ F @ T_cam."""
    F = np.diag([1.0, -1.0, -1.0])
    grav = F @ R_cam @ np.array([0.0, -1.0, 0.0])
    target = np.array([0.0, -1.0, 0.0])
    axis = np.cross(grav, target)
    sin_a = np.linalg.norm(axis)
    cos_a = np.dot(grav, target)
    if sin_a > 1e-8:
        axis /= sin_a
        R_grav = sRot.from_rotvec(axis * np.arctan2(sin_a, cos_a)).as_matrix()
    else:
        R_grav = np.eye(3)
    R_total = R_grav @ F @ R_cam
    T_offset = R_grav @ F @ T_cam
    return R_total, T_offset


def main():
    parser = argparse.ArgumentParser(description='Generate GT SMPL params from AIST++')
    parser.add_argument('--aist_dir', type=str, default='data/AIST++', help='AIST++ data directory (with motions/ and cameras/)')
    parser.add_argument('--eval_pkl', type=str, default='data/eval/AIST++_test.pkl', help='Eval pkl to determine which clips to generate')
    parser.add_argument('--output_dir', type=str, default='data/gt/GT_AIST++', help='Output directory')
    args = parser.parse_args()

    motion_dir = os.path.join(args.aist_dir, "motions")
    camera_dir = os.path.join(args.aist_dir, "cameras")

    os.makedirs(args.output_dir, exist_ok=True)
    mapping = load_mapping(camera_dir)
    cameras = load_cameras(camera_dir)

    # Read clip names from eval pkl
    eval_data = joblib.load(args.eval_pkl)
    clip_names = sorted(eval_data.keys())
    print(f"Processing {len(clip_names)} clips from {args.eval_pkl}...")

    motion_cache = {}
    generated = 0
    skipped = 0

    for clip_name in tqdm(clip_names):
        out_path = os.path.join(args.output_dir, clip_name + '.npz')
        if os.path.exists(out_path):
            generated += 1
            continue

        # Parse: gBR_sBM_c01_d06_mBR2_ch01_00000
        parts = clip_name.split('_')
        cam_id = parts[2]
        clip_start_30fps = int(parts[-1])
        cAll_name = '_'.join(parts[:2]) + '_cAll_' + '_'.join(parts[3:-1])

        setting = mapping.get(cAll_name)
        if not setting:
            skipped += 1; continue
        cam_key = (setting, cam_id)
        if cam_key not in cameras:
            skipped += 1; continue
        R_cam, T_cam = cameras[cam_key]

        pkl_path = os.path.join(motion_dir, cAll_name + '.pkl')
        if cAll_name not in motion_cache:
            if not os.path.exists(pkl_path):
                skipped += 1; continue
            motion_cache[cAll_name] = pickle.load(open(pkl_path, 'rb'))
        motion = motion_cache[cAll_name]

        poses_all = motion['smpl_poses']
        trans_all = motion['smpl_trans']
        scale = float(motion['smpl_scaling'][0])

        # 60fps -> 30fps, skip first frame of clip
        clip_start_60fps = clip_start_30fps * 2
        idx_60 = np.arange(clip_start_60fps + 2, clip_start_60fps + 200, 2)
        if idx_60[-1] >= poses_all.shape[0]:
            skipped += 1; continue

        poses = poses_all[idx_60].copy()
        trans = trans_all[idx_60].copy()

        R_total, T_offset = compute_transform(R_cam, T_cam)

        # Apply R_total to global_orient
        root_rot = sRot.from_rotvec(poses[:, :3])
        poses[:, :3] = sRot.from_matrix(R_total[None] @ root_rot.as_matrix()).as_rotvec().astype(np.float32)

        # Transform translation: new_trans = R_total @ trans + T_offset, then cm -> m
        new_trans = (R_total @ trans.T).T + T_offset
        new_trans_m = (new_trans / 100.0).astype(np.float32)
        scale_m = np.float32(scale / 100.0)

        np.savez_compressed(out_path,
                           poses=poses.astype(np.float32),
                           trans=new_trans_m,
                           scale=scale_m)
        generated += 1

    print(f"\nDone. Generated: {generated}, Skipped: {skipped}")


if __name__ == '__main__':
    main()
