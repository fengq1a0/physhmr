"""
Visualize and evaluate exported PhysHMR results.

Usage:
    # Evaluate metrics against GT (main use case)
    python scripts/visualize.py export/AIST++_test/ --eval --gt_path data/gt_verts/GT_AIST++

    # Render video for a single sequence
    python scripts/visualize.py export/AIST++_test/gLH_sBM_c01_d18_mLH2_ch06_00000.npz

    # Render videos for all sequences in a directory
    python scripts/visualize.py export/AIST++_test/

    # Only save PLY meshes, no video
    python scripts/visualize.py export/AIST++_test/ --save_mesh --no_video

    # Eval + render videos
    python scripts/visualize.py export/AIST++_test/ --eval --gt_path data/gt_verts/GT_AIST++
"""
import os
import argparse
import numpy as np
import torch
import open3d as o3d
from smplx import SMPL
from tqdm import tqdm
import cv2


# ========================== SMPL Forward ==========================

def smpl_forward(smpl_model, poses, trans, R_isaac2smpl, scale, device):
    """Run SMPL forward pass for predictions (Isaac space -> Y-up). Returns vertices [T, 6890, 3]."""
    with torch.no_grad():
        poses_t = torch.from_numpy(poses.reshape(-1, 72)).float().to(device)
        trans_t = torch.from_numpy(trans).float().to(device)
        R_t = torch.from_numpy(R_isaac2smpl).float().to(device)
        res = smpl_model(
            global_orient=poses_t[:, :3],
            body_pose=poses_t[:, 3:],
        )
        verts = res.vertices - res.joints[:, 0:1, :] + trans_t[:, None, :]
        verts = (verts @ R_t * float(np.array(scale).flatten()[0])).cpu().numpy()
    return verts, smpl_model.faces


def smpl_forward_gt(smpl_model, poses, trans, scale, device):
    """Run SMPL forward pass for GT data. verts = (SMPL(pose) - j0) * scale + trans."""
    with torch.no_grad():
        poses_t = torch.from_numpy(poses.reshape(-1, 72)).float().to(device)
        trans_t = torch.from_numpy(trans).float().to(device)
        s = float(np.array(scale).flatten()[0])
        res = smpl_model(
            global_orient=poses_t[:, :3],
            body_pose=poses_t[:, 3:],
        )
        j0 = res.joints[:, 0:1, :]
        verts = ((res.vertices - j0) * s + trans_t[:, None, :]).cpu().numpy()
    return verts, smpl_model.faces


# ========================== Metrics ==========================

def p_mpjpe(target, predicted):
    """PA-MPJPE: MPJPE after Procrustes alignment (scale, rotation, translation)."""
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    X0 = target - muX
    Y0 = predicted - muY
    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    X0 /= normX
    Y0 /= normY
    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))
    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)
    a = tr * normX / normY
    t = muX - a * np.matmul(muY, R)
    predicted_aligned = a * np.matmul(predicted, R) + t
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=-1)) * 1000


def local_mpjpe(target, predicted):
    """Root-relative MPJPE."""
    t_tmp = target - target[:, 0:1, :]
    p_tmp = predicted - predicted[:, 0:1, :]
    return np.mean(np.linalg.norm(p_tmp - t_tmp, axis=-1)) * 1000


def wa_mpjpe(GT, PR, window_size=100):
    """Windowed and Aligned MPJPE: translation-align per 100-frame window."""
    GT = torch.from_numpy(GT) if isinstance(GT, np.ndarray) else GT
    PR = torch.from_numpy(PR) if isinstance(PR, np.ndarray) else PR
    num_frames = GT.shape[0]
    total = 0.0
    for start in range(0, num_frames, window_size):
        end = min(start + window_size, num_frames)
        gt_seg = GT[start:end].clone()
        pr_seg = PR[start:end].clone()
        translation = gt_seg[:1, 0].mean(dim=0) - pr_seg[:1, 0].mean(dim=0)
        pr_seg_aligned = pr_seg + translation
        total += torch.norm(gt_seg - pr_seg_aligned, dim=-1).mean().item() * (end - start)
    return total / num_frames * 1000


def procrustes_align_sequence(A, B):
    """Align sequence A to B using Procrustes (rotation + translation)."""
    A_flat = A.reshape(-1, 3)
    B_flat = B.reshape(-1, 3)
    centroid_A = A_flat.mean(axis=0)
    centroid_B = B_flat.mean(axis=0)
    AA = A_flat - centroid_A
    BB = B_flat - centroid_B
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    T = centroid_B - R @ centroid_A
    C_flat = (R @ A_flat.T).T + T
    return C_flat.reshape(A.shape)


def compute_foot_sliding(target_verts, pred_verts, thr=1e-2):
    """Foot sliding error (from GVHMR). Input: [N, 6890, 3] torch tensors."""
    foot_idxs = [3216, 3387, 6617, 6787]
    foot_loc = target_verts[:, foot_idxs]
    foot_disp = (foot_loc[1:] - foot_loc[:-1]).norm(2, dim=-1)
    contact = foot_disp < thr
    pred_feet_loc = pred_verts[:, foot_idxs]
    pred_disp = (pred_feet_loc[1:] - pred_feet_loc[:-1]).norm(2, dim=-1)
    error = pred_disp[contact]
    if error.shape[0] == 0:
        return 0
    return error.cpu().numpy().mean() * 1000


def height_variance(verts, k=0.25):
    """Variance of lowest y-coordinates on the lowest k proportion."""
    y_min = np.min(verts[:, :, 1], axis=1)
    y_min_sorted = np.sort(y_min)
    cutoff = int(len(y_min_sorted) * k)
    return np.var(y_min_sorted[:cutoff])


def velocity_error(pr, gt):
    pr = torch.from_numpy(pr) if isinstance(pr, np.ndarray) else pr
    gt = torch.from_numpy(gt) if isinstance(gt, np.ndarray) else gt
    return torch.norm((pr[1:, 0] - pr[:-1, 0]) - (gt[1:, 0] - gt[:-1, 0]), dim=1).mean().item()


def acceleration_error(pr, gt):
    pr = torch.from_numpy(pr) if isinstance(pr, np.ndarray) else pr
    gt = torch.from_numpy(gt) if isinstance(gt, np.ndarray) else gt
    pr_acc = pr[2:, 0] - 2 * pr[1:-1, 0] + pr[:-2, 0]
    gt_acc = gt[2:, 0] - 2 * gt[1:-1, 0] + gt[:-2, 0]
    return torch.norm(pr_acc - gt_acc, dim=1).mean().item()


# ========================== Visualization ==========================

def render_mesh_sequence(verts, faces, output_path=None, fps=30, resolution=(1024, 768)):
    """Render a mesh sequence to video using Open3D offscreen rendering."""
    W, H = resolution
    renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = [0.7, 0.7, 0.9, 1.0]

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts[0])
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    traj_center = verts[:, 0, :].mean(axis=0)
    ground_size = 50
    ground = o3d.geometry.TriangleMesh.create_box(width=ground_size, height=0.01, depth=ground_size)
    ground.translate([traj_center[0] - ground_size / 2, -0.01, traj_center[2] - ground_size / 2])
    ground.paint_uniform_color([0.8, 0.8, 0.8])
    ground.compute_vertex_normals()
    renderer.scene.add_geometry("ground", ground, mat)

    center = verts[0].mean(axis=0)
    eye = center + np.array([0, 1.5, 4.0])
    renderer.setup_camera(60.0, center, eye, [0, 1, 0])

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    frames = []
    for t in range(verts.shape[0]):
        mesh.vertices = o3d.utility.Vector3dVector(verts[t])
        mesh.compute_vertex_normals()
        if renderer.scene.has_geometry("body"):
            renderer.scene.remove_geometry("body")
        renderer.scene.add_geometry("body", mesh, mat)
        img = np.asarray(renderer.render_to_image())
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if writer:
            writer.write(img_bgr)
        frames.append(img_bgr)

    if writer:
        writer.release()
    del renderer
    return frames


def save_ply_sequence(verts, faces, output_dir):
    """Save each frame as a PLY file."""
    os.makedirs(output_dir, exist_ok=True)
    for t in range(verts.shape[0]):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts[t])
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(os.path.join(output_dir, f"{t:05d}.ply"), mesh)


# ========================== Main ==========================

def main():
    parser = argparse.ArgumentParser(description='Visualize and evaluate PhysHMR results')
    parser.add_argument('input', type=str, help='Path to .npz file or directory of .npz files')
    parser.add_argument('--eval', action='store_true', help='Compute evaluation metrics against GT')
    parser.add_argument('--gt_path', type=str, default=None, help='Path to GT vertex npy directory (required for --eval)')
    parser.add_argument('--j_regressor', type=str, default='data/J_regressor.npy', help='Path to SMPL J_regressor')
    parser.add_argument('--save_mesh', action='store_true', help='Save PLY meshes')
    parser.add_argument('--no_video', action='store_true', help='Skip video rendering')
    parser.add_argument('--smpl_path', type=str, default='data/smpl', help='Path to SMPL model')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    args = parser.parse_args()

    do_video = not args.no_video and not args.eval  # default: video unless --eval
    do_mesh = args.save_mesh
    do_eval = args.eval

    if args.eval and args.gt_path is None:
        print("Error: --eval requires --gt_path")
        return

    if not do_video and not do_mesh and not do_eval:
        print("Nothing to do. Use --eval, --save_mesh, or remove --no_video.")
        return

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    smpl_model = SMPL(model_path=args.smpl_path, gender="neutral").to(device)

    if os.path.isfile(args.input) and args.input.endswith('.npz'):
        files = [args.input]
    elif os.path.isdir(args.input):
        files = sorted([os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith('.npz')])
    else:
        print(f"Error: {args.input} is not a valid .npz file or directory")
        return

    J_R = None
    if do_eval:
        J_R = np.load(args.j_regressor)

    metrics = {"PA": [], "WA": [], "MPJPE": [], "FS": [], "HV": [], "ACC": [], "VEL": []}
    total_count = 0
    skipped_count = 0
    skipped_names = []

    print(f"Processing {len(files)} sequences...")
    for f in tqdm(files):
        data = np.load(f)
        poses = data['poses']
        trans = data['trans']
        scale = data['scale']
        R = data['extra_R']
        name = os.path.splitext(os.path.basename(f))[0]
        parent_dir = os.path.dirname(f)

        verts, faces = smpl_forward(smpl_model, poses, trans, R, scale, device)

        if do_eval:
            # Support both legacy .npy (verts) and new .npz (SMPL params) GT formats
            gt_npz = os.path.join(args.gt_path, name + '.npz')
            gt_npy = os.path.join(args.gt_path, name + '.npy')
            if os.path.exists(gt_npz):
                gt_data = np.load(gt_npz)
                gt_verts, _ = smpl_forward_gt(smpl_model, gt_data['poses'], gt_data['trans'],
                                              gt_data['scale'], device)
            elif os.path.exists(gt_npy):
                gt_verts = np.load(gt_npy)
            else:
                continue
            gt_verts = gt_verts[:verts.shape[0]]

            # Align prediction to GT
            verts_aligned = procrustes_align_sequence(verts, gt_verts)

            # Align prediction to GT and compute joint positions
            pr_joints = J_R @ verts_aligned  # [T, 24, 3]
            gt_joints = J_R @ gt_verts

            pa = p_mpjpe(gt_joints, pr_joints)
            wa = wa_mpjpe(gt_joints, pr_joints)
            total_count += 1
            if pa > 100:
                skipped_count += 1
                skipped_names.append((name, pa, wa))
                continue  # skip failed sequences

            metrics["PA"].append(pa)
            metrics["WA"].append(wa)
            metrics["MPJPE"].append(local_mpjpe(gt_joints, pr_joints))
            metrics["FS"].append(compute_foot_sliding(
                torch.from_numpy(gt_verts), torch.from_numpy(verts_aligned)))
            metrics["HV"].append(height_variance(verts))
            metrics["ACC"].append(acceleration_error(pr_joints, gt_joints))
            metrics["VEL"].append(velocity_error(pr_joints, gt_joints))

        if do_mesh:
            save_ply_sequence(verts, faces, os.path.join(parent_dir, 'mesh', name))

        if do_video:
            video_dir = os.path.join(parent_dir, 'video')
            os.makedirs(video_dir, exist_ok=True)
            render_mesh_sequence(verts, faces, output_path=os.path.join(video_dir, name + '.mp4'))

    if do_eval:
        print("\n==================== Results ====================")
        print(f"  Total: {total_count}, Skipped (PA>100): {skipped_count}, Evaluated: {len(metrics['PA'])}")
        if skipped_names:
            input_name = os.path.basename(args.input.rstrip('/'))
            skipped_file = os.path.join(os.path.dirname(args.input.rstrip('/')), f'{input_name}_skipped.txt')
            with open(skipped_file, 'w') as sf:
                for sn, spa, swa in skipped_names:
                    sf.write(f"{sn}\tPA={spa:.2f}\tWA={swa:.2f}\n")
            print(f"  Skipped names saved to: {skipped_file}")
        print(f"  PA-MPJPE:  {np.mean(metrics['PA']):.2f}")
        print(f"  WA-MPJPE:  {np.mean(metrics['WA']):.2f}")
        print(f"  MPJPE:     {np.mean(metrics['MPJPE']):.2f}")
        print(f"  Foot Slid: {np.mean(metrics['FS']):.2f}")
        print(f"  Height V:  {np.mean(metrics['HV']) * 1e6:.2f}")
        print(f"  Accel Err: {np.mean(metrics['ACC']) * 1000:.2f}")
        print(f"  Vel Err:   {np.mean(metrics['VEL']) * 1000:.2f}")
        print("=================================================")

    print("Done.")


if __name__ == '__main__':
    main()
