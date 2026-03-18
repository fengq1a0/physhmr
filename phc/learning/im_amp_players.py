import os
import sys
sys.path.append(os.getcwd())

import time
import numpy as np
import torch
from rl_games.common.tr_helpers import unsqueeze_obs
from phc.learning import amp_players
from phc.utils.flags import flags

from tqdm import tqdm

COLLECT_Z = False

def compute_metrics_lite(pred, gt):
    """ Quick MPJPE computation for progress tracking """
    res = {"mpjpe_g": [], "mpjpe_l": [], "mpjpe_pa": [], "accel_dist": [], "vel_dist": []}
    for p, g in zip(pred, gt):
        if len(p) < 2: continue
        res["mpjpe_g"].append(np.linalg.norm(p - g, axis=-1).mean() * 1000)
        p_local = p - p[:, 0:1]
        g_local = g - g[:, 0:1]
        res["mpjpe_l"].append(np.linalg.norm(p_local - g_local, axis=-1).mean() * 1000)
        res["accel_dist"].append(np.linalg.norm((p[2:] + p[:-2] - 2*p[1:-1]) - (g[2:] + g[:-2] - 2*g[1:-1]), axis=-1).mean() * 1000)
        res["vel_dist"].append(np.linalg.norm((p[1:] - p[:-1]) - (g[1:] - g[:-1]), axis=-1).mean() * 1000)
    return res


class IMAMPPlayerContinuous(amp_players.AMPPlayerContinuous):
    def __init__(self, config):
        super().__init__(config)

        self.terminate_state = torch.zeros(
            self.env.task.num_envs, device=self.device
        )
        self.terminate_memory = []

        self.mpjpe, self.mpjpe_all = [], []
        self.gt_pos, self.gt_pos_all = [], []
        self.gt_rot, self.gt_rot_all = [], []
        self.pred_pos, self.pred_pos_all = [], []
        self.pred_rot, self.pred_rot_all = [], []
        self.curr_stpes = 0
        self.keys_all = []
        self.scale_all = []

        if COLLECT_Z:
            self.zs, self.zs_all = [], []

        humanoid_env = self.env.task
        humanoid_env._termination_distances[:] = 0.5
        humanoid_env._recovery_episode_prob, humanoid_env._fall_init_prob = 0, 0

        if humanoid_env.collect_dataset:
            self.obs_buf, self.obs_buf_all = [], []
            self.env_actions, self.actions_all = [], []
            self.motion_length_all = []
            self.clean_actions, self.clean_actions_all = [], []
            self.reset_buf, self.reset_buf_all = [], []

        if flags.im_eval:
            self.success_rate = 0
            self.pbar = tqdm(range(humanoid_env._motion_lib._num_unique_motions // humanoid_env.num_envs))
            humanoid_env.zero_out_far = False
            humanoid_env.zero_out_far_train = False

            if len(humanoid_env._reset_bodies_id) > 15:
                humanoid_env._reset_bodies_id = humanoid_env._eval_track_bodies_id

            humanoid_env.cycle_motion = False
            self.print_stats = False

        return

    def _post_step(self, info, done):
        super()._post_step(info)

        if flags.im_eval:

            humanoid_env = self.env.task

            termination_state = torch.logical_and(self.curr_stpes <= humanoid_env._motion_lib.get_motion_num_steps() - 1, info["terminate"])
            self.terminate_state = torch.logical_or(termination_state, self.terminate_state)
            if (~self.terminate_state).sum() > 0:
                max_possible_id = humanoid_env._motion_lib._num_unique_motions - 1
                curr_ids = humanoid_env._motion_lib._curr_motion_ids
                if (max_possible_id == curr_ids).sum() > 0:
                    bound = (max_possible_id == curr_ids).nonzero()[0] + 1
                    if (~self.terminate_state[:bound]).sum() > 0:
                        curr_max = humanoid_env._motion_lib.get_motion_num_steps()[:bound][~self.terminate_state[:bound]].max()
                    else:
                        curr_max = (self.curr_stpes - 1)
                else:
                    curr_max = humanoid_env._motion_lib.get_motion_num_steps()[~self.terminate_state].max()

                if self.curr_stpes >= curr_max: curr_max = self.curr_stpes + 1
            else:
                curr_max = humanoid_env._motion_lib.get_motion_num_steps().max()

            if humanoid_env.collect_dataset:
                self.obs_buf.append(info['obs_buf'])
                self.clean_actions.append(info['clean_actions'])
                self.env_actions.append(info['actions'])
                self.reset_buf.append(info['reset_buf'])

            self.mpjpe.append(info["mpjpe"])
            self.gt_pos.append(info["body_pos_gt"].copy())
            self.gt_rot.append(info["body_rot_gt"].copy())
            self.pred_pos.append(info["body_pos"].copy())
            self.pred_rot.append(info["body_rot"].copy())
            if COLLECT_Z: self.zs.append(info["z"])
            self.curr_stpes += 1

            if self.curr_stpes >= curr_max or self.terminate_state.sum() == humanoid_env.num_envs:

                self.terminate_memory.append(self.terminate_state.cpu().numpy())

                # MPJPE
                all_mpjpe = torch.stack(self.mpjpe)
                assert(all_mpjpe.shape[0] == curr_max or self.terminate_state.sum() == humanoid_env.num_envs)

                all_mpjpe = [all_mpjpe[: (i - 1), idx].mean() for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
                all_body_pos_pred = np.stack(self.pred_pos)
                all_body_pos_pred = [all_body_pos_pred[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]

                all_body_rot_pred = np.stack(self.pred_rot)
                all_body_rot_pred = [all_body_rot_pred[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]

                all_body_pos_gt = np.stack(self.gt_pos)
                all_body_pos_gt = [all_body_pos_gt[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]

                all_body_rot_gt = np.stack(self.gt_rot)
                all_body_rot_gt = [all_body_rot_gt[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]

                if COLLECT_Z:
                    all_zs = torch.stack(self.zs)
                    all_zs = [all_zs[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
                    self.zs_all += all_zs

                if humanoid_env.collect_dataset:
                    all_obs_buf = np.stack(self.obs_buf)
                    all_obs_buf = [all_obs_buf[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
                    self.obs_buf_all += all_obs_buf

                    all_clean_actions = np.stack(self.clean_actions)
                    all_clean_actions = [all_clean_actions[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
                    self.clean_actions_all += all_clean_actions

                    all_actions = np.stack(self.env_actions)
                    all_actions = [all_actions[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
                    self.actions_all += all_actions

                    all_reset_buf = np.stack(self.reset_buf)
                    all_reset_buf = [all_reset_buf[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
                    self.reset_buf_all += all_reset_buf

                    self.motion_length_all += [obs.shape[0] for obs in all_obs_buf]

                self.mpjpe_all.append(all_mpjpe)
                self.pred_pos_all += all_body_pos_pred
                self.gt_pos_all += all_body_pos_gt
                self.gt_rot_all += all_body_rot_gt
                self.pred_rot_all += all_body_rot_pred
                self.keys_all += humanoid_env._motion_lib.curr_motion_keys.tolist()
                self.scale_all += humanoid_env._motion_lib.curr_scale.tolist()

                if (humanoid_env.start_idx + humanoid_env.num_envs >= humanoid_env._motion_lib._num_unique_motions):
                    num_unique = humanoid_env._motion_lib._num_unique_motions
                    pred_pos_all = self.pred_pos_all[:num_unique]
                    gt_pos_all = self.gt_pos_all[:num_unique]
                    gt_rot_all = self.gt_rot_all[:num_unique]
                    pred_rot_all = self.pred_rot_all[:num_unique]

                    if flags.real_traj:
                        pred_pos_all = [i[:, humanoid_env._reset_bodies_id] for i in pred_pos_all]
                        gt_pos_all = [i[:, humanoid_env._reset_bodies_id] for i in gt_pos_all]

                    # Export results; compute final metrics offline from exported files
                    motion_name = humanoid_env.cfg.env.motion_file.split('/')[-1].split('.')[0]
                    export_path = './export/%s/' % motion_name
                    os.makedirs(export_path, exist_ok=True)

                    # Isaac (Z-up) -> SMPL (Y-up), consistent with phys2kin.py
                    R_isaac2smpl = np.array([[ 1.0,  0.0,  0.0],
                                             [ 0.0,  0.0, -1.0],
                                             [ 0.0,  1.0,  0.0]], dtype=np.float32)
                    for i in tqdm(range(len(pred_pos_all)), desc="Exporting"):
                        key_name = self.keys_all[i].replace("/", "_")
                        poses = pred_rot_all[i][:, humanoid_env.mujoco_2_smpl, :].reshape(-1, 72)  # [T, 72]
                        trans = pred_pos_all[i][:, humanoid_env.mujoco_2_smpl, :][:, 0]  # [T, 3]
                        scale = self.scale_all[i]

                        # Save SMPL params
                        np.savez(os.path.join(export_path, key_name),
                                    poses = poses,
                                    trans = trans,
                                    scale = scale,
                                    extra_R = R_isaac2smpl,
                                )

                    print("------------------------------------------")
                    print(f"Exported {len(pred_pos_all)} motions to {export_path}")
                    print(f"  Use scripts/visualize.py to render videos or save meshes.")
                    exit()

                done[:] = 1  # Turning all of the sequences done and reset for the next batch of eval.

                humanoid_env.forward_motion_samples()
                self.terminate_state = torch.zeros(
                    self.env.task.num_envs, device=self.device
                )

                self.pbar.update(1)
                self.pbar.refresh()
                self.mpjpe, self.gt_pos, self.pred_pos = [], [], []
                self.gt_rot, self.pred_rot = [], []
                if humanoid_env.collect_dataset:
                    self.obs_buf, self.env_actions, self.clean_actions, self.reset_buf, self.keys = [], [], [], [], []
                if COLLECT_Z: self.zs = []
                self.curr_stpes = 0

            # Metrics are computed offline from exported files, not here
            update_str = f"Batch: {humanoid_env.start_idx // humanoid_env.num_envs} | steps {self.curr_stpes}"
            self.pbar.set_description(update_str)

        return done

    def get_z(self, obs_dict):
        obs = obs_dict['obs']
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': obs,
            'rnn_states': self.states
        }
        with torch.no_grad():
            z = self.model.a2c_network.eval_z(input_dict)
            return z

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_determenistic = self.is_determenistic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn
        for t in range(n_games):
            if games_played >= n_games:
                break
            obs_dict = self.env_reset()

            batch_size = 1
            batch_size = self.get_batch_size(obs_dict["obs"], batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

            print_game_res = False

            done_indices = []

            with torch.no_grad():
                for n in range(self.max_steps):
                    obs_dict = self.env_reset(done_indices)

                    if COLLECT_Z: z = self.get_z(obs_dict)

                    if has_masks:
                        masks = self.env.get_action_mask()
                        action = self.get_masked_action(obs_dict, masks, is_determenistic)
                    else:
                        action = self.get_action(obs_dict, is_determenistic)

                    obs_dict, r, done, info = self.env_step(self.env, action)

                    cr += r
                    steps += 1

                    if COLLECT_Z: info['z'] = z
                    done = self._post_step(info, done.clone())

                    if render:
                        self.env.render(mode="human")
                        time.sleep(self.render_sleep)

                    all_done_indices = done.nonzero(as_tuple=False)
                    done_indices = all_done_indices[:: self.num_agents]
                    done_count = len(done_indices)
                    games_played += done_count

                    if done_count > 0:
                        if self.is_rnn:
                            for s in self.states:
                                s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0

                        cur_rewards = cr[done_indices].sum().item()
                        cur_steps = steps[done_indices].sum().item()

                        cr = cr * (1.0 - done.float())
                        steps = steps * (1.0 - done.float())
                        sum_rewards += cur_rewards
                        sum_steps += cur_steps

                        game_res = 0.0
                        if isinstance(info, dict):
                            if "battle_won" in info:
                                print_game_res = True
                                game_res = info.get("battle_won", 0.5)
                            if "scores" in info:
                                print_game_res = True
                                game_res = info.get("scores", 0.5)
                        if self.print_stats:
                            if print_game_res:
                                print("reward:", cur_rewards / done_count, "steps:", cur_steps / done_count, "w:", game_res)
                            else:
                                print("reward:", cur_rewards / done_count, "steps:", cur_steps / done_count)

                        sum_game_res += game_res
                        if games_played >= n_games:
                            break

                    done_indices = done_indices[:, 0]

        print(sum_rewards)
        if print_game_res:
            print("av reward:", sum_rewards / games_played * n_game_life, "av steps:", sum_steps / games_played * n_game_life, "winrate:", sum_game_res / games_played * n_game_life)
        else:
            print("av reward:", sum_rewards / games_played * n_game_life, "av steps:", sum_steps / games_played * n_game_life)
