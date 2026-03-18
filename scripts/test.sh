#!/bin/bash
# Usage: bash scripts/test.sh <motion_file>
# Example:
#   bash scripts/test.sh data/eval/AIST++_test.pkl   # AIST++ (4200 clips, 100 frames)
#   bash scripts/test.sh data/eval/EMDB2_test.pkl    # EMDB2  (295 clips, 100 frames)

# Add conda env lib to LD_LIBRARY_PATH if needed
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:$LD_LIBRARY_PATH"

MOTION_FILE=${1:?"Usage: bash scripts/test.sh <motion_file>"}
EXP_NAME="BC+RL+kp2d_mixed_rerun"
EPOCH=67500

python phc/run_hydra.py \
  learning=im_big \
  exp_name=$EXP_NAME \
  env=env_im robot=smpl_humanoid \
  env.motion_file=$MOTION_FILE \
  env.obs_v=358 \
  env.kin_loss=True \
  env.kin_weight=1001000 \
  env.kin_dict_size=69 \
  env.kin_lr=0.0005 \
  env.teacher_mode=pnn \
  epoch=$EPOCH \
  test=True env.num_envs=512 \
  im_eval=True
