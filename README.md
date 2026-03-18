# PhysHMR

This repository contains the official implementation of **PhysHMR**, a physics-based human motion reconstruction framework from monocular videos.

PhysHMR converts image features extracted by [GVHMR](https://github.com/zju3dv/GVHMR) into physically plausible human motion using reinforcement learning in a physics simulator ([Isaac Gym](https://developer.nvidia.com/isaac-gym)). It is built on top of the [PHC](https://github.com/ZhengyiLuo/PHC) framework.

---

## Setup

### 1. Create conda environment

```bash
conda create -n physhmr python=3.8 -y
conda activate physhmr
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. Install Isaac Gym

Download from https://developer.nvidia.com/isaac-gym, then:

```bash
cd isaacgym/python
pip install -e .
cd ../..
```

Fix NumPy compatibility:
```
File: isaacgym/python/isaacgym/torch_utils.py
Line 135: replace np.float with np.float32
```

Add to your shell config (required for Isaac Gym):
```bash
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:$LD_LIBRARY_PATH"
```

### 3. Download SMPL models

Download from [SMPL](https://smpl.is.tue.mpg.de/) (v1.1.0) and [SMPLX](https://smpl-x.is.tue.mpg.de/) (v1.1). Rename and place in `data/smpl/`:

```
data/smpl/
  SMPL_FEMALE.pkl
  SMPL_NEUTRAL.pkl
  SMPL_MALE.pkl
  SMPLX_FEMALE.pkl
  SMPLX_NEUTRAL.pkl
  SMPLX_MALE.pkl
```

### 4. Download pretrained models and data

```bash
bash download_data.sh
```

This downloads:
- PHC teacher checkpoints (`output/HumanoidIm/phc_3/`, `phc_comp_3/`)
- PhysHMR trained model (`output/HumanoidIm/BC+RL+kp2d_mixed_rerun/`)
- Test data (`data/eval/`)
- J_regressor (`data/J_regressor.npy`)

---

## Inference

Run PhysHMR on test data:

```bash
bash scripts/test.sh data/eval/AIST++_test.pkl
```

---

## Evaluation

### 1. Prepare AIST++ ground truth

Download **Motion Data Only** and **Camera Data Only** from the [AIST++ dataset page](https://google.github.io/aistplusplus_dataset/download.html). Extract and place them as:

```
data/AIST++/
  motions/          # .pkl files (e.g. gBR_sBM_cAll_d04_mBR0_ch01.pkl)
  cameras/
    mapping.txt
    setting1.json
    setting2.json
    ...
```

Then generate GT SMPL parameters:

```bash
python scripts/gen_gt_from_aist.py
```

This produces `data/gt/GT_AIST++/` with one `.npz` per clip, matching the clips in `data/eval/AIST++_test.pkl`.

### 2. Run evaluation

```bash
python scripts/visualize.py export/AIST++_test/ --eval --gt_path data/gt/GT_AIST++
```

---

## Visualization

Render result videos:

```bash
# Single sequence
python scripts/visualize.py export/AIST++_test/gLH_sBM_c01_d18_mLH2_ch06_00000.npz

# All sequences in a directory
python scripts/visualize.py export/AIST++_test/

# Save PLY meshes
python scripts/visualize.py export/AIST++_test/ --save_mesh --no_video
```

---

## References

This project is built on top of:

- [PHC](https://github.com/ZhengyiLuo/PHC) (BSD 3-Clause Clear License, see `THIRD_PARTY_LICENSES/`)
- [Isaac Gym](https://developer.nvidia.com/isaac-gym)
- [GVHMR](https://github.com/zju3dv/GVHMR)
- [TRAM](https://github.com/yufu-wang/tram)
