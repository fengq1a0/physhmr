# PhysHMR

🚧 **Code coming soon.**

This repository contains the official implementation of **PhysHMR**, a physics-based human motion reconstruction framework from monocular videos.

The codebase is currently under active reorganization and will be released incrementally.

---

## Code Status

The core implementation of PhysHMR is **built on top of the PHC framework**.  
During the original research development, the implementation evolved organically with extensive task-specific adaptations and experimental modifications.
As a result, the current codebase requires careful refactoring before being released in a clean and reproducible form.


---

## Roadmap

- [ ] Data preprocessing scripts  
- [ ] Pretrained models on Human3.6M  
- [ ] Refactored training code  


---

## Dependencies
Since this repository is built on top of PHC, the environment setup largely follows and adapts the instructions from PHC.

### 1. Create a new conda environment and install PyTorch

```
conda create -n physhmr python=3.8 -y
conda activate physhmr
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. Download and setup [Isaac Gym](https://developer.nvidia.com/isaac-gym)
Download Isaac Gym from
https://developer.nvidia.com/isaac-gym

Then install it as a Python package:
```
cd isaacgym/python
pip install -e .
cd ../..
```
A small manual modification is required for compatibility with recent NumPy versions:
```
File: physhmr/isaacgym/python/isaacgym/torch_utils.py
Line 135: replace `np.float` with `np.float32`
```
You also need to set `LD_LIBRARY_PATH` so that Isaac Gym can find the correct shared libraries.
Add the following line to your shell configuration or to all relevant bash scripts:
```
export LD_LIBRARY_PATH="/__path_to_your_conda__/envs/physhmr/lib:$LD_LIBRARY_PATH"
```



### 3. Download [SMPL](https://smpl.is.tue.mpg.de/) and [SMPLX](https://smpl-x.is.tue.mpg.de/) models
Download SMPL paramters from [SMPL](https://smpl.is.tue.mpg.de/) and [SMPLX](https://smpl-x.is.tue.mpg.de/). Put them in the `data/smpl` folder, unzip them into 'data/smpl' folder. For SMPL, please download the v1.1.0 version, which contains the neutral humanoid. Rename the files `basicmodel_neutral_lbs_10_207_0_v1.1.0`, `basicmodel_m_lbs_10_207_0_v1.1.0.pkl`, `basicmodel_f_lbs_10_207_0_v1.1.0.pkl` to `SMPL_NEUTRAL.pkl`, `SMPL_MALE.pkl` and `SMPL_FEMALE.pkl`. For SMPLX, please download the v1.1 version. 
After renaming, the directory structure should look like:

```
|-- data
    |-- smpl
        |-- SMPL_FEMALE.pkl
        |-- SMPL_NEUTRAL.pkl
        |-- SMPL_MALE.pkl
        |-- SMPLX_FEMALE.pkl
        |-- SMPLX_NEUTRAL.pkl
        |-- SMPLX_MALE.pkl
        |-- SMPLX_FEMALE.npz
        |-- SMPLX_NEUTRAL.npz
        |-- SMPLX_MALE.npz
```
> **Tip:** You may have already downloaded most of these files when setting up GVHMR or TRAM.


### 4. Download PHC data and checkpoints
Since PHC is used as the teacher model, its pretrained checkpoints are required:
```
bash download_data.sh
```


---

## Data Preprocessing

Our preprocessing pipeline relies on [**TRAM**](https://github.com/yufu-wang/tram) and [**GVHMR**](https://github.com/zju3dv/GVHMR).  
Please install and run them first. **Only single-person videos are supported.**

### 1. Run TRAM
Install TRAM and follow its **“Run demo on videos”** instructions.

From the TRAM output directory, we require:
- `camera.npy`
- `hps/hps_track_0.npy`

If multiple tracks are produced, we select the **longest** one.

### 2. Run GVHMR
Install GVHMR and follow its **Demo** instructions.

From the GVHMR output directory, we require:
- `hmr4d_results.pt`
- `preprocess/vitpose.pt`

### 3. Prepare PhysHMR inputs
We provide a script to align the TRAM camera with GVHMR outputs and convert them into the format required by PhysHMR.

```bash
python scripts/data_process/preprocess_physhmr.py \
  --gvhmr_root /path/to/gvhmr/output \
  --tram_root  /path/to/tram/output \
  --out_root   /path/to/physhmr/processed
```


---
## References
This project is based on and adapted from the following repository:

- [PHC](https://github.com/ZhengyiLuo/PHC)

PHC is under the BSD 3-Clause Clear License. See `THIRD_PARTY_LICENSES/` for details.
PHC in turn builds upon [IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs), [UHC](https://github.com/ZhengyiLuo/UniversalHumanoidControl), and [SMPL-X](https://github.com/vchoutas/smplx).
Please refer to the original repository and its license files for
additional third-party license information.

