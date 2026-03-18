#!/bin/bash
# Download pretrained models and test data for PhysHMR

# PhysHMR data
mkdir -p data/eval
mkdir -p output/HumanoidIm/BC+RL+kp2d_mixed_rerun

# Test data
gdown https://drive.google.com/uc?id=15HvIU0MuJWdjIyWtzW30Swwg6vqrC-Oz -O data/eval/AIST++_test.pkl

# PhysHMR trained model
gdown https://drive.google.com/uc?id=1N5WZ5i2GqJORsbYBKV2cuGkxByFYHv38 -O output/HumanoidIm/BC+RL+kp2d_mixed_rerun/Humanoid_00067500.pth

# PHC teacher checkpoints
mkdir -p output/HumanoidIm/phc_comp_3 output/HumanoidIm/phc_3
gdown https://drive.google.com/uc?id=1w76yqjYsfWgybzCizcvnGVa_FHlMwe8R -O output/HumanoidIm/phc_3/
gdown https://drive.google.com/uc?id=1p2Cvoy1GVHPgMgnTDZNtmdK8l1AkwaFC -O output/HumanoidIm/phc_comp_3/

# PHC original data (from PHC repo)
mkdir -p sample_data
gdown https://drive.google.com/uc?id=1bLp4SNIZROMB7Sxgt0Mh4-4BLOPGV9_U -O  sample_data/ # filtered shapes from AMASS
gdown https://drive.google.com/uc?id=1arpCsue3Knqttj75Nt9Mwo32TKC4TYDx -O  sample_data/ # all shapes from AMASS
gdown https://drive.google.com/uc?id=1fFauJE0W0nJfihUvjViq9OzmFfHo_rq0 -O  sample_data/ # sample standing neutral data.
gdown https://drive.google.com/uc?id=1uzFkT2s_zVdnAohPWHOLFcyRDq372Fmc -O  sample_data/ # amass_occlusion_v3
gdown https://drive.google.com/uc?id=1vUb7-j_UQRGMyqC_uY0YIdy6May297K5 -O  sample_data/ # PHC_X standing 
gdown https://drive.google.com/uc?id=11qLnYQR9FgOjXwCdWtYILrOevSfGbt8_ -O  sample_data/ # H1 sample dacne
gdown https://drive.google.com/uc?id=1tFEouQWhj9-5NtsfHfY0dUSdT0bjFJOW -O  sample_data/ # G1 sample dacne
