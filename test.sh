#!/bin/bash

module load anaconda3/2021.11
conda activate aspire_gpu

python GPU_hackathon_power_method.py

./check_result.sh 10 J_sync_vec_n10.npy
