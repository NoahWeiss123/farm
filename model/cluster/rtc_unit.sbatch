#!/bin/bash
# One-shot GPU unit test of the RTC guided sampler (writes to a file so we don't
# have to hold a kubectl exec stream open for the whole run).
#SBATCH --job-name=rtc-unit
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00
#SBATCH --output=/home/nhweiss/farm-train/rtc-unit-%j.out

cd /home/nhweiss/farm-train/openpi
srun --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
     --container-mounts=/home/nhweiss:/home/nhweiss \
     --container-workdir=/home/nhweiss/farm-train/openpi \
     bash -lc 'set -e; export HOME=/home/nhweiss; export JAX_PLATFORMS=cuda; export TF_CPP_MIN_LOG_LEVEL=3; pip install -q uv 2>/dev/null; cd /home/nhweiss/farm-train/openpi; uv run python /home/nhweiss/farm-train/unit_test.py'
