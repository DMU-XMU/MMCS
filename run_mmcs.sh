#!/usr/bin/env bash
envname=("walker_walk")
types=("expert")
seeds=(0) 

for seed in "${seeds[@]}"
do
  for env in "${envname[@]}"
  do
    for typ in "${types[@]}"
    do
        echo "Running task: offline_${env}_${typ}, seed=${seed}"
        
        # Assuming you have already activated the conda environment
        python train.py task_name="offline_${env}_${typ}" offline_dir="your_data_path" algo=mmcs dist_level=none_vd4rl seed=${seed} device=cuda
    done
  done
done

