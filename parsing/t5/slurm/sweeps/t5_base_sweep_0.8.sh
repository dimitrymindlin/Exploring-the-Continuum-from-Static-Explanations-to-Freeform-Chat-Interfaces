#!/bin/bash
#SBATCH --job-name=base_sweep
#SBATCH --output=./parsing/t5/base_sweep_0.8.txt
#SBATCH --time=21-00:00
#SBATCH --partition=ava_s.p
#SBATCH --nodelist=ava-s3
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --mem=15000MB

srun python parsing/t5/start_fine_tuning.py --gin parsing/t5/gin_configs/t5-base.gin --dataset diabetes --train_this_many 5 --down_sample_pct 0.8
