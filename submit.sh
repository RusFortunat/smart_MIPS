#!/bin/bash -l
#SBATCH --output=collect_trash.out
#SBATCH --get-user-env
#SBATCH --time=23:59:00
#SBATCH --partition=th-ws

echo $PWD
module load python/3
python3 smart_code.py 20 ${SLURM_ARRAY_TASK_ID}