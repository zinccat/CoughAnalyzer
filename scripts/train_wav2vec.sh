#!/bin/bash
#
#SBATCH -p seas_gpu # partition (queue)
#SBATCH -c 12 # number of cores
#SBATCH --gpus 1
#SBATCH --mem 40G # memory pool for all cores
#SBATCH -t 0-16:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR

# module load Mambaforge/23.11.0-fasrc01 cmake/3.27.5-fasrc01 gcc/14.2.0-fasrc01 cuda/12.4.1-fasrc01 cudnn/9.5.1.17_cuda12-fasrc01
# micromamba activate ml

python src/models/wav2vec2.py