#!/bin/bash
#SBATCH --job-name=AL
#SBATCH --ntasks=1 --cpus-per-task=6 --mem=6000M
##SBATCH -p gpu --gres=gpu:titanx:1
#SBATCH --time=00-02:00:00
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err

#echo $CUDA_VISIBLE_DEVICES

python3 AL.py
