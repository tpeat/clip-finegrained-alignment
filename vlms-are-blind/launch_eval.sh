#!/bin/bash
#SBATCH -Jvlmsareblind
#SBATCH -N1 -n1
#SBATCH --mem-per-gpu 16GB
#SBATCH -G 1
#SBATCH -t 00:30:00
#SBATCH -oReport-%j.out

module load python/3.10.10
module load anaconda3/2022.05.0.1
module load gcc/12.3.0
module load mvapich2/2.3.7-1
module load cuda/12.1.1

MODEL='openai/clip-vit-large-patch14'
CONF=0.25
MARGIN=0.01
CKPT="../finetune/checkpoints/dummy_data/best.pt"

echo "Beginning eval for $MODEL with $CONF confidence and $MARGIN"

 ~/.conda/envs/nlp/bin/python eval.py --model $MODEL --confidence $CONF --margin $MARGIN --checkpoint $CKPT