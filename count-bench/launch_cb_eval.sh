#!/bin/bash
#SBATCH -Jcbeval
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

MODEL='openai/clip-vit-base-patch32'
CONF=0.20
MARGIN=0.01
CKPT="../finetune/checkpoints/count_spd_v1/best.pt"
OUT_DIR="results/count_spd_v1"

echo "Beginning eval for $CKPT with $CONF confidence and $MARGIN"

 ~/.conda/envs/nlp/bin/python cb_eval.py --model $MODEL \
 --confidence $CONF \
 --margin $MARGIN \
 --checkpoint $CKPT \
 --output_dir $OUT_DIR \
 --debug \
 --samples 2 50 75 450