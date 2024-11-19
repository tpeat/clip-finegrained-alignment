#!/bin/bash
#SBATCH -Jdata-gen
#SBATCH -N1 -n1
#SBATCH --mem-per-gpu 16GB
#SBATCH -G 1
#SBATCH -t 04:00:00
#SBATCH -oReport-%j.out

module load python/3.10.10
module load anaconda3/2022.05.0.1
module load gcc/12.3.0
module load mvapich2/2.3.7-1
module load cuda/12.1.1

NUM_SAMPLES=50000
SIZE="small"
ANN_MODE="count"

echo "Beginning generating $NUM_SAMPLES of $SIZE synethic data with $ANN_MODE annotations"

~/.conda/envs/nlp/bin/python gen_synthetic_data.py --num_samples $NUM_SAMPLES --size_category $SIZE --annotation_mode $ANN_MODE