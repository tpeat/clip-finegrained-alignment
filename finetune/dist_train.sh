#!/bin/bash
#SBATCH -Jvlmtrain
#SBATCH -N1                    # Number of nodes
#SBATCH --ntasks-per-node=2    # Number of tasks (processes) per node - set this to number of GPUs
#SBATCH --mem-per-gpu 8GB     
#SBATCH -G V100:2                # Number of GPUs - make sure this matches ntasks-per-node
#SBATCH -t 00:10:00
#SBATCH --cpus-per-task=4
#SBATCH -oReport-%j.out

module load python/3.10.10
module load anaconda3/2022.05.0.1
module load gcc/12.3.0
module load mvapich2/2.3.7-1
module load cuda/12.1.1

# # Environment variables for distributed training
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(hostname)
export WORLD_SIZE=$(($SLURM_NTASKS * $SLURM_GPUS_ON_NODE))
export LOCAL_WORLD_SIZE=$SLURM_GPUS_ON_NODE

EXP_NAME="dist_clip_spd_v1"
EPOCHS=100
LOSS_TYPE="clip"
OPTIM="adamspd"

nvidia-smi

# Get number of GPUs
if [ -n "$SLURM_GPUS_ON_NODE" ]; then
    NUM_GPUS=$SLURM_GPUS_ON_NODE
else
    NUM_GPUS=$(nvidia-smi -L | wc -l)
fi
echo "Number of GPUs allocated: $NUM_GPUS"

# Launch with torchrun
~/.conda/envs/nlp/bin/python -m torch.distributed.run \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --rdzv_id=clip_train \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:$MASTER_PORT \
    dist_finetuner.py \
    --exp_name $EXP_NAME \
    --epochs $EPOCHS \
    --loss_type $LOSS_TYPE \
    --optimizer $OPTIM