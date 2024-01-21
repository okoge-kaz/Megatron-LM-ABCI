#!/bin/bash

cd /mnt/nfs/home/llmjp0/Megatron-LM
source .env/bin/activate

module load nccl
module load cuda/12.1
module load ucx/1.14.1
module load openmpi/4.1.5

export MASTER_ADDR=172.16.17.109
export MASTER_PORT=12345

NUM_NODES=2
NUM_GPU_PER_NODE=8
NUM_GPUS=$(($NUM_NODES * $NUM_GPU_PER_NODE))

HOSTFILE_NAME=/mnt/nfs/home/llmjp0/Megatron-LM/scripts/sakura/hostfile/2node
echo $(which python)

mpirun -np $NUM_GPUS \
    --npernode $NUM_GPU_PER_NODE \
    -hostfile $HOSTFILE_NAME \
    -x MASTER_ADDR=$MASTER_ADDR \
    -x MASTER_PORT=$MASTER_PORT \
    -x NCCL_IB_DISABLE=1 \
    -x PATH \
    -x NCCL_P2P_DISABLE=1 \
    -x LD_LIBRARY_PATH \
    python test_barrier.py
