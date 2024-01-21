#!/bin/bash

cd /mnt/nfs/home/llmjp0/Megatron-LM
source .env/bin/activate

module load nccl
module load cuda/12.1
module load ucx/1.14.1
module load openmpi/4.1.5

export MASTER_ADDR=172.16.17.109
export MASTER_PORT=12345

NUM_NODES=48
NUM_GPU_PER_NODE=8
NUM_GPUS=$(($NUM_NODES * $NUM_GPU_PER_NODE))

# model config
# https://arxiv.org/pdf/2005.14165.pdf
HIDDEN_SIZE=12288
FFN_HIDDEN_SIZE=49152
NUM_LAYERS=96
NUM_HEADS=96
SEQ_LENGTH=$1

# distributed settings
TENSOR_PARALLEL_SIZE=$2
PIPELINE_PARALLEL_SIZE=$3
DATA_PARALLEL_SIZE=$((${NUM_GPUS} / (${TENSOR_PARALLEL_SIZE} * ${PIPELINE_PARALLEL_SIZE})))

# training config
MICRO_BATCH_SIZE=$4
GLOBAL_BATCH_SIZE=1120
TRAIN_STEPS=25000 # e.g. llama: 1T tokens / 4M tokens_per_batch = 250000 steps
# 今回は約100B Tokensなので 1/10

LR=1e-4
MIN_LR=3.3e-6
LR_WARMUP_STEPS=1000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# model config
TOKENIZER_MODEL=/mnt/nfs2/data0/work/llm-jp-tokenizer/models/ver2.2/code20K_en40K_ja60K.ver2.2.model
CHECKPOINT_SAVE_DIR=/mnt/nfs2/data0/work/checkpoints/gpt-175b-base-megatron/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config
DATASET_DIR=/mnt/nfs2/data0/work/binarized/ver2.2/code20K_en40K_ja60K/ja_wiki

DATA_PATH="${DATASET_DIR}/ja_wiki_text_document"

# job name
JOB_NAME="gpt-175b-base-selective-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-${SEQ_LENGTH}s-DP=${DATA_PARALLEL_SIZE}-TP=${TENSOR_PARALLEL_SIZE}-PP=${PIPELINE_PARALLEL_SIZE}-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}"

# --norm-epsilon 1e-5 : conifg.json (RMS norm)

# checkpoint load
if [[ -f "${CHECKPOINT_SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
  # resume training
  CHECKPOINT_ARGS="--load ${CHECKPOINT_SAVE_DIR}"
else
  # first training
  CHECKPOINT_ARGS="--load ${CHECKPOINT_SAVE_DIR}"
fi

# export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_DEBUG=INFO

HOSTFILE_NAME=/mnt/nfs/home/llmjp0/Megatron-LM/scripts/sakura/hostfile/48node

mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x tmp=/mnt/nfs/home/llmjp0/Megatron-LM/tmp \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -x NCCL_IB_GID_INDEX=3 \
  -x NCCL_NET_GDR_LEVEL=1 \
  -x NCCL_DEBUG=INFO \
  -x LD_LIBRARY_PATH \
  -bind-to none -map-by slot \
  -x PATH \
  python pretrain_gpt.py \
  --tensor-model-parallel-size ${TENSOR_PARALLEL_SIZE} \
  --pipeline-model-parallel-size ${PIPELINE_PARALLEL_SIZE} \
  --sequence-parallel \
  --use-distributed-optimizer \
  --num-layers ${NUM_LAYERS} \
  --hidden-size ${HIDDEN_SIZE} \
  --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
  --num-attention-heads ${NUM_HEADS} \
  --seq-length ${SEQ_LENGTH} \
  --max-position-embeddings ${SEQ_LENGTH} \
  --micro-batch-size ${MICRO_BATCH_SIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --train-iters ${TRAIN_STEPS} \
  --tokenizer-type SentencePieceTokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --use-checkpoint-args \
  ${CHECKPOINT_ARGS} \
  --save ${CHECKPOINT_SAVE_DIR} \
  --data-path ${DATA_PATH} \
  --split 949,50,1 \
  --distributed-backend nccl \
  --init-method-std 0.006 \
  --lr ${LR} \
  --min-lr ${MIN_LR} \
  --lr-decay-style cosine \
  --weight-decay ${WEIGHT_DECAY} \
  --clip-grad ${GRAD_CLIP} \
  --lr-warmup-iters ${LR_WARMUP_STEPS} \
  --optimizer adam \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --log-interval 1 \
  --save-interval 500 \
  --eval-interval 100 \
  --eval-iters 10 \
  --bf16 \
  --untie-embeddings-and-output-weights \
  --use-rotary-position-embeddings \
  --no-position-embedding \
  --no-masked-softmax-fusion \
  --recompute-activations \
  --recompute-granularity "selective" \
  --use-flash-attn \
  --use-mpi \
  --wandb-name ${JOB_NAME} \
  --wandb-project "sakura-h100-flops-175b-56node" \
  --wandb-entity "llm-jp"

