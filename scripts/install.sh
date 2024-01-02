#!/bin/bash
#$ -l rt_AG.small=1
#$ -l h_rt=1:00:00
#$ -j y
#$ -o outputs/install/
#$ -cwd

# module load
source /etc/profile.d/modules.sh
module load cuda/12.1/12.1.1
module load cudnn/8.9/8.9.5
module load nccl/2.18/2.18.3-1
module load hpcx/2.12

# python virtualenv
cd /bb/llm/gaf51275/llama/Megatron-LM
source .env/bin/activate

# pip install
pip install --upgrade pip
pip install -r requirements.txt

pip install ninja wheel packaging

# apex install
git clone git@github.com:NVIDIA/apex.git
cd apex

pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# flash-attention install
pip install flash-attn --no-build-isolation

# huggingface install
pip install transformers

pip install accelerate

# distributed checkpointing
pip install zarr

pip install tensorstore
