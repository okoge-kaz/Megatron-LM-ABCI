#!/bin/bash

set -e

start=5000
end=25000
increment=5000

upload_base_dir=/bb/llm/gaf51275/llama/from_megatron_hf_checkpoints/hf_checkpoints/Llama2-7b-base-parallel/shuffle-normal-quality-next-token

for ((i = start; i <= end; i += increment)); do
  upload_dir=$upload_base_dir/iter_$(printf "%07d" $i)

  python scripts/abci/upload/upload.py \
    --ckpt-path $upload_dir \
    --repo-name tokyotech-llm/Llama2-7b-base-parallel-corpus-next-token-shuffle-normal-quality-megatron-iter$(printf "%07d" $i)
done
