#!/bin/bash

set -e

start=5000
end=15000
increment=5000

upload_base_dir=/mnt/nfs/Users/kazuki/checkpoints/llama/hf_checkpoints/Llama-2-70b

for ((i = start; i <= end; i += increment)); do
  upload_dir=$upload_base_dir/iter_$(printf "%07d" $i)

  python scripts/abci/upload/upload.py \
    --ckpt-path $upload_dir \
    --repo-name tokyotech-llm/Llama2-70b-base-cc-megatron-iter$(printf "%07d" $i)
done
