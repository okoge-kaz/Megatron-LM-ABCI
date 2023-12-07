#!/bin/bash

set -e

start=2000
end=4000
increment=2000

upload_base_dir=/groups/gaf51275/llama/checkpoints/hf_checkpoints/pubmed/shuffle/llama-2-13b-base-extended-cc

for ((i = start; i <= end; i += increment)); do
  upload_dir=$upload_base_dir/iter_$(printf "%07d" $i)

  python scripts/abci/upload/upload.py \
    --ckpt-path $upload_dir \
    --repo-name kotoba-tech/PUBMED-shuffle-Llama2-13b-base-extended-cc-megatron-iter$(printf "%07d" $i)
done
