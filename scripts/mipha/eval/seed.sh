#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT=MiphaPhi2-v0-3b-finetune

#model_name=MiphaGemma-v0-2b-finetune
model_name=MiphaPhi2-v0-3b-finetune
SLM=phi_2
VIT=siglip
MODELDIR=./ckpts/checkpoints-$VIT/$SLM/$model_name

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} $SCRATCH/code/llava-phi/pytorch-example/python -m mipha.eval.model_vqa_loader \
        --model-path $MODELDIR \
        --question-file $VAST/eval/seed_bench/llava-seed-bench.jsonl \
        --image-folder $VAST/eval/seed_bench \
        --answers-file $VAST/eval/seed_bench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode phi &
done

wait

output_file=$VAST/eval/seed_bench/answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $VAST/eval/seed_bench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# Evaluate
$SCRATCH/code/llava-phi/pytorch-example/python scripts/convert_seed_for_submission.py \
    --annotation-file $VAST/eval/seed_bench/SEED-Bench.json \
    --result-file $output_file \
    --result-upload-file $VAST/eval/seed_bench/answers_upload/$model_name.jsonl

