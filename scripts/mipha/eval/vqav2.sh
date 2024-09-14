#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


SPLIT="llava_vqav2_mscoco_test-dev2015"

#model_name=MiphaGemma-v0-2b-finetune
MODEL_NAME=MiphaPhi2-v0-3b-finetune
SLM=phi_2
VIT=siglip
MODELDIR=./ckpts/checkpoints-$VIT/$SLM/MODEL_NAME

CKPT=$MODEL_NAME

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} $SCRATCH/code/llava-phi/pytorch-example/python -m mipha.eval.model_vqa_loader \
        --model-path $MODELDIR \
        --question-file $VAST/eval/vqav2/$SPLIT.jsonl \
        --image-folder /path/to/data/coco/test2015 \
        --answers-file $VAST/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode phi &
done

wait

output_file=$VAST/eval/vqav2/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $VAST/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

$SCRATCH/code/llava-phi/pytorch-example/python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT

