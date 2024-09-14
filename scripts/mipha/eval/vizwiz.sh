#!/bin/bash

#model_name=MiphaGemma-v0-2b-finetune
model_name=MiphaPhi2-v0-3b-finetune
SLM=phi_2
VIT=siglip
MODELDIR=./ckpts/checkpoints-$VIT/$SLM/$model_name

$SCRATCH/code/llava-phi/pytorch-example/python -m mipha.eval.model_vqa_loader \
    --model-path $MODELDIR \
    --question-file $VAST/eval/vizwiz/llava_test.jsonl \
    --image-folder $VAST/eval/vizwiz/test \
    --answers-file $VAST/eval/vizwiz/answers/$model_name.jsonl \
    --temperature 0 \
    --conv-mode phi

$SCRATCH/code/llava-phi/pytorch-example/python scripts/convert_vizwiz_for_submission.py \
    --annotation-file $VAST/eval/vizwiz/llava_test.jsonl \
    --result-file $VAST/eval/vizwiz/answers/$model_name.jsonl \
    --result-upload-file $VAST/eval/vizwiz/answers_upload/$model_name.json
