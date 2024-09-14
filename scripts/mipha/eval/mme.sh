#!/bin/bash

#model_name=MiphaGemma-v0-2b-finetune
model_name=MiphaPhi2-v0-3b-finetune
SLM=phi_2
VIT=siglip
MODELDIR=./ckpts/checkpoints-$VIT/$SLM/$model_name

$SCRATCH/code/llava-phi/pytorch-example/python -m mipha.eval.model_vqa_loader \
    --model-path $MODELDIR \
    --question-file $VAST/eval/MME/llava_mme.jsonl \
    --image-folder $VAST/eval/MME/MME_Benchmark_release_version \
    --answers-file $VAST/eval/MME/answers/$model_name.jsonl \
    --temperature 0 \
    --conv-mode phi

cd $VAST/eval/MME

$SCRATCH/code/llava-phi/pytorch-example/python convert_answer_to_mme.py --experiment $model_name

cd eval_tool

$SCRATCH/code/llava-phi/pytorch-example/python calculation.py --results_dir answers/$model_name
