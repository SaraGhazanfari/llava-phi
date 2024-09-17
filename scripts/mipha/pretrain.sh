#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=25:00:00
#SBATCH --mem=512GB
#SBATCH --gres=gpu:4
#SBATCH --constraint='h100'
#SBATCH --job-name=mip_tok
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=sg7457@nyu.edu
## vision_encoder
#vision_encoder=openai/clip-vit-large-patch14-336
vision_encoder=siglip-so400m-patch14-384

## gemma
# model_dir=./ckpts/checkpoints-siglip/base_checkpoints/mipha_gemma
# outputdir=./ckpts/checkpoints-siglip/gemma_2b/MiphaGemma-v0-2b-pretrain

model_dir=./ckpts/checkpoints-siglip/phi_2/MiphaPhi2-v0-3b-pretrain
outputdir=./ckpts/checkpoints-siglip/phi_2/MiphaPhi2-v0-3b-finetune

#singularity exec --nv --overlay /scratch/sg7457/code/llava-phi/pytorch-example/my_pytorch.ext3:ro \
# /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c 'source /ext3/env.sh;
#deepspeed --master_port 29600 mipha/train/train.py \
#    --deepspeed ./scripts/zero2.json \
#    --model_name_or_path "./ckpts/checkpoints-siglip/base_checkpoints/mipha-phi-2" \
#    --version plain \
#    --data_path $VAST/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
#    --image_folder $VAST/LLaVA-Pretrain/images \
#    --tune_mm_mlp_adapter True \
#    --freeze_vision_tower True \
#    --freeze_backbone True \
#    --mm_use_im_start_end False \
#    --mm_use_im_patch_token False \
#    --bf16 True \
#    --output_dir ./ckpts/checkpoints-siglip/phi_2/MiphaPhi2-v0-3b-pretrain \
#    --num_train_epochs 1 \
#    --per_device_train_batch_size 32 \
#    --per_device_eval_batch_size 4 \
#    --gradient_accumulation_steps 1 \
#    --evaluation_strategy "no" \
#    --save_strategy "steps" \
#    --save_steps 24000 \
#    --save_total_limit 1 \
#    --learning_rate 1e-3 \
#    --weight_decay 0. \
#    --warmup_ratio 0.03 \
#    --lr_scheduler_type "cosine" \
#    --logging_steps 1 \
#    --tf32 True \
#    --model_max_length 8192 \
#    --gradient_checkpointing True \
#    --dataloader_num_workers 4 \
#    --lazy_preprocess True \
#    --report_to wandb \
#    --cache_dir ./'

cp $vision_encoder/preprocessor_config.json  $model_dir

singularity exec --nv --overlay /scratch/sg7457/code/llava-phi/pytorch-example/my_pytorch.ext3:ro \
 /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c 'source /ext3/env.sh;
deepspeed --master_port 29600 mipha/train/train.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./ckpts/checkpoints-siglip/phi_2/MiphaPhi2-v0-3b-pretrain \
    --version v0 \
    --data_path $SCRATCH/code/super_mllm/playground/data/llava_v1_5_mix665k.json \
    --image_folder  $VAST/LLaVA-Pretrain \
    --tune_mm_mlp_adapter True \
    --freeze_vision_tower True \
    --freeze_backbone False \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./ckpts/checkpoints-siglip/phi_2/MiphaPhi2-v0-3b-finetune \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb'

cp $vision_encoder/preprocessor_config.json  $outputdir



#  \ TOOOOOOOOOOOOOODOOOOOOOOOOOOOO
