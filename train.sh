#!/bin/bash
# Train ADM (unconditional) on real COCO images
# Multi-GPU via MPI + NCCL DDP

NUM_GPUS=3  # adjust to your setup

MODEL_FLAGS="--class_cond False --num_colors 3 --image_size 256 --num_channels 128 --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"

DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine --rescale_learned_sigmas False --rescale_timesteps False --use_scale_shift_norm False"

TRAIN_FLAGS="--use_topo 0 --lr 2e-5 --batch_size 4 --save_interval 5000 --resume_checkpoint models/lsun_uncond_100M_2400K_bs64.pt"

export OPENAI_LOGDIR="logs/adm_coco_real"

CUDA_VISIBLE_DEVICES="0,1,2" mpiexec -n $NUM_GPUS python image_train.py --data_dir datasets/coco_real $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS