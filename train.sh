#export HF_ENDPOINT=https://hf-mirror.com

CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 18888 main_train.py \
    --pretrained_model= \
    --val_path= \
    --learning_rate=5e-5 \
    --gradient_accumulation_steps=1 \
    --enable_xformers_memory_efficient_attention --checkpointing_steps 12500 \
    --mixed_precision='fp16' \
    --report_to "tensorboard" \
    --seed 123 \
    --lora_rank=4 \
    --tracker_project_name ""