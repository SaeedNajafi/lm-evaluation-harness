#!/bin/bash

# eval "$(conda shell.bash hook)"
# conda activate llm-env

# export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}
# export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
# export CUDA_HOME=/usr/local/cuda-12.4

run_name="llama3-8b-offline-mmpo-beta-0.1-lr-0.000001-reward_eps_0.0-relu-epsilon-0.5-full_standard_log_checkpoint-449"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
NUM_GPUs=8

date;pwd

export MASTER_ADDR="$(hostname --fqdn)"
export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')"
export RDVZ_ID=$RANDOM
echo "RDZV Endpoint $MASTER_ADDR:$MASTER_PORT"

# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_DEBUG=WARN
# export NCCL_DEBUG_SUBSYS=WARN
# export TORCH_CPP_LOG_LEVEL=INFO
# export LOGLEVEL=INFO

export WANDB_PROJECT="mmpo-project-lm-harness"
export WANDB_MODE="offline"
export ACCELERATE_LOG_LEVEL="info"

LOG_DIR="training_logs"
LOG_PATH="${LOG_DIR}/log_${run_name}.log"
# Make logging directories.
mkdir -p "${LOG_DIR}"
echo "Placing logs in: ${LOG_DIR}"

accelerate launch \
    --multi-gpu \
    --num_machines=1 \
    --num_processes=$NUM_GPUs \
    --machine_rank=0 \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    --rdzv_backend=static \
    lm_eval --model hf \
    --model_args pretrained=/work/saeed/narval/mmpo_8b_full_standard_log/llama3-8b-offline-mmpo-beta-0.1-lr-0.000001-reward_eps_0.0-relu-epsilon-0.5-full_standard_log/checkpoint-449,dtype=bfloat16 \
    --tasks winogrande,arc_easy,piqa,boolq,hellaswag,openbookqa,arc_challenge,mmlu,gsm8k,mathqa,race,commonsense_qa \
    --batch_size 22 \
    --num_fewshot 3 \
    --trust_remote_code \
    --gen_kwargs max_new_tokens=1024,do_sample=False \
    --output_path training_logs/lm_harness_output_${run_name} > ${LOG_PATH} 2>&1
