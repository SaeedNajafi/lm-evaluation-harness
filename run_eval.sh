
#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate llm-env

export CUDA_HOME=$CONDA_PREFIX
export NCCL_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib


run_name="Tinyllama_Flex_FUSED_DTRNet_k_0.35_1.0KD+0.0CE_part_one_lr_2e-5_checkpoint-10000"
export CUDA_VISIBLE_DEVICES="4,5,6,7"
NUM_GPUs=4

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

export WANDB_PROJECT="fused_dtrnet_project_march18_eval"
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
    --model_args pretrained=/work/saeed/dtrnet/Tinyllama_Flex_FUSED_DTRNet_k_0.35_1.0KD+0.0CE_part_one_lr_2e-5/checkpoint-10000,dtype=bfloat16,attn_implementation=eager \
    --tasks winogrande,arc_easy,piqa,boolq,hellaswag,openbookqa,arc_challenge,mmlu \
    --batch_size 1 \
    --num_fewshot 0 \
    --trust_remote_code \
    --gen_kwargs max_new_tokens=1024,do_sample=False \
    --output_path /work/saeed/dtrnet/Tinyllama_Flex_FUSED_DTRNet_k_0.35_1.0KD+0.0CE_part_one_lr_2e-5/checkpoint-10000/lm_harness_output > ${LOG_PATH} 2>&1

