#!/bin/bash

# DPO
# epochs=(446 892 1338 1784 2225)
# logps=(yes no)
# main_path="/home/saeednjf/projects/def-afyshe-ab/saeednjf/checkpoints/llama-3.2-1b-dpo-v9"

# for logps_i in ${!logps[@]};
#     do
#         logp=${logps[$logps_i]}
#         for epoch_i in ${!epochs[@]};
#         do
#             epoch=${epochs[$epoch_i]}
#             model_dir=${main_path}/llama3.2-1b-offline-dpo-beta_0.01-lr_0.0005-avg_logps_${logp}-v9/checkpoint-${epoch}
#             run_name=llama3.2-1b-offline-dpo-beta_0.01-lr_0.0005-avg_logps_${logp}-v9_checkpoint-${epoch}
#             sbatch run_eval_command.slrm MODEL_PATH=${model_dir} NAME=${run_name}
#             sleep 2m
#     done
# done

# # SimPO
# # epochs=(446 892 1338 1784 2225)
# epochs=(446 1784)
# logps=(yes no)
# main_path="/home/saeednjf/projects/def-afyshe-ab/saeednjf/checkpoints/llama-3.2-1b-simpo-v9"

# for logps_i in ${!logps[@]};
#     do
#         logp=${logps[$logps_i]}
#         for epoch_i in ${!epochs[@]};
#         do
#             epoch=${epochs[$epoch_i]}
#             model_dir=${main_path}/llama3.2-1b-offline-simpo-beta_0.01-lr_0.0005-gamma-to-beta_1.0-avg_logps_${logp}-v9/checkpoint-${epoch}
#             run_name=llama3.2-1b-offline-simpo-beta_0.01-lr_0.0005-gamma-to-beta_1.0-avg_logps_${logp}-v9_checkpoint-${epoch}
#             sbatch run_eval_command.slrm MODEL_PATH=${model_dir} NAME=${run_name}
#             sleep 3m
#     done
# done


# # MMPO with entropy
# epochs=(446 892 1338 1784 2225)
# logps=(yes no)
# main_path="/home/saeednjf/projects/def-afyshe-ab/saeednjf/checkpoints/llama-3.2-1b-mmpo-v9"

# for logps_i in ${!logps[@]};
#     do
#         logp=${logps[$logps_i]}
#         for epoch_i in ${!epochs[@]};
#         do
#             epoch=${epochs[$epoch_i]}
#             model_dir=${main_path}/llama3.2-1b-offline-mmpo-beta_0.01-lr_0.0005-reward_eps_0.9-avg_logps_${logp}-v9-with-entropy/checkpoint-${epoch}
#             run_name=llama3.2-1b-offline-mmpo-beta_0.01-lr_0.0005-reward_eps_0.9-avg_logps_${logp}-v9-with-entropy_checkpoint-${epoch}
#             sbatch run_eval_command.slrm MODEL_PATH=${model_dir} NAME=${run_name}
#             sleep 3m
#     done
# done

# MMPO
epochs=(575 1150 1725 2300 2870)
logps=(yes no)
sizes=(360M 135M)
main_path="/home/saeednjf/projects/def-afyshe-ab/saeednjf/checkpoints/smollm2/v2-runs"

for size_i in ${!sizes[@]};
do
    size=${sizes[$size_i]}
    for logps_i in ${!logps[@]};
    do
        logp=${logps[$logps_i]}
        for epoch_i in ${!epochs[@]};
        do
            epoch=${epochs[$epoch_i]}
            model_dir=${main_path}/smollm2-${size}-orca_bin_ultra-offline-mmpo-beta_0.01-lr_0.0005-reward_eps_0.9-avg_logps_${logp}-v2-with-entropy/checkpoint-${epoch}
            run_name=smollm2-${size}-orca_bin_ultra-offline-mmpo-beta_0.01-lr_0.0005-reward_eps_0.9-avg_logps_${logp}-v2-with-entropy_checkpoint-${epoch}
            sbatch run_eval_command.slrm MODEL_PATH=${model_dir} NAME=${run_name}
            sleep 3m
        done
    done
done