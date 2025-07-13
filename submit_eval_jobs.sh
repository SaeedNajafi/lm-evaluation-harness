#!/bin/bash

# # DPO
# epochs=(575 1150 1725 2300 2875)
# logps=(yes no)
# main_path="/home/saeednjf/links/projects/def-afyshe-ab/saeednjf/checkpoints/llama-3.2-1b-dpo-v10"


# for logps_i in ${!logps[@]};
#     do
#         logp=${logps[$logps_i]}
#         for epoch_i in ${!epochs[@]};
#         do
#             epoch=${epochs[$epoch_i]}
#             model_dir=${main_path}/llama3.2-1b-offline-dpo-beta_0.01-lr_0.0005-avg_logps_${logp}-v10/checkpoint-${epoch}_full_model
#             run_name=llama3.2-1b-offline-dpo-beta_0.01-lr_0.0005-avg_logps_${logp}-v10_checkpoint-${epoch}
#             sbatch run_eval_command.slrm MODEL_PATH=${model_dir} NAME=${run_name}
#             # sleep 2m
#     done
# done

# SimPO
epochs=(575 1150 1725 2300 2875)
logps=(yes no)
main_path="/home/saeednjf/links/projects/def-afyshe-ab/saeednjf/checkpoints/llama-3.2-1b-simpo-v10"

for logps_i in ${!logps[@]};
    do
        logp=${logps[$logps_i]}
        for epoch_i in ${!epochs[@]};
        do
            epoch=${epochs[$epoch_i]}
            model_dir=${main_path}/llama3.2-1b-offline-simpo-beta_0.01-lr_0.0005-gamma-to-beta_1.0-avg_logps_${logp}-v10/checkpoint-${epoch}_full_model
            run_name=llama3.2-1b-offline-simpo-beta_0.01-lr_0.0005-gamma-to-beta_1.0-avg_logps_${logp}-v10_checkpoint-${epoch}
            sbatch run_eval_command.slrm MODEL_PATH=${model_dir} NAME=${run_name}
    done
done


# MMPO with entropy
epochs=(575 1150 1725 2300 2875)
logps=(yes no)
main_path="/home/saeednjf/links/projects/def-afyshe-ab/saeednjf/checkpoints/llama-3.2-1b-mmpo-v10"


for logps_i in ${!logps[@]};
    do
        logp=${logps[$logps_i]}
        for epoch_i in ${!epochs[@]};
        do
            epoch=${epochs[$epoch_i]}
            model_dir=${main_path}/llama3.2-1b-offline-mmpo-beta_0.01-lr_0.0005-reward_eps_0.9-avg_logps_${logp}-v10-with-entropy/checkpoint-${epoch}_full_model
            run_name=llama3.2-1b-offline-mmpo-beta_0.01-lr_0.0005-reward_eps_0.9-avg_logps_${logp}-v10-with-entropy_checkpoint-${epoch}
            sbatch run_eval_command.slrm MODEL_PATH=${model_dir} NAME=${run_name}
    done
done

# MMPO no entropy
epochs=(575 1150 1725 2300 2875)
logps=(yes no)
main_path="/home/saeednjf/links/projects/def-afyshe-ab/saeednjf/checkpoints/llama-3.2-1b-mmpo-v10"


for logps_i in ${!logps[@]};
    do
        logp=${logps[$logps_i]}
        for epoch_i in ${!epochs[@]};
        do
            epoch=${epochs[$epoch_i]}
            model_dir=${main_path}/llama3.2-1b-offline-mmpo-beta_0.01-lr_0.0005-reward_eps_0.9-avg_logps_${logp}-v10-no-entropy/checkpoint-${epoch}_full_model
            run_name=llama3.2-1b-offline-mmpo-beta_0.01-lr_0.0005-reward_eps_0.9-avg_logps_${logp}-v10-no-entropy_checkpoint-${epoch}
            sbatch run_eval_command.slrm MODEL_PATH=${model_dir} NAME=${run_name}
    done
done