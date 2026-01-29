#!/bin/bash

run_experiment() {
    set -e

    local experiment="splade_multiple_negatives_ohsumed_sens_reg"
    local weight_val=$1
    local k_val=$2
    local device=$3

    export WANDB_API_KEY="89144673f34dfdcbf35383f8c9040f5e7eb2fcb1"
    export HYDRA_FULL_ERROR="1"

    cd lsr_package

    echo "Starting $experiment with weight=$weight_val and k=$k_val"

    CUDA_VISIBLE_DEVICES=$device python -m lsr.train \
        +experiment=$experiment \
        training_arguments.per_device_train_batch_size=8 \
        +training_arguments.gradient_accumulation_steps=16 \
        training_arguments.logging_steps=50 \
        loss.sens_reg.weight=$weight_val \
        loss.sens_reg.k=$k_val \
        exp_name="weight_${weight_val}_k_${k_val}" \
        training_arguments.num_train_epochs=10
        
    cd ../splade_sans
    echo "$experiment training finished for weight=$weight_val and k=$k_val"

    CUDA_VISIBLE_DEVICES=$device python ohsumed_eval.py \
        --model_path "../splade_class/lsr_package/outputs/$experiment/model" \
        --index_loc "../splade_class/indices/sens_reg_exp/${experiment}_${weight_val}_${k_val}" \
        --run_output_path "../splade_class/runs/sens_reg_exp/${experiment}_${weight_val}_${k_val}" \
        --experiment_name "$experiment"
}

# for weight in $(seq 0.2 0.2 1.2); do
#     for k in 100 1000; do
#         run_experiment $weight $k "3"
#     done
# done

weights=(400)
ks=(1000)

count=0
for weight in "${weights[@]}"; do
    for k in "${ks[@]}"; do
        run_experiment "$weight" "$k" "0"
    done
done

# Wait for any remaining jobs (in case of odd total number)
wait

echo "All experiments completed!"