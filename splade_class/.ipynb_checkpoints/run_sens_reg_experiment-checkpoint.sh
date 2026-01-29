#!/bin/bash

# echo "Running pre-experiment runs..."
# ./launch_train_and_test_custom_reg.sh splade_multiple_negatives_ohsumed "0" &
# ./launch_train_and_test_custom_reg.sh splade_msmarco_multiple_negative "1"
# wait
# echo "Pre-experiment runs finished!"


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
        exp_name="weight_${weight_val}_k_${k_val}"
        
    cd ../splade_sans
    echo "$experiment training finished for weight=$weight_val and k=$k_val"

    CUDA_VISIBLE_DEVICES=$device python ohsumed_eval.py \
        --model_path "../splade_class/lsr_package/outputs/$experiment/model" \
        --index_loc "../splade_class/indices/sens_reg_exp/${experiment}_${weight_val}_${k_val}" \
        --run_output_path "../splade_class/runs/sens_reg_exp/${experiment}_${weight_val}_${k_val}" \
        --experiment_name "$experiment"
}

# Create an array of all experiment combinations
declare -a experiments
for weight in $(seq 500 -100 100); do
    for k in 100 1000; do
        experiments+=("$weight $k")
    done
done

total_experiments=${#experiments[@]}
echo "Total experiments to run: $total_experiments"

# Track running processes
declare -a pids
declare -a gpu_busy=(0 0 0)
declare -a gpu_experiment=("" "" "")

# Function to start an experiment on a specific GPU
start_experiment_on_gpu() {
    local exp_idx=$1
    local gpu_id=$2
    
    if [ $exp_idx -ge $total_experiments ]; then
        return 1  # No more experiments to run
    fi
    
    IFS=' ' read -r weight k <<< "${experiments[$exp_idx]}"
    run_experiment "$weight" "$k" "$gpu_id" &
    local pid=$!
    pids+=($pid)
    gpu_busy[$gpu_id]=1
    gpu_experiment[$gpu_id]="weight=$weight, k=$k, PID=$pid"
    echo "Started experiment on GPU $gpu_id: weight=$weight, k=$k, PID=$pid"
    return 0
}

# Function to check GPU status and start new experiments
check_and_start_experiments() {
    local next_exp=$1
    
    for gpu_id in 0 1 2; do
        if [ ${gpu_busy[$gpu_id]} -eq 0 ] && [ $next_exp -lt $total_experiments ]; then
            start_experiment_on_gpu $next_exp $gpu_id
            next_exp=$((next_exp + 1))
        fi
    done
    
    return $next_exp
}

# Initial start of experiments on all GPUs
next_exp=0
for gpu_id in 0 1 2; do
    if [ $next_exp -lt $total_experiments ]; then
        start_experiment_on_gpu $next_exp $gpu_id
        next_exp=$((next_exp + 1))
    fi
done

# Monitor running experiments and start new ones when GPUs become available
while [ $next_exp -lt $total_experiments ] || [ ${#pids[@]} -gt 0 ]; do
    # Check if any processes have finished
    for i in "${!pids[@]}"; do
        pid=${pids[$i]}
        if ! ps -p $pid > /dev/null; then
            echo "Process $pid has finished"
            # Find which GPU was running this process
            for gpu_id in 0 1 2; do
                if [[ ${gpu_experiment[$gpu_id]} == *"PID=$pid"* ]]; then
                    echo "GPU $gpu_id is now available"
                    gpu_busy[$gpu_id]=0
                    gpu_experiment[$gpu_id]=""
                    break
                fi
            done
            # Remove PID from array
            unset 'pids[$i]'
        fi
    done
    
    # Recreate pids array to remove empty slots
    pids=("${pids[@]}")
    
    # Start new experiments on available GPUs
    check_and_start_experiments $next_exp
    next_exp=$?
    
    # Print current status
    echo "--- Status Update ---"
    echo "Experiments completed/running/total: $((next_exp - ${#pids[@]}))/${#pids[@]}/$total_experiments"
    for gpu_id in 0 1 2; do
        if [ ${gpu_busy[$gpu_id]} -eq 1 ]; then
            echo "GPU $gpu_id: ${gpu_experiment[$gpu_id]}"
        else
            echo "GPU $gpu_id: idle"
        fi
    done
    echo "-------------------"
    
    # Sleep to avoid excessive CPU usage
    sleep 10
done

echo "All experiments completed!"