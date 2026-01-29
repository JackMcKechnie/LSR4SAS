set -e

export WANDB_API_KEY="89144673f34dfdcbf35383f8c9040f5e7eb2fcb1"
export HYDRA_FULL_ERROR="1"

cd ./lsr_package

echo "Starting $1 experiment"

CUDA_VISIBLE_DEVICES=$2 python -m lsr.train \
    +experiment=$1 \
    training_arguments.per_device_train_batch_size=8 \
    +training_arguments.gradient_accumulation_steps=16 \
    training_arguments.logging_steps=50 \
    
cd ../../splade_sans
echo "$1 training finished!"

CUDA_VISIBLE_DEVICES=$2 python ohsumed_eval.py \
    --model_path "../splade_class/lsr_package/outputs/$1/model" \
    --index_loc "../splade_class/indices/$1" \
    --run_output_path "../splade_class/runs/$1" \
    --experiment_name "$1"