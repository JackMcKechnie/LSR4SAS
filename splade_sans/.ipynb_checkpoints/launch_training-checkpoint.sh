set -e

cd learned-sparse-retrieval-1.0.0

echo "Starting $1 experiment"

CUDA_VISIBLE_DEVICES=$2 python -m lsr.train \
    +experiment=$1 \
    training_arguments.per_device_train_batch_size=8 \
    +training_arguments.gradient_accumulation_steps=16
    
cd ..
echo "$1 training finished!"

CUDA_VISIBLE_DEVICES=$2 python ohsumed_eval.py \
    --model_path "./learned-sparse-retrieval-1.0.0/outputs/$1/model" \
    --index_loc "./indices/$1" \
    --run_output_path "./runs/$1" \
    --experiment_name "$1"