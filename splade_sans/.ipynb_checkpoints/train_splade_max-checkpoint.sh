set -e

cd learned-sparse-retrieval-1.0.0

echo "Starting SPLADE Max training with OHSUMED..."

python -m lsr.train \
    +experiment=splade_ohsumed_multiple_negatives \
    training_arguments.per_device_train_batch_size=8 \
    +training_arguments.gradient_accumulation_steps=4
    
cd ..
echo "SPLADE Max training finished!"

python ohsumed_eval.py \
    --model_path "./learned-sparse-retrieval-1.0.0/outputs/splade_ohsumed_multiple_negative_0.01_0.05/model" \
    --index_loc "./indices/splade_ohsumed_multiple_negative_0.01_0.05" \
    --run_output_path "./runs/splade_ohsumed_multiple_negative_0.01_0.05"