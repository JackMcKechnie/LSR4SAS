export WANDB_API_KEY="89144673f34dfdcbf35383f8c9040f5e7eb2fcb1"
export HYDRA_FULL_ERROR=1

./launch_training.sh splade_ohsumed_classification_vector "0"

# CUDA_VISIBLE_DEVICES=$2 python combined_model_evaluate.py \
#     --model_path "./learned-sparse-retrieval-1.0.0/outputs/$1/model" \
#     --index_loc "./indices_v2/$1" \
#     --run_output_path "./runs_v2/$1" \
#     --experiment_name "$1"