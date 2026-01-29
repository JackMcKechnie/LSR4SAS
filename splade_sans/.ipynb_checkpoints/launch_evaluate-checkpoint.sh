set -e
echo "Starting $1 evaluation"

# if [ -e "./indices/$1" ]; then
#   echo "./indices/$1 exists"
#   exit
# fi

# if [ -d "./indices/$1" ] && [ -n "$(ls -A "./indices/$1")" ]; then
#   echo "./indices/$1 exists and is not empty"
#   exit
# fi

# CUDA_VISIBLE_DEVICES=$2 python ohsumed_eval.py \
#     --model_path "./learned-sparse-retrieval-1.0.0/outputs/$1/model" \
#     --index_loc "./indices/$1" \
#     --run_output_path "./runs/$1" \
#     --experiment_name "$1"

cd /nfs/primary/SPLADE/splade_sans/

CUDA_VISIBLE_DEVICES=$2 python ohsumed_eval.py \
    --model_path "./learned-sparse-retrieval-1.0.0/outputs/$1/model" \
    --index_loc "../splade_class/train_with_class/indices/$1" \
    --run_output_path "../splade_class/train_with_class/runs/$1" \
    --experiment_name "$1"