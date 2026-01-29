export WANDB_API_KEY="89144673f34dfdcbf35383f8c9040f5e7eb2fcb1"
export HYDRA_FULL_ERROR="1"

(
  cd ../../splade_sans
  ./launch_training.sh splade_msmarco_classification_vector "0"
  ./launch_evaluate.sh splade_msmarco_classification_vector "0"
) &

# (
#   cd ../../splade_sans
#   # ./launch_training.sh splade_ohsumed_classification_vector "0"
#   ./launch_evaluate.sh splade_ohsumed_multiple_negative_0.01_0.05 "0"
# ) &
wait