cd ../../splade_sans/

export WANDB_API_KEY="89144673f34dfdcbf35383f8c9040f5e7eb2fcb1"
export HYDRA_FULL_ERROR=1

./launch_training.sh splade_ohsumed_add_logit "0"
