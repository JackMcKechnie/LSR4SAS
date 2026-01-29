models=(
  hn_t5_retrain_retrain
  hn_t5_base_retrain
  hn_t5_base_scratch
  hn_t5_retrain_scratch
  hn_t5_large_scratch
)

cd ../splade_sans
# for model in "${models[@]}"; do
#   echo "Running for model: $model"
#   latest_checkpoint=$(ls /nfs/primary/SPLADE/splade_class/lsr_package/outputs/$model | grep checkpoint- | sort -V | tail -n 1)
#   model_path="/nfs/primary/SPLADE/splade_class/lsr_package/outputs/$model/$latest_checkpoint/shared_encoder"
#   index_path="/nfs/primary/SPLADE/splade_class/indices/${model}_${latest_checkpoint}"
#   run_path="/nfs/primary/SPLADE/splade_class/runs/${model}_${latest_checkpoint}"
#   experiment_name="${model}_${latest_checkpoint}"
  
#   CUDA_VISIBLE_DEVICES="0" python ohsumed_eval.py \
#     --model_path $model_path \
#     --index_loc $index_path \
#     --run_output_path $run_path \
#     --experiment_name $experiment_name
# done


for model in "${models[@]}"; do
  echo "Running for model: $model"
  latest_checkpoint=$(ls /nfs/primary/SPLADE/splade_class/lsr_package/outputs/$model | grep checkpoint- | sort -V | tail -n 1)
  model_path="/nfs/primary/SPLADE/splade_class/lsr_package/outputs/$model/model"
  index_path="/nfs/primary/SPLADE/splade_class/indices/${model}"
  run_path="/nfs/primary/SPLADE/splade_class/runs/${model}"
  experiment_name="${model}"
  
  CUDA_VISIBLE_DEVICES="0" python ohsumed_eval.py \
    --model_path $model_path \
    --index_loc $index_path \
    --run_output_path $run_path \
    --experiment_name $experiment_name
done