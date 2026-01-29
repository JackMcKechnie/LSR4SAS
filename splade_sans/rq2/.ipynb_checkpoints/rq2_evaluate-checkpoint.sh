run_evaluation() {
    local model_path="$1"
    local index_loc="$2"
    local run_output_path="$3"
    local experiment_name="$4"

    if [ -e "$3" ]; then
      echo "$3 exists"
      return 0
    fi

    python rq2_evaluate.py \
        --model_path "$model_path" \
        --index_loc "$index_loc" \
        --run_output_path "$run_output_path" \
        --experiment_name "$experiment_name"
}

model_outputs_path=../learned-sparse-retrieval-1.0.0/outputs

#############################################
## Evaluation with non-frozen query encoder##
#############################################

# for model in splade_max unicoil tilde; do
#     for strategy in 1 2 3; do
#         for group in {2..8}; do

#             # QEnc: SANS. DEnc: Relevance.
#             model_path="${model_outputs_path}/${model}_strategy${strategy}_group${group}/model"
#             index_loc="../indices/${model}_baseline"
#             run_output_path="./qenc_sans_denc_relevance/qenc_${model}_strategy${strategy}_group${group}_denc_${model}_baseline"
#             experiment_name="qenc_${model}_strategy${strategy}_group${group}_denc_${model}_baseline"
            
#             run_evaluation "$model_path" "$index_loc" "$run_output_path" "$experiment_name" &

#             # QEnc: Relevance. DEnc: SANS.
#             model_path="${model_outputs_path}/${model}_baseline/model"
#             index_loc="../indices/${model}_strategy${strategy}_group${group}"
#             run_output_path="./qenc_relevance_denc_sans/qenc_${model}_baseline_denc_${model}_strategy${strategy}_group${group}"
#             experiment_name="qenc_${model}_baseline_denc_${model}_strategy${strategy}_group${group}" 
            
#             run_evaluation "$model_path" "$index_loc" "$run_output_path" "$experiment_name" &
        
#         done
#     done
# done

###########################################
## Evaluation with a frozen query encoder##
###########################################

# Initialize an array to store matching directories
matching_dirs=()

# Loop through directories with "frozen" in their name
for dir in "$model_outputs_path"/*frozen*/; do
    # Check if the directory contains a "model" subdirectory
    if [ -d "$dir/model" ]; then
        matching_dirs+=("$dir")
    fi
done

cd ..

for model_path in "${matching_dirs[@]}"; do
    index_loc="./rq2/frozen_qenc_indices"
    run_output_path="./rq2/runs/frozen_qenc_runs"
    experiment_name=$(basename "$model_path")
    
    FILE="./runs_v2/$experiment_name"
    
    # ./launch_evaluate.sh "${experiment_name}" &
    if [ -f "$FILE" ]; then
        echo "The file '$FILE' exists."
    else
        echo "The file '$FILE' does not exist."
        ./launch_evaluate.sh "${experiment_name}" "0"
    fi
done

wait
echo "Done"