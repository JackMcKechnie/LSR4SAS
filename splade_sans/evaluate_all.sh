set -e
export WANDB_API_KEY="89144673f34dfdcbf35383f8c9040f5e7eb2fcb1"

# echo "Generating negatives files"
# python generate_sans_files.py --strategy "strat1" &
# python generate_sans_files.py --strategy "strat2" &
# python generate_sans_files.py --strategy "strat3" &
# wait
# echo "Negative files generated"

# Function to run the command and manage parallel jobs
run_command() {
  while [ "$(jobs -rp | wc -l)" -ge 2 ]; do
    sleep 1
  done
  "$@" &
}

# Variable to alternate argument
toggle=0

# First loop: Run group8 only
for model in splade_max unicoil tilde; do
  for strategy in 1 2 3; do
    run_command ./launch_evaluate.sh "${model}_strategy${strategy}_group8" "$toggle"
    # Toggle between 0 and 1
    if [ "$toggle" -eq 0 ]; then
      toggle=1
    else
      toggle=1
    fi
  done
done

# Second loop: Run group2 to group8
for model in splade_max unicoil tilde; do
  for strategy in 1 2 3; do
    for group in {2..8}; do
      run_command ./launch_evaluate.sh "${model}_strategy${strategy}_group${group}" "$toggle"
    # Toggle between 0 and 1
    if [ "$toggle" -eq 0 ]; then
      toggle=1
    else
      toggle=0
    fi
    done
  done
done

# Now evaluate baselines
./launch_evaluate.sh "splade_max_baseline" "0" &
./launch_evaluate.sh "tilde_baseline" "1"
./launch_evaluate.sh "unicoil_baseline" "0"

# Wait for all background jobs to complete
wait