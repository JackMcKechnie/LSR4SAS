export WANDB_API_KEY="89144673f34dfdcbf35383f8c9040f5e7eb2fcb1"

# device="0"

# for model in splade unicoil tilde; do
#   for strategy in 1 2 3; do
#           echo "Running ${model}_strategy${strategy}_distil..."
#         ./launch_training.sh "${model}_strategy${strategy}_distil" "${device}"
#         echo "${model}_strategy${strategy}_distil finished!"
#   done
# done

# Function to run the command and manage parallel jobs
run_command() {
  # Wait until there are less than 3 jobs running
  while [ "$(jobs -rp | wc -l)" -ge 3 ]; do
    sleep 0.1  # Sleep for a short time to avoid unnecessary delay
  done
  "$@" &  # Run the command in the background
}

# Variable to alternate device argument
device=0

for model in splade unicoil tilde; do
  for strategy in 1 2 3; do
    echo "Running ${model}_strategy${strategy}_distil on device ${device}..."
    run_command ./launch_training.sh "${model}_strategy${strategy}_distil" "$device"
    # Toggle between devices (0, 1, 2)
    device=$(( (device + 1) % 3 ))
  done
done

# Wait for all background jobs to finish
wait

echo "All tasks finished!"