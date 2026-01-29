export WANDB_API_KEY="89144673f34dfdcbf35383f8c9040f5e7eb2fcb1"
export HYDRA_FULL_ERROR=1

echo "Starting experiment..."

cd ..

# Define the strategies and groups
strategies=("strategy1" "strategy2" "strategy3")
models=("splade_max" "unicoil")
groups=(2 3 4 5 6 7 8)

(
for group in "${groups[@]}"; do
    for strategy in "${strategies[@]}"; do
        ./launch_training.sh "splade_max_${strategy}_group${group}_frozen_qenc" "0"
    done
done
) &

( 
for group in "${groups[@]}"; do
    for strategy in "${strategies[@]}"; do
        ./launch_training.sh "unicoil_${strategy}_group${group}_frozen_qenc" "1"
    done
done
) &

wait
echo "Experiment finished!"


