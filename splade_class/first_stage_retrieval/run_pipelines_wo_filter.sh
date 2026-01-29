set -e

for p_val in 1000; do
    echo "Running with p_val=$p_val"
    python run_pipelines_wo_filter.py --p "${p_val}" --k 100
done