set -e

for p_val in 1000 500 100; do
    echo "Running with p_val=$p_val"
    # Replace the line below with your actual command
    CUDA_VISIBLE_DEVICES="0" python run_pipelines.py --p "${p_val}"
    CUDA_VISIBLE_DEVICES="0" python run_intermediate_pipelines.py --p "${p_val}"
done