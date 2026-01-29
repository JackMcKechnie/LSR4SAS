files=($(ls | grep -i "hn_splade_sans" | grep -E 'base|large|retrain'))

# Loop over array
for file in "${files[@]}"; do
    echo "Processing ${file}"
    CUDA_VISIBLE_DEVICES=0 python evaluate_monot5.py --model_path "${file}" --output_run_path "./runs"
done