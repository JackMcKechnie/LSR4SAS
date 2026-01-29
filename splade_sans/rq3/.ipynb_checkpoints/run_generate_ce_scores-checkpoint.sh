set -e

echo "Generating cross-encoder scores..."

CUDA_VISIBLE_DEVICES=0 python generate_ce_scores.py \
    --model_choice "strategy1" \
    --cutoff 100 \
    --output_path "./outputs/strategy1_scores.json" &

CUDA_VISIBLE_DEVICES=1 python generate_ce_scores.py \
    --model_choice "strategy2" \
    --cutoff 100 \
    --output_path "./outputs/strategy2_scores.json" &

CUDA_VISIBLE_DEVICES=2 python generate_ce_scores.py \
    --model_choice "strategy3" \
    --cutoff 100 \
    --output_path "./outputs/strategy3_scores.json" &

wait
echo "Cross-encoder scores generated"