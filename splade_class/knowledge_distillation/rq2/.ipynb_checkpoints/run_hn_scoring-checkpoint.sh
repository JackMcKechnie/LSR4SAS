CUDA_VISIBLE_DEVICES="0" python get_score_files.py \
    --INPUT_MODEL "./hn_splade_sans1_retrain-2" \
    --NAME "hn_retrained_8negs" \
    --OUTPUT_PATH "./score_files/" &

CUDA_VISIBLE_DEVICES="1" python get_score_files.py \
    --INPUT_MODEL "./hn_splade_sans1_t5_base-30" \
    --NAME "hn_t5base_8negs" \
    --OUTPUT_PATH "./score_files/" &

wait

CUDA_VISIBLE_DEVICES="0" python get_score_files.py \
    --INPUT_MODEL "./hn_splade_sans1_t5_large-2" \
    --NAME "hn_t5large_8negs" \
    --OUTPUT_PATH "./score_files/"

wait
echo "Finished"