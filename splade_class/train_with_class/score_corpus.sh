# CUDA_VISIBLE_DEVICES="0" python score_corpus.py --start_qid "0" --end_qid "55689"
# CUDA_VISIBLE_DEVICES="1" python score_corpus.py --start_qid "55690" --end_qid "111378"
# CUDA_VISIBLE_DEVICES="0" python score_corpus.py --start_qid "111379" --end_qid "167067"
# CUDA_VISIBLE_DEVICES="1" python score_corpus.py --start_qid "167068" --end_qid "222756"
# CUDA_VISIBLE_DEVICES="0" python score_corpus.py --start_qid "222757" --end_qid "278445"
# CUDA_VISIBLE_DEVICES="1" python score_corpus.py --start_qid "278446" --end_qid "334134"

#!/bin/bash

pid=160
start_time=$(date +%s)

echo "Waiting for PID $pid to finish..."
while kill -0 $pid 2>/dev/null; do
    elapsed=$(( $(date +%s) - start_time ))
    hours=$((elapsed / 3600))
    minutes=$(( (elapsed % 3600) / 60 ))
    seconds=$((elapsed % 60))
    printf "\rElapsed time: %02d:%02d:%02d" $hours $minutes $seconds
    sleep 1
done

echo -e "\nPID $pid has finished. Running score_corpus.py..."
python score_corpus.py
