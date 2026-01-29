set -e

./run_pipelines.sh
wait
for p in 1000 500 100; do

    echo "bm25 % ${p} >> filter % 100 >> get_text >> monot5_relevance"
    python rank_change_plot.py \
        --one "bm25 % ${p}" \
        --two "bm25 % ${p} >> filter % 100" \
        --three "bm25 % ${p} >> filter % 100 >> get_text >> monot5_relevance" &

    echo "bm25 % 100 >> get_text >> monot5_sans"
    python rank_change_plot.py \
        --one "bm25 % 100" \
        --two "bm25 % 100 >> get_text >> monot5_sans" &

    echo "bm25 % ${p} >> filter % 100 >> get_text >> monot5_sans"
    python rank_change_plot.py \
        --one "bm25 % ${p}" \
        --two "bm25 % ${p} >> filter % 100" \
        --three "bm25 % ${p} >> filter % 100 >> get_text >> monot5_sans" &

    echo "splade_relevance % ${p} >> filter % 100 >> get_text >> monot5_relevance"
    python rank_change_plot.py \
        --one "splade_relevance % ${p}" \
        --two "splade_relevance % ${p} >> filter % 100" \
        --three "splade_relevance % ${p} >> filter % 100 >> get_text >> monot5_relevance" &

    echo "splade_relevance % 100 >> get_text >> monot5_sans"
    python rank_change_plot.py \
        --one "splade_relevance % 100" \
        --two "splade_relevance % 100 >> get_text >> monot5_sans" &

    echo "splade_relevance % ${p} >> filter % 100 >> get_text >> monot5_sans"
    python rank_change_plot.py \
        --one "splade_relevance % ${p}" \
        --two "splade_relevance % ${p} >> filter % 100" \
        --three "splade_relevance % ${p} >> filter % 100 >> get_text >> monot5_sans" &

    echo "splade_sans % ${p} >> filter % 100 >> get_text >> monot5_relevance"
    python rank_change_plot.py \
        --one "splade_sans % ${p}" \
        --two "splade_sans % ${p} >> filter % 100" \
        --three "splade_sans % ${p} >> filter % 100 >> get_text >> monot5_relevance" &

    echo "splade_sans % 100 >> get_text >> monot5_sans"
    python rank_change_plot.py \
        --one "splade_sans % 100" \
        --two "splade_sans % 100 >> get_text >> monot5_sans" &

    echo "splade_sans % ${p} >> filter % 100 >> get_text >> monot5_sans"
    python rank_change_plot.py \
        --one "splade_sans % ${p}" \
        --two "splade_sans % ${p} >> filter % 100" \
        --three "splade_sans % ${p} >> filter % 100 >> get_text >> monot5_sans" &
done
wait