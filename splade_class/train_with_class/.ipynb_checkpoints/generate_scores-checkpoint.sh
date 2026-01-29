CUDA_VISIBLE_DEVICES="0" python score_corpus.py \
    --model "/mnt/primary/sas_cross_encoder/trained_models/ohsumed_1bm25_proportionate_neg_synth_data_monot5_sas_random_negs_best-31" \
    --name "one"

CUDA_VISIBLE_DEVICES="0" python score_corpus.py \
    --model "/mnt/primary/sas_cross_encoder/trained_models/ohsumed_synth_data_proportionate_negs_sas_easy_negs_best-21" \
    --name "two"

CUDA_VISIBLE_DEVICES="0" python score_corpus.py \
    --model "/mnt/primary/sas_cross_encoder/trained_models/ohsumed_randomneg_synth_data_monot5_sas_best-1" \
    --name "three"