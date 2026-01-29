for qrels in "no_pos_rs_neg_nrs_qrels.json" "pos_rns_neg_nrns_qrels.json" "pos_rns_neg_nrs_qrels"; do
    python get_scores_monot5strategy1.py --strategy ${qrels} &
done

wait