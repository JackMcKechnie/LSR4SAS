import pandas as pd
import pyterrier as pt
if not pt.started():
    pt.init()
import pyterrier_pisa
from pyterrier_t5 import *
import os
import argparse
from ir_measures import *

parser = argparse.ArgumentParser(description="Script with p and k arguments")
parser.add_argument('--p', type=int, required=True, help='Value for p')
parser.add_argument('--k', type=int, default = 100, help='Value for k')

args = parser.parse_args()

p = args.p
k = args.k

ohsumed_docs = pd.read_pickle("/nfs/primary/sas_reranker/ohsumed_docs_w_t5base_sensitivity.pkl")
queries = pd.read_pickle("/nfs/primary/graph_adaptive_reranking/queries.pkl")
qrels = pd.read_pickle("/nfs/primary/graph_adaptive_reranking/qrels.pkl")

def calculate_csdcg_10(qrels, run):
    if "sensitivity" in run.columns:
        run = run.drop(columns = ["sensitivity"])
    merged = pd.merge(run, qrels, on = "doc_id", how = "left").fillna(0).head(10)
    merged = pd.merge(merged, ohsumed_docs, left_on = "doc_id", right_on = "docno")
    # print(merged[["relevance", "sensitivity"]])
    csdcg = 0
    for rank, doc in enumerate(merged.iterrows()):
        relevance = int(doc[1].relevance)
        sensitivity = int(doc[1]["sensitivity"])
        up = (2 ** relevance) - 1
        low = math.log(rank + 2)
        penalty = 12 if sensitivity == 1 else 0
        csdcg += ((up / low) - penalty)
    return csdcg

def _ncsdcg_cut_10(qrels, run):    
    # Calculate query csdcg
    query_csdcg = calculate_csdcg_10(qrels, run)
    
    # Calculate best csdcg
    merged_qrels = pd.merge(qrels, ohsumed_docs, left_on = "doc_id", right_on = "docno").drop(columns = ["title", "abstract", "text", "t5_sensitivity", "docno"])
    best_qrels =  merged_qrels.sort_values(by=['sensitivity', 'relevance'], ascending=[True, False])
    best_qrels = best_qrels.drop(columns = ["relevance", "sensitivity"])
    best_csdcg = calculate_csdcg_10(qrels, best_qrels)
    
    # Calculate worst csdcg
    worst_qrels =  merged_qrels.sort_values(by=['sensitivity', 'relevance'], ascending=[False, True])
    worst_qrels = worst_qrels.drop(columns = ["relevance", "sensitivity"])
    worst_csdcg = calculate_csdcg_10(qrels, worst_qrels)
    
    if query_csdcg <= worst_csdcg:
        query_csdcg = 0
        
    normalised_csdcg = (query_csdcg - worst_csdcg) / (best_csdcg - worst_csdcg)

    assert normalised_csdcg >= 0
    assert normalised_csdcg <= 1
    
    return normalised_csdcg
    
import ir_measures
nCSDCG_cut_10 = ir_measures.define_byquery(
    _ncsdcg_cut_10, 
    name="nCSDCG_cut_10")


def calculate_csdcg_20(qrels, run):
    if "sensitivity" in run.columns:
        run = run.drop(columns = ["sensitivity"])
    merged = pd.merge(run, qrels, on = "doc_id", how = "left").fillna(0).head(20)
    merged = pd.merge(merged, ohsumed_docs, left_on = "doc_id", right_on = "docno")
    # print(merged.columns)
    # print(merged[["relevance","sensitivity"]])
    csdcg = 0
    for rank, doc in enumerate(merged.iterrows()):
        relevance = int(doc[1].relevance)
        sensitivity = int(doc[1]["sensitivity"])
        up = (2 ** relevance) - 1
        low = math.log(rank + 2)
        penalty = 21 if sensitivity == 1 else 0
        csdcg += ((up / low) - penalty)
    return csdcg

def _ncsdcg_cut_20(qrels, run):    
    # Calculate query csdcg
    query_csdcg = calculate_csdcg_20(qrels, run)
    
    # Calculate best csdcg
    merged_qrels = pd.merge(qrels, ohsumed_docs, left_on = "doc_id", right_on = "docno").drop(columns = ["title", "abstract", "text", "t5_sensitivity", "docno"])
    best_qrels =  merged_qrels.sort_values(by=['sensitivity', 'relevance'], ascending=[True, False])
    best_qrels = best_qrels.drop(columns = ["relevance", "sensitivity"])
    best_csdcg = calculate_csdcg_20(qrels, best_qrels)
    
    # Calculate worst csdcg
    worst_qrels =  merged_qrels.sort_values(by=['sensitivity', 'relevance'], ascending=[False, True])
    worst_qrels = worst_qrels.drop(columns = ["relevance", "sensitivity"])
    worst_csdcg = calculate_csdcg_20(qrels, worst_qrels)

    normalised_csdcg = (query_csdcg - worst_csdcg) / (best_csdcg - worst_csdcg)
    return normalised_csdcg
    
import ir_measures
nCSDCG_cut_20 = ir_measures.define_byquery(
    _ncsdcg_cut_20, 
    name="nCSDCG_cut_20")

def _sens_docs(qrels, run):
    run.doc_id = run.doc_id.astype(str)
    original_run_length = len(run)
    if "sensitivity" in run.columns:
        run = run.drop(columns = ["sensitivity"])
    merged = pd.merge(run, ohsumed_docs, left_on = "doc_id", right_on = "docno")
    assert len(merged) == original_run_length
    return merged.sensitivity.sum()
    
import ir_measures
sens_docs = ir_measures.define_byquery(
    _sens_docs, 
    name="sens_docs")

def _ns_recall(qrels, run):
    combined = pd.merge(run, qrels, on = ["doc_id", "query_id"], how = "left").fillna(0)
    combined = pd.merge(combined, ohsumed_docs, left_on = "doc_id", right_on = "docno")
    assert len(combined) == len(run)
    num_rel_ns = combined[(combined['sensitivity'] < 1) & (combined['relevance'] > 1)].shape[0]
    qrels_w_sensitivity = pd.merge(qrels, ohsumed_docs, left_on = "doc_id", right_on = "docno")
    total = qrels_w_sensitivity[(qrels_w_sensitivity['sensitivity'] < 1) & (qrels_w_sensitivity['relevance'] > 1)].shape[0]
    assert num_rel_ns <= total
    if total == 0:
        return 0
    return num_rel_ns / total
    
import ir_measures
NSR = ir_measures.define_byquery(
    _ns_recall, 
    name="NSR")

def _ns_recall_judged(qrels, run):
    # Get judged documents that are non-sensitive and relevant
    qrels_with_sensitivity = pd.merge(
        qrels, 
        ohsumed_docs, 
        left_on="doc_id", 
        right_on="docno"
    )
    target_docs = qrels_with_sensitivity[
        (qrels_with_sensitivity['sensitivity'] < 1) & 
        (qrels_with_sensitivity['relevance'] > 1)
    ]
    
    # Count how many target documents were retrieved
    retrieved_target_docs = run[run['doc_id'].isin(target_docs['doc_id'])]
    num_retrieved = len(retrieved_target_docs)
    
    total = len(target_docs)
    
    if total == 0:
        return 0
    
    return num_retrieved / total

import ir_measures
NSRJudged = ir_measures.define_byquery(_ns_recall_judged, name="NSRJudged")


def t5_filter_out(run):
    merged = pd.merge(run, ohsumed_docs, on=["docno"])
    merged = merged[merged["t5_sensitivity"] == 0]
    merged = merged.drop(columns = ["rank", "sensitivity"])
    merged = add_ranks(merged)
    return merged

def kill(run):
    print("Killing run, not implemented yet")
    run["score"] = [0] * len(run)
    return run

def get_text(run):
    if "text" in run.columns:
        return run
    original_run_length = len(run)
    run = pd.merge(run, ohsumed_docs, on = "docno")
    assert len(run) == original_run_length
    return run


filter = pt.apply.generic(t5_filter_out)
kill = pt.apply.generic(kill)
get_text = pt.apply.generic(get_text)

relevance_monot5_path = "/nfs/primary/sas_cross_encoder/trained_models/ohsumed_1bm25_easyneg_synth_data_monot5_best-44"
monot5_sans_path = "/nfs/primary/sas_cross_encoder/trained_models/ohsumed_randomneg_synth_data_monot5_sas_best-1"


bm25 = pt.Transformer.from_df(pt.io.read_results("/nfs/primary/standard_run_files/ohsumed/bm25_pisa_1000.run"))
monot5_relevance = MonoT5ReRanker(model = relevance_monot5_path, batch_size = 128, verbose = True)
monot5_sans = MonoT5ReRanker(model = monot5_sans_path, batch_size = 128, verbose = True)
splade_relevance = pt.Transformer.from_df(pt.io.read_results("/nfs/primary/SPLADE/splade_class/runs/splade_ohsumed_multiple_negative"))
splade_sans = pt.Transformer.from_df(pt.io.read_results("/nfs/primary/SPLADE/splade_sans/runs/splade_max_strategy1_group8"))
splade_distil = pt.Transformer.from_df(pt.io.read_results("/nfs/primary/SPLADE/splade_class/runs/hn_t5_retrain_scratch"))


pipelines = [
    # No reranking
    bm25 % k,
    splade_relevance % k,
    splade_sans % k,
    splade_distil % k,
        
    # Filter
    bm25 % k >> filter,
    splade_relevance % k >> filter,
    splade_sans % k >> filter,
    splade_distil % k >> filter,
    
    # Filter >> Rerank
    bm25 % k >> filter >> get_text >> monot5_relevance,
    splade_relevance % k >> filter >> get_text >> monot5_relevance,
    splade_sans % k >> filter >> get_text >> monot5_relevance,
    splade_distil % k >> filter >> get_text >> monot5_relevance,
    bm25 % k >> filter >> get_text >> monot5_sans,
    splade_relevance % k >> filter >> get_text >> monot5_sans,
    splade_sans % k >> filter >> get_text >> monot5_sans,
    splade_distil % k >> filter >> get_text >> monot5_sans,
    
    # Rerank
    bm25 % k >>  get_text >> monot5_relevance,
    splade_relevance % k >> get_text >> monot5_relevance,
    splade_sans % k >> get_text >> monot5_relevance,
    splade_distil % k >> get_text >> monot5_relevance,
    bm25 % k >> get_text >> monot5_sans,
    splade_relevance % k  >> get_text >> monot5_sans,
    splade_sans % k >> get_text >> monot5_sans,
    splade_distil % k >> get_text >> monot5_sans,
]

names = [
    # No reranking
    f"bm25 % {k}",
    f"splade_relevance % {k}",
    f"splade_sans % {k}",
    f"splade_distil % {k}",
        
    # Filter
    f"bm25 % {k} >> filter",
    f"splade_relevance % {k} >> filter",
    f"splade_sans % {k} >> filter",
    f"splade_distil % {k} >> filter",
    
    # Filter >> Rerank
    f"bm25 % {k} >> filter >> get_text >> monot5_relevance",
    f"splade_relevance % {k} >> filter >> get_text >> monot5_relevance",
    f"splade_sans % {k} >> filter >> get_text >> monot5_relevance",
    f"splade_distil % {k} >> filter >> get_text >> monot5_relevance",
    f"bm25 % {k} >> filter >> get_text >> monot5_sans",
    f"splade_relevance % {k} >> filter >> get_text >> monot5_sans",
    f"splade_sans % {k} >> filter >> get_text >> monot5_sans",
    f"splade_distil % {k} >> filter >> get_text >> monot5_sans",
    
    # Rerank
    f"bm25 % {k} >> get_text >> monot5_relevance",
    f"splade_relevance % {k} >> get_text >> monot5_relevance",
    f"splade_sans % {k} >> get_text >> monot5_relevance",
    f"splade_distil % {k} >> get_text >> monot5_relevance",
    f"bm25 % {k} >> get_text >> monot5_sans",
    f"splade_relevance % {k}  >> get_text >> monot5_sans",
    f"splade_sans % {k} >> get_text >> monot5_sans",
    f"splade_distil % {k} >> get_text >> monot5_sans",
]

for_eval = []
paths = []
for pipeline, name in zip(pipelines, names):
    print(name)
    # results = pipeline(queries)
    # pt.io.write_results(results, f"./runs_wo_filter_v2/cut_1000/{name}")
    # for_eval.append(results)
    for_eval.append(pt.Transformer.from_df(pt.io.read_results(f"./runs_wo_filter_v2/{name}")))
    paths.append(f"./runs_wo_filter_v2/{name}")

exp = pt.Experiment(
    for_eval,
    queries,
    qrels,
    eval_metrics = [
        nDCG@10,
        sens_docs@10,
        nCSDCG_cut_10,
        nDCG@20,
        sens_docs@20,
        nCSDCG_cut_20,
        NSR@100,
        NSR@500,
        NSR@1000,
        NSRJudged@100,
        NSRJudged@500,
        NSRJudged@1000,
        MAP@100,
        MAP@1000,
        sens_docs@1000,
        sens_docs@100,
        R@100,
        R@1000,
        nDCG@1000
    ],
    names = names,
    verbose = True,
    perquery = True
)

# exp["path"] = paths

exp.to_csv("./runs_wo_filter_v2/results_w_paths_extra_metrics_perquery.csv", mode='a', header=not os.path.exists("./runs_wo_filter_v2/results_w_paths_extra_metrics_perquery.csv"), index=False)