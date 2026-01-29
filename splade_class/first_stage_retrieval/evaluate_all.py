import pandas as pd
import pyterrier as pt
if not pt.started():
    pt.init()
import pyterrier_pisa
from pyterrier_t5 import *
import os
import argparse
from ir_measures import *

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
    combined = pd.merge(run, qrels, on = ["doc_id"])
    combined = pd.merge(combined, ohsumed_docs, left_on = "doc_id", right_on = "docno")
    combined.sensitivity = combined.sensitivity.astype(int)
    combined.relevance = combined.relevance.astype(int)
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

directory_path = './runs'

# List all files (excluding directories)
files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f)) and "csv" not in f]

names = []
run_files = []

for file in files:
    if "kill" not in file and "k" not in file:
        names.append(file)
        run_files.append(pt.io.read_results(f"./runs/{file}"))
        
exp = pt.Experiment(
    run_files,
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
        NSR@1000
    ],
    names = names,
    verbose = True,
    perquery = True
)

exp.to_csv("./runs/results_evaluate_all_v3.csv", mode='a', header=not os.path.exists("./runs/results_evaluate_all.csv"), index=False)