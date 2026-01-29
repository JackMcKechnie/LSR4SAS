import pandas as pd
from pyterrier_t5 import MonoT5ReRanker
import argparse
from ir_measures import *
import pyterrier as pt
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os

if not pt.started():
    pt.init()

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default=1)
parser.add_argument('--output_run_path', type=str, default=42)
args = parser.parse_args()

ohsumed_docs = pd.read_pickle("/nfs/primary/sas_reranker/ohsumed_docs_w_t5base_sensitivity.pkl")
queries = pd.read_pickle("/nfs/primary/graph_adaptive_reranking/queries.pkl")
qrels = pd.read_pickle("/nfs/primary/graph_adaptive_reranking/qrels.pkl")

def _sens_docs(qrels, run):
    if "sensitivity" in run.columns:
        run = run.drop(columns = ["sensitivity"])
    merged = pd.merge(run, ohsumed_docs, left_on = "doc_id", right_on = "docno")
    return merged.sensitivity.sum()
    
import ir_measures
sens_docs = ir_measures.define_byquery(
    _sens_docs, 
    name="sens_docs")


def get_text(run):
    l = len(run)
    run = pd.merge(run, ohsumed_docs, on = "docno")
    assert len(run) == l
    return run

model = T5ForConditionalGeneration.from_pretrained(f"./{args.model_path}").cuda()
tokenizer = T5Tokenizer.from_pretrained("t5-base")

BATCH_SIZE = 16

reranker = MonoT5ReRanker(verbose=True, batch_size=BATCH_SIZE)
reranker.model = model

pipeline = pt.io.read_results("/nfs/primary/standard_run_files/ohsumed/bm25_terrier_1000.run") >> pt.apply.generic(get_text) >> reranker
results = pt.io.write_results(pipeline(queries), f"{args.output_run_path}/{args.model_path}_bm25_rerank")
eval_results = pt.Experiment(
    [
        pt.io.read_results(f"{args.output_run_path}/{args.model_path}_bm25_rerank")
    ]
    ,
    queries,
    qrels,
    eval_metrics=[nDCG@10, sens_docs@10],
    names = [f"{args.model_path}"]
)

file_path = "./monot5_sans1_hn_eval.csv"

# Append if exists, create if not
if os.path.exists(file_path):
    eval_results.to_csv(file_path, mode='a', header=False, index=False)
else:
    eval_results.to_csv(file_path, mode='w', header=True, index=False)

print("Finished evaluation!")