import pyterrier as pt
import pandas as pd
import pyt_splade
import argparse
from ir_measures import *
import pyterrier as pt
import sys
sys.path.append("../learned-sparse-retrieval-1.0.0")

from lsr.transformer import LSR
from pyterrier_pisa import PisaIndex, PisaToksIndexer
from tqdm import tqdm

# Argument parsing
parser = argparse.ArgumentParser(description='Evaluate trained SPLADE model on OHSUMED')

parser.add_argument('--model_path', type=str, help='Query model to be used.')
parser.add_argument('--index_loc', type=str, help='Path where the index is stored')
parser.add_argument('--run_output_path', type=str, help='Path to output the run file to')
parser.add_argument('--experiment_name', type=str, help='Name of the experiment')
args = parser.parse_args()

# Setup
docs = pd.read_pickle("/nfs/primary/sas_reranker/ohsumed_docs_w_t5base_sensitivity.pkl")
queries = pd.read_pickle("/nfs/primary/graph_adaptive_reranking/queries.pkl")
qrels = pd.read_pickle("/nfs/primary/graph_adaptive_reranking/qrels.pkl")

def _sens_docs(qrels, run):
    if "sensitivity" in run.columns:
        run = run.drop(columns = ["sensitivity"])
    merged = pd.merge(run, docs, left_on = "doc_id", right_on = "docno")
    return merged.sensitivity.sum()
    
import ir_measures
sens_docs = ir_measures.define_byquery(
    _sens_docs, 
    name="sens_docs")

lsr = LSR(args.model_path) # load a trained LSR model

# Index the corpus
print("Loading index...")
index = PisaIndex(args.index_loc)
print("Index loaded!")

splade_retr = lsr.query_encoder() >> index.quantized()

print("Starting retrieval...")
run_df = splade_retr(queries)
print("Retrieval completed!\n")

print("Saving results...")
pt.io.write_results(run_df, args.run_output_path)
print("Results saved!")

retrieval_results = pt.Experiment(
    [run_df],
    queries,
    qrels,
    eval_metrics=[nDCG@10, sens_docs@10]
)

print("Retrieval results:")
print(retrieval_results)

with open("./logs/test_logs.txt", 'a') as file:
    file.write(f"{args.experiment_name}")
    file.write(f"{retrieval_results}")
    file.write("\n")
    
print(f"Evaluation completed: Run using {args.model_path} saved in {args.run_output_path}. Index stored in {args.index_loc}")