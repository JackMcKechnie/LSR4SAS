import pandas as pd
import pyterrier as pt
if not pt.started():
    pt.init()
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import torch
import sys
sys.path.append("./lsr_package/")
from lsr.transformer import LSR
from pyterrier.model import add_ranks
from ir_measures import *
import argparse

parser = argparse.ArgumentParser(description='Evaluate SPLADE model on OHSUMED')
parser.add_argument('--i', type=int, help='dims to remove')

args = parser.parse_args()

docs = pd.read_pickle("/nfs/primary/sas_reranker/ohsumed_docs_w_t5base_sensitivity.pkl")
queries = pd.read_pickle("/nfs/primary/graph_adaptive_reranking/queries.pkl")
qrels = pd.read_pickle("/nfs/primary/graph_adaptive_reranking/qrels.pkl")

sensitive_texts = docs[docs["sensitivity"] == 1].text.tolist()
non_sensitive_texts = docs[docs["sensitivity"] == 0].text.tolist()

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def _sens_docs(qrels, run):
    if "sensitivity" in run.columns:
        run = run.drop(columns = ["sensitivity"])
    merged = pd.merge(run, docs, left_on = "doc_id", right_on = "docno")
    return merged.sensitivity.sum()
    
import ir_measures
sens_docs = ir_measures.define_byquery(
    _sens_docs, 
    name="sens_docs")

vocab_size = len(tokenizer.vocab)
sensitive_frequency_matrix = np.zeros((len(sensitive_texts), vocab_size), dtype=int)
non_sensitive_frequency_matrix = np.zeros((len(non_sensitive_texts), vocab_size), dtype=int)

for i, text in tqdm(enumerate(sensitive_texts), total = len(sensitive_texts), desc = "Tokenizing"):
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    sensitive_frequency_matrix[i], _ = np.histogram(token_ids, bins=np.arange(vocab_size + 1))

for i, text in tqdm(enumerate(non_sensitive_texts), total = len(non_sensitive_texts), desc = "Tokenizing"):
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    non_sensitive_frequency_matrix[i], _ = np.histogram(token_ids, bins=np.arange(vocab_size + 1))

oracle_vector = sensitive_frequency_matrix.sum(axis = 0) - non_sensitive_frequency_matrix.sum(axis = 0)

docs_encoded = torch.load("docs_encoded.pt")
queries_encoded = torch.load("queries_encoded.pt")
docnos = docs.docno.tolist()
qids = queries.qid.tolist()

def oracle_zero_out(k):
    top_k_indices = np.argpartition(oracle_vector, -k)[-k:]
    mask = np.ones_like(oracle_vector)
    mask[top_k_indices] = 0
    zeroed_docs = docs_encoded * mask
    zeroed_docs = zeroed_docs.to(torch.float32)
    scores = zeroed_docs @ queries_encoded.T  # Shape: (14430, 106)
    df_scores = pd.DataFrame(scores, index=docnos, columns=qids)
    
    # Melt into long format (docno, qid, score)
    res = df_scores.reset_index().melt(id_vars="index", var_name="qid", value_name="score")
    
    # Rename columns
    res.rename(columns={"index": "docno"}, inplace=True)
    res = add_ranks(res)
    return res

retrieval_results = pt.Experiment(
    [
        oracle_zero_out(args.i)
    ],
    queries,
    qrels,
    eval_metrics=[nDCG@10, sens_docs@10],
    names = [
        f"{args.i}"
    ]
)

print(retrieval_results)
retrieval_results.to_csv("./oracle_runs/results.csv", mode='a', header=not pd.io.common.file_exists("./oracle_runs/results.csv"), index=False)


pt.io.write_results(oracle_zero_out(args.i), f"./oracle_runs/oracle_{args.i}_removed.res.gz")