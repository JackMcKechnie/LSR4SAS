import pyterrier as pt
import pandas as pd
import pyt_splade
import argparse
from ir_measures import *
import pyterrier as pt
import sys
sys.path.append("./learned-sparse-retrieval-1.0.0")

from lsr.transformer import LSR
from pyterrier_pisa import PisaIndex, PisaToksIndexer
from tqdm import tqdm

import torch
import numpy as np
import itertools
from contextlib import ExitStack
from more_itertools import chunked

class CombinedModel(LSR):
    def __init__(self, model_path):
        super().__init__(model_path)

    def encode_docs(self, texts, out_fmt='dict', topk=None):
        outputs = []
        if out_fmt != 'dict':
            assert topk is None, "topk only supported when out_fmt='dict'"
        with ExitStack() as stack:
            stack.enter_context(torch.no_grad())
            if self.fp16:
                stack.enter_context(torch.cuda.amp.autocast())
            for batch in chunked(texts, self.batch_size):
                enc = self.tokenizer(batch, padding=True, truncation=True, return_special_tokens_mask=True, return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                res = self.model.encode_docs(**enc)
                assert res[:, 0].sum() == 0
                if out_fmt == 'dict':
                    res = self.vec2dicts(res, topk=topk)
                    outputs.extend(res)
                else:
                    outputs.append(res.cpu().float().numpy())
        if out_fmt == 'np':
            outputs = np.concatenate(outputs, axis=0)
        elif out_fmt == 'np_list':
            outputs = list(itertools.chain.from_iterable(outputs))
        return outputs

# Argument parsing
parser = argparse.ArgumentParser(description='Evaluate trained SPLADE model on OHSUMED')

parser.add_argument('--model_path', type=str, help='Model to be used.')
parser.add_argument('--index_loc', type=str, help='Path where the index should be stored')
parser.add_argument('--run_output_path', type=str, help='Path to output the run file to')
parser.add_argument('--experiment_name', type=str, help='Name of the experiment')

args = parser.parse_args()
# Setup
docs = pd.read_pickle("/nfs/primary/sas_reranker/ohsumed_docs_w_t5base_sensitivity.pkl")
queries = pd.read_pickle("/nfs/primary/graph_adaptive_reranking/queries.pkl")
qrels = pd.read_pickle("/nfs/primary/graph_adaptive_reranking/qrels.pkl")

# Util functions
def ohsumed_iterator():
    for row in tqdm(docs.iterrows(), total = len(docs), desc = "Indexing"):
        row = row[1]
        yield {"docno" : row.docno, "text" : row.text}

def _sens_docs(qrels, run):
    if "sensitivity" in run.columns:
        run = run.drop(columns = ["sensitivity"])
    merged = pd.merge(run, docs, left_on = "doc_id", right_on = "docno")
    return merged.sensitivity.sum()
    
import ir_measures
sens_docs = ir_measures.define_byquery(
    _sens_docs, 
    name="sens_docs")

print(args.model_path)
lsr = CombinedModel(args.model_path) # load a trained LSR model

# Index the corpus
print("Starting indexing...")
index_pipeline = lsr >> PisaToksIndexer(args.index_loc)
index = index_pipeline.index(ohsumed_iterator())
print("Indexing completed!")

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

with open("./logs/india_queue.txt", 'a') as file:
    file.write(f"{args.experiment_name}")
    file.write(f"{retrieval_results}")
    file.write("\n")
    
print(f"Evaluation completed: Run using {args.model_path} saved in {args.run_output_path}. Index stored in {args.index_loc}")