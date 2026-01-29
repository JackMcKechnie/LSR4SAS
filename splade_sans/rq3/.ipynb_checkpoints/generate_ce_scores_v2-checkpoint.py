import pandas as pd
import pyterrier as pt
import math
import warnings
import itertools
import pyterrier as pt
from collections import defaultdict
from pyterrier.model import add_ranks
import torch
from torch.nn import functional as F
from transformers import T5Tokenizer, T5ForConditionalGeneration, MT5ForConditionalGeneration
import os
from tqdm import tqdm
import json
import gzip
import pickle
import argparse

parser = argparse.ArgumentParser(description="A simple argument parsing example.")

parser.add_argument('--model_choice', type=str, help="Model to use for scoring")
parser.add_argument('--cutoff', type=int, help="K cutoff to use")
parser.add_argument('--output_path', type=str, help="Path to save scores at")

args = parser.parse_args()

print("Loading and setting up...")

queries = pd.read_csv("../data/training_queries.tsv", sep = "\t", names = ["qid", "query"])
text = pd.read_csv("../data/text.tsv", sep = "\t", names = ["docno", "text"])
queries = pt.model.coerce_dataframe_types(queries)
text = pt.model.coerce_dataframe_types(text)
bm25_res = pt.io.read_results("./bm25_k1000_training_data.run.gz")
bm25_res = pt.Transformer.from_df(bm25_res)


if args.model_choice == "strategy1":
    model_path = "/nfs/primary/sas_cross_encoder/trained_models/ohsumed_randomneg_synth_data_monot5_sas_best-1"
elif args.model_choice == "strategy2":
    model_path = "/nfs/primary/sas_cross_encoder/trained_models/ohsumed_1bm25_proportionate_neg_synth_data_monot5_sas_random_negs_best-31"
elif args.model_choice == "strategy3":
    model_path = "/nfs/primary/sas_cross_encoder/trained_models/ohsumed_synth_data_proportionate_negs_sas_easy_negs_best-21"
else:
    raise ValueError(f"Invalid strategy: {args.model_choice}")

def get_text(run):
    run = pd.merge(run, text, on = "docno")
    return run
    
def transform_data_to_ce_format(data):
    res = {}
    # Iterate over each sublist in the data
    for sublist in data:
        # Iterate over each dictionary in the sublist
        for entry in sublist:
            qid = str(entry['qid'])  # Convert qid to string
            docno = str(entry['docno'])  # Convert docno to string
            score = entry['score']
            
            # Initialize the dictionary for the qid if it doesn't exist
            if qid not in res:
                res[qid] = {}
                
            # Assign the score to the docno for the given qid
            res[qid][docno] = score
    
    return res

def convert():
    with open(args.output_path, 'r') as file:
        data = json.load(file)
    
    data = transform_data_to_ce_format(data)
    
    with gzip.open(f'{args.output_path}.pkl.gz', 'wb') as f:
        pickle.dump(data, f)
        
class MonoT5ReRanker(pt.Transformer):
    def __init__(self, 
                 tok_model='t5-base',
                 model='castorini/monot5-base-msmarco',
                 batch_size=4,
                 text_field='text',
                 verbose=True,
                 output_path = None):
        self.verbose = verbose
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = T5Tokenizer.from_pretrained(tok_model)
        self.model_name = model
        self.model = T5ForConditionalGeneration.from_pretrained(model)
        self.model.to(self.device)
        self.model.eval()
        self.text_field = text_field
        self.REL = self.tokenizer.encode('true')[0]
        self.NREL = self.tokenizer.encode('false')[0]
        self.output_path = output_path

    def __str__(self):
        return f"MonoT5({self.model_name})"

    def transform(self, run):
            scores = []
            queries, texts = run['query'], run[self.text_field]
            it = range(0, len(queries), self.batch_size)
            prompts = self.tokenizer.batch_encode_plus(
                ['Relevant:' for _ in range(self.batch_size)],
                return_tensors='pt', 
                padding='longest'
            )
            max_vlen = self.model.config.n_positions - prompts['input_ids'].shape[1]
    
            # Prepare file for incremental saving
            output_file = self.output_path
                
            with open(output_file, 'a') as f:
                first_batch = os.path.getsize(output_file) == 0  # Check if file is empty (first batch)
                
                if first_batch:
                    f.write('[')  # Start the JSON array
                
                if self.verbose:
                    it = tqdm(it, desc='monoT5', unit='batches')
    
                for start_idx in it:
                    rng = slice(start_idx, start_idx + self.batch_size)
                    enc = self.tokenizer.batch_encode_plus(
                        [f'Query: {q} Document: {d}' for q, d in zip(queries[rng], texts[rng])],
                        return_tensors='pt', 
                        padding='longest'
                    )
                    for key, enc_value in list(enc.items()):
                        enc_value = enc_value[:, :-1]  # Chop off end of sequence token
                        enc_value = enc_value[:, :max_vlen]  # Truncate to max length
                        enc[key] = torch.cat([enc_value, prompts[key][:enc_value.shape[0]]], dim=1)  # Add prompt
    
                    enc['decoder_input_ids'] = torch.full(
                        (len(queries[rng]), 1),
                        self.model.config.decoder_start_token_id,
                        dtype=torch.long
                    )
                    enc = {k: v.to(self.device) for k, v in enc.items()}
    
                    with torch.no_grad():
                        result = self.model(**enc).logits
                    result = result[:, 0, (self.REL, self.NREL)]
                    scores += F.log_softmax(result, dim=1)[:, 0].cpu().detach().tolist()
    
                    current_slice = run.iloc[start_idx:start_idx + self.batch_size]
                    out_qids = current_slice.qid.tolist()
                    out_docnos = current_slice.docno.tolist()
                    out_scores = scores[start_idx:start_idx + self.batch_size]
    
                    # Write the batch results incrementally to the file
                    batch_results = []
                    for qid, docno, score in zip(out_qids, out_docnos, out_scores):
                        batch_results.append({
                            "qid": qid,
                            "docno": docno,
                            "score": score
                        })
                    
                    # If not the first batch, write a comma before appending
                    if not first_batch:
                        f.write(',\n')
    
                    # Write the results for this batch to the file
                    json.dump(batch_results, f)
                    
                    # Set first_batch to False after the first batch is processed
                    first_batch = False
    
                # Close the JSON array after the last batch
                f.write(']')
            
            run = run.drop(columns=['score', 'rank'], errors='ignore').assign(score=scores)
            run = add_ranks(run)
            return run
            
ce = MonoT5ReRanker(model = model_path, output_path = args.output_path, batch_size = 64)
pipeline = bm25_res % args.cutoff >> pt.apply.generic(get_text) >> ce

print("Loading and setup done!")

print("Scoring...")
results = pipeline(queries)
pt.io.write_results(results, f"{args.output_path.split('.json')[0]}.run")
print("Scoring done...")

print("Converting...")
convert()
print("Converted!")