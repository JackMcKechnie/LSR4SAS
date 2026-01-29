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
from pyterrier_t5 import MonoT5ReRanker


parser = argparse.ArgumentParser(description="A simple argument parsing example.")

parser.add_argument('--strategy', type=str, help="Model to use for scoring")
args = parser.parse_args()

def read_collection(collection_path: str, text_fields=["text"]):
    doc_dict = {}
    if collection_path.startswith(IRDS_PREFIX):
        irds_name = collection_path.replace(IRDS_PREFIX, "")
        dataset = ir_datasets.load(irds_name)
        for doc in tqdm(
            dataset.docs_iter(),
            desc=f"Loading doc collection from ir_datasets: {irds_name}",
        ):
            doc_id = doc.doc_id
            texts = [getattr(doc, field) for field in text_fields]
            text = " ".join(texts)
            doc_dict[doc_id] = text
    elif collection_path.startswith(HFG_PREFIX):
        hfg_name = collection_path.replace(HFG_PREFIX, "")
        dataset = load_dataset(hfg_name)
        for row in tqdm(
            dataset["passage"],
            desc=f"Loading data from HuggingFace datasets: {hfg_name}",
        ):
            doc_dict[row["id"]] = row["text"]
    else:
        with open(collection_path, "r") as f:
            for line in tqdm(f, desc=f"Reading doc collection from {collection_path}"):
                doc_id, doc_text = line.strip().split("\t")
                doc_dict[doc_id] = doc_text
    return doc_dict


def read_queries(queries_path: str, text_fields=["text"]):
    queries = []
    if queries_path.startswith(IRDS_PREFIX):
        irds_name = queries_path.replace(IRDS_PREFIX, "")
        dataset = ir_datasets.load(irds_name)
        for query in tqdm(
            dataset.queries_iter(),
            desc=f"Loading queries from ir_datasets: {queries_path}",
        ):
            query_id = query.query_id
            texts = [getattr(query, field) for field in text_fields]
            text = " ".join(texts)
            queries.append((query_id, text))
    else:
        with open(queries_path, "r") as f:
            for line in tqdm(f, desc=f"Reading queries from {queries_path}"):
                query_id, query_text = line.strip().split("\t")
                queries.append((query_id, query_text))
    return queries


def read_qrels(qrels_path: str, rel_threshold=0):
    qid2pos = {}
    if qrels_path.startswith(IRDS_PREFIX):
        irds_name = qrels_path.replace(IRDS_PREFIX, "")
        dataset = ir_datasets.load(irds_name)
        for qrel in dataset.qrels_iter():
            if qrel.relevance > rel_threshold:
                qid, did = qrel.query_id, qrel.doc_id
                if not qid in qid2pos:
                    qid2pos[qid] = []
                qid2pos[qid].append(did)
    else:
        qrels = json.load(open(qrels_path, "r"))
        for qid in qrels:
            qid2pos[str(qid)] = [str(did) for did in qrels[qid]]
    return qid2pos


file_map = {
    "sentence-transformers/msmarco-hard-negatives": "https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz"
}


def read_ce_score(ce_path: str):
    if ce_path.startswith(HFG_PREFIX):
        hf_name = ce_path.replace(HFG_PREFIX, "")
        _url = file_map[hf_name]
        dl_manager = DownloadManager()
        ce_path = dl_manager.download(_url)
    res = {}
    with gzip.open(ce_path, "rb") as f:
        data = pickle.load(f)
        for qid in tqdm(data, desc=f"Preprocessing data from {ce_path}"):
            res[str(qid)] = {str(did): data[qid][did] for did in data[qid]}
    return res



IRDS_PREFIX = "irds"
HFG_PREFIX = "hf"
queries = read_queries( '/nfs/primary/SPLADE/splade_sans/data/training_queries.tsv')
qrels = read_qrels(f'/nfs/primary/SPLADE/splade_sans/data/{args.strategy}_training_qrels.json')
doc_dict = read_collection('/nfs/primary/SPLADE/splade_sans/data/text.tsv')

if args.strategy == "strategy1":
    model_path = "/nfs/primary/sas_cross_encoder/trained_models/ohsumed_randomneg_synth_data_monot5_sas_best-1"
elif args.strategy == "strategy2":
    model_path = "/nfs/primary/sas_cross_encoder/trained_models/ohsumed_1bm25_proportionate_neg_synth_data_monot5_sas_random_negs_best-31"
elif args.strategy == "strategy3":
    model_path = "/nfs/primary/sas_cross_encoder/trained_models/ohsumed_synth_data_proportionate_negs_sas_easy_negs_best-21"
else:
    raise ValueError(f"Invalid strategy: {args.model_choice}")


monot5_scorer = MonoT5ReRanker(model = model_path, batch_size = 8, verbose = False)

output_dict = {}
for qid in tqdm(qrels.keys(), total = len(qrels.keys()), desc = "processing and scoring"):
    try:
        query = queries[int(qid)][1]
    except:
        query = "error"
    for_df = []
    for docno in qrels[qid]:
        text = doc_dict[docno]
        for_df.append({"qid" : qid, "query" : query, "docno" : docno, "text" : text})
    
    unscored = pd.DataFrame(for_df)
    scored = monot5_scorer(unscored)
    scored = scored.drop_duplicates(subset=["qid", "docno"])
    output_dict.update(scored.set_index(["qid", "docno"])["score"].unstack(level=0).to_dict())


with gzip.open(f"./ce_score_{args.strategy}.pkl.gz", "wb") as f:
    pickle.dump(output_dict, f)