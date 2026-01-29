import ir_datasets
import numpy as np
from tqdm import tqdm
import random
from tqdm import tqdm
import ir_datasets
# import irds_robust_anserini
import json
from datasets import DownloadManager
from collections import defaultdict
import gzip
import pickle
from pathlib import Path
import requests
import json
import sys
from datasets import load_dataset
import numpy as np
import pandas as pd
import glob
import torch
import hashlib
import copy

IRDS_PREFIX = "irds:"
HFG_PREFIX = "hfds:"

file_map = {
    "sentence-transformers/msmarco-hard-negatives": "https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz"
}


def read_qrels(qrels_path: str, rel_threshold=0):
    qid2pos = defaultdict(dict)
    if qrels_path.startswith(IRDS_PREFIX):
        irds_name = qrels_path.replace(IRDS_PREFIX, "")
        dataset = ir_datasets.load(irds_name)
        for qrel in dataset.qrels_iter():
            # if qrel.relevance > rel_threshold:
            qid, did = qrel.query_id, qrel.doc_id
            qid2pos[qid][did] = qrel.relevance
    else:
        qrels = json.load(open(qrels_path, "r"))
        for qid in qrels:
            qid2pos[str(qid)] = {str(did): qrels[qid][did]
                                 for did in qrels[qid]}
    return qid2pos
    
def read_ce_score(ce_path: str):
    if ce_path.endswith(".json"):
        return json.load(open(ce_path, "r"))
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

random.seed(42)
ce_scores = read_ce_score("hfds:sentence-transformers/msmarco-hard-negatives")
qrels = read_qrels("irds:msmarco-passage/train")
query_ids = list(qrels.keys())
with open("./msmarco_triplets.tsv", "w") as f:
    for epoch in tqdm(range(50)):
        np.random.shuffle(query_ids)
        for q_id in query_ids:
            pos_id = random.choice(list(qrels[q_id].keys()))
            if pos_id in ce_scores[q_id]:
                neg_lists = list(ce_scores[q_id].keys())
                neg_id = pos_id
                while neg_id == pos_id:
                    neg_id = random.choice(neg_lists)
                f.write(
                    f"{q_id}\t{pos_id}\t{ce_scores[q_id][pos_id]}\t{neg_id}\t{ce_scores[q_id][neg_id]}\n")