#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyterrier as pt
if not pt.started():
    pt.init()
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
from pyterrier_splade import PyTerrierSPLADE
import pyterrier_dr
from ir_measures import *
import argparse
import pandas as pd
import re
import string

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument('--dataset')
parser.add_argument("--model_path")
parser.add_argument("--index_location")
parser.add_argument("--index_name")
parser.add_argument("--run_file_path")


args = parser.parse_args()



splade = PyTerrierSPLADE(
    model_name = args.model_path,
    verbose = True,
    index_location = args.index_location,
    index_name = args.index_name
)

if "ohsumed" not in args.dataset:
    dataset = pt.get_dataset(args.dataset)
    splade.index(dataset.get_corpus_iter())
    pt.io.write_results(splade(dataset.get_topics()), args.run_file_path)


if args.dataset == "ohsumed":
    ohsumed_docs = pd.read_pickle("/nfs/primary/sas_reranker/ohsumed_docs_w_t5base_sensitivity.pkl")
    queries = pd.read_pickle("/nfs/primary/graph_adaptive_reranking/queries.pkl")
    qrels = pd.read_pickle("/nfs/primary/graph_adaptive_reranking/qrels.pkl")
    def ohsumed_iterator():
        for idx, doc in ohsumed_docs.iterrows():
            yield {"docno" : doc["docno"], "text" : doc["text"]}
    splade.index(ohsumed_iterator())
    pt.io.write_results(splade(queries), args.run_file_path)
def get_d2qmm():
    d2qmm = pd.read_pickle("/nfs/primary/sas_cross_encoder/d2qmm_ohsumed_10samples_0.1filter.pkl")
    max_score_indices = []
    for row_scores in d2qmm['querygen_score']:
        if len(row_scores) > 0:
            max_score_indices.append(np.argmax(np.array(row_scores)))
        else:
            # Handle the case of an empty list
            max_score_indices.append(None)

    # Update 'querygen' to keep only the query with the top score
    d2qmm['querygen'] = [queries.split('\n')[index] if (index is not None and queries) else None for queries, index in zip(d2qmm['querygen'], max_score_indices)]

    # Drop the 'querygen_score' column as it's no longer needed
    d2qmm = d2qmm.drop('querygen_score', axis=1)
    d2qmm['querygen'] = d2qmm['querygen'].str.replace(f"[{re.escape(string.punctuation)}]", "", regex = True)
    
    return d2qmm


if args.dataset == "ohsumed_train_non_sensitive":
    ohsumed_training_docs = get_d2qmm()
    ohsumed_training_docs.medline_ui = ohsumed_training_docs.medline_ui.astype(str)
    non_sensitive_documents = ohsumed_training_docs[ohsumed_training_docs["sensitivity"] == 0]
    print(len(non_sensitive_documents))
    def ohsumed_iterator():
        for idx, doc in non_sensitive_documents.iterrows():
            yield {"docno" : doc["medline_ui"], "text" : doc["text"]}
    splade.index(ohsumed_iterator())

if args.dataset == "ohsumed_train_sensitive":
    ohsumed_training_docs = get_d2qmm()
    ohsumed_training_docs.medline_ui = ohsumed_training_docs.medline_ui.astype(str)
    sensitive_documents = ohsumed_training_docs[ohsumed_training_docs["sensitivity"] == 1]
    print(len(sensitive_documents))
    def ohsumed_iterator():
        for idx, doc in sensitive_documents.iterrows():
            yield {"docno" : doc["medline_ui"], "text" : doc["text"]}
    splade.index(ohsumed_iterator())

if args.dataset == "ohsumed_train_all":
    ohsumed_training_docs = get_d2qmm()
    ohsumed_training_docs.medline_ui = ohsumed_training_docs.medline_ui.astype(str)
    def ohsumed_iterator():
        for idx, doc in ohsumed_training_docs.iterrows():
            yield {"docno" : doc["medline_ui"], "text" : doc["text"]}
    splade.index(ohsumed_iterator())


if args.dataset == "irds:vaswani":
    bm25 = pt.BatchRetrieve(pt.get_dataset("vaswani").get_index(), wmodel="BM25")
    tfidf = pt.BatchRetrieve(pt.get_dataset("vaswani").get_index(), wmodel="TF_IDF")

    exp = pt.Experiment(
        [
            bm25,
            tfidf,
            pt.io.read_results(args.run_file_path)
        ],
        dataset.get_topics(),
        dataset.get_qrels(),
        eval_metrics=[nDCG@10, R@1000],
        names = [
            "BM25",
            "TF-IDF",
            "SPLADE"
        ]
    )
    
    print(exp)

if args.dataset == "ohsumed":
    index_ref = pt.IndexFactory.of("/nfs/primary/listwise/indices/ohsumed")
    bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25")
    tfidf = pt.BatchRetrieve(index_ref, wmodel="TF_IDF")


    exp = pt.Experiment(
        [
            bm25,
            tfidf,
            pt.io.read_results(args.run_file_path)
        ],
        queries,
        qrels,
        eval_metrics=[nDCG@10, R@1000],
        names = [
            "BM25",
            "TF-IDF",
            "SPLADE"
        ]
    )
    
    print(exp)
    

