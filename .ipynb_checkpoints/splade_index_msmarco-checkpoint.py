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


# In[2]:


dataset = pt.get_dataset("irds:msmarco-passage/trec-dl-2019/judged")


# In[3]:


splade = PyTerrierSPLADEBatch(
    model_name = 'naver/splade-cocondenser-ensembledistil',
    verbose = True,
    index_location = "./indices/msmarco_passage",
    index_name = "msmarco_passage"
)

# In[4]:

# splade.index(dataset.get_corpus_iter())


# In[8]:


index_ref = pt.IndexFactory.of("/nfs/primary/monoFiD/FiD-main/indices/msmarco-passage")
tfidf = pt.BatchRetrieve(index_ref, wmodel="TF_IDF")
bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25")

exp = pt.Experiment(
    [
        bm25,
        tfidf,
        splade
    ],
    dataset.get_topics(),
    dataset.get_qrels(),
    eval_metrics=[nDCG@10, R@1000],
    names = [
        "BM25",
        "TF-IDF",
        "SPLADE Co-Condenser Ensemble Distil"
    ]
)

print(exp)

