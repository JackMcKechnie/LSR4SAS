import pyterrier as pt
import pandas as pd
import ir_datasets
import torch
from tqdm import tqdm
from transformers import DefaultDataCollator
from itertools import islice
import pandas as pd
import numpy as np
import re
import string

def flatten(xss):
    return [x for xs in xss for x in xs]


class MsMarcoTripletDataset():
    
    def __init__(self):
        self.dataset = ir_datasets.load("msmarco-passage/train/judged")
        print("Building doc_store...")
        self.doc_store = {d.doc_id: d.text for d in tqdm(self.dataset.docs_iter(), total = self.dataset.docs_count(), desc = "doc_store")}
        print("Docstore built!")
        print("Building query_store...")
        self.query_store = {q.query_id: q.text for idx, q in enumerate(tqdm(self.dataset.queries_iter(), total = self.dataset.queries_count(), desc = "query_store"))}
        print("Querystore built!")
        print("Building doc_pair_store...")
        # self.doc_pair_store = [d for d in tqdm(self.dataset.docpairs_iter(), total = self.dataset.docpairs_count(), desc = "doc_pair_store")]
        self.doc_pair_store = [d for d in tqdm(islice(self.dataset.docpairs_iter(), 100000), total = self.dataset.docpairs_count(), desc = "doc_pair_store")]
        print("Doc_pair_store built!")
        
    def __len__(self):
        return 100000
        return self.dataset.docpairs_count()

    def __getitem__(self, idx):
        query = self.query_store[self.doc_pair_store[idx].query_id]
        pos_doc = self.doc_store[self.doc_pair_store[idx].doc_id_a]
        neg_doc = self.doc_store[self.doc_pair_store[idx].doc_id_b]
        docs = [query, pos_doc, neg_doc]
        scores = torch.tensor([0, 0, 0]).view(1, -1)
        return docs, scores

class OHSUMEDDataset():
    
    def __init__(self):
        print("Loading docs")
        
        ohsumed_training_docs = self.get_d2qmm()
        
        sensitive_docs = ohsumed_training_docs[ohsumed_training_docs["sensitivity"] == 1].reset_index()
        non_sensitive_docs = ohsumed_training_docs[ohsumed_training_docs["sensitivity"] == 0].head(len(sensitive_docs)).reset_index()
        self.len = len(non_sensitive_docs)

        self.sensitive_docs = {index: f"{row['text']}" for index, row in sensitive_docs.iterrows()}
        self.non_sensitive_docs = {index: f"{row['text']}" for index, row in non_sensitive_docs.iterrows()}
        self.queries = {index: f"{row['querygen']}" for index, row in sensitive_docs.iterrows()}
        print("Docs loaded")

    def get_d2qmm(self):
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
        
    def __len__(self):
        return self.len
        
    def __getitem__(self, idx):
        query = self.queries[idx]
        pos_doc = self.sensitive_docs[idx]
        neg_doc = self.non_sensitive_docs[idx]
        docs = [query, pos_doc, neg_doc]
        scores = torch.tensor([0, 0, 0]).view(1, -1)
        return docs, scores

class OHSUMEDProportionateRandomSensitiveDataset():
    
    def __init__(self):
        print("Loading docs")
        
        ohsumed_training_docs = self.get_d2qmm()
        
        sensitive_docs = ohsumed_training_docs[ohsumed_training_docs["sensitivity"] == 1].reset_index()
        non_sensitive_docs = ohsumed_training_docs[ohsumed_training_docs["sensitivity"] == 0].head(len(sensitive_docs)).reset_index()
        self.len = len(ohsumed_training_docs)

        self.all_docs = ohsumed_training_docs
        self.sensitive_docs = sensitive_docs
        self.non_sensitive_docs = non_sensitive_docs
        self.queries = {index: f"{row['querygen']}" for index, row in ohsumed_training_docs.iterrows()}
        print("Docs loaded")

    def get_d2qmm(self):
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
        
    def __len__(self):
        return self.len
        
    def __getitem__(self, idx):
        current_doc = self.all_docs.iloc[[idx]]
        query = self.queries[idx]

        pos_doc = current_doc["text"].iloc[0]        
        if current_doc.sensitivity.iloc[0] == 1:
            neg_doc = self.sensitive_docs['text'].sample(n = 1, random_state = 42).iloc[0]
        else:
            neg_doc = self.all_docs['text'].sample(n = 1, random_state = 42).iloc[0]
            
        docs = [query, pos_doc, neg_doc]
        scores = torch.tensor([0, 0, 0]).view(1, -1)
        return docs, scores

class MsMarcoCollator(DefaultDataCollator):
    def __init__(self, tokenizer, max_length=350, *args, **kwargs):
        super(MsMarcoCollator, self).__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def torch_call(self, examples):
        docs,scores = zip(*examples)
        docs = flatten(docs)
        scores = torch.cat(scores,dim=0)
        tokenized = self.tokenizer(docs,
                           add_special_tokens=True,
                           padding="longest",
                           truncation="longest_first",
                           max_length=self.max_length,
                           return_attention_mask=True,
                           return_tensors="pt")       
        tokenized["scores"] = scores
        return tokenized        

from dataclasses import dataclass, field
from typing import Optional, Literal
from transformers import TrainingArguments

class SPLADEArgs(TrainingArguments):
    """
    SPLADE Arguments for training.
    """
    output_dir: str = field(
        metadata={"help": "Output path dir"},
        default=None
    )

    training_loss: str = field(
        metadata={"help": "Which losses to use: contrastive, kldiv, mse_margin, kldiv_mse_margin_with_weights, kldiv_mse_margin_without_weights, kldiv_contrastive_without_weights, kldiv_contrastive_with_weights"},
        default="kldiv_contrastive_with_weights"
    )

    l0d: float = field(
        metadata={"help": "lambda for document"},
        default=5e-4
    )

    l0q: float = field(
        metadata={"help": "lambda for query"},
        default=5e-4
    )

    T_d: int = field(
        metadata={"help": "Exponential FLOPS growth for lambda_d"},
        default=0
    )

    T_q: int = field(
        metadata={"help": "Exponential FLOPS growth for lambda_q"},
        default=0
    )

    top_d: int = field(
        metadata={"help": "TOP_k document pruning"},
        default=-1
    )

    top_q: int = field(
        metadata={"help": "TOP_k query pruning"},
        default=-1
    )

    lexical_type: str = field(
        metadata={"help": "Type of splade lexical to do: none, document, query or both"},
        default="none",
    )

    evaluation_strategy : str
    do_eval : bool