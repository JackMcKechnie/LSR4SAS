#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import json
from tqdm import tqdm
import argparse

# In[3]:
parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--strat', help='Description for foo argument', required=True)
args = parser.parse_args()

training_data = pd.read_csv("/nfs/primary/SPLADE/splade_sans/data/training_data_w_qids.csv")


# In[23]:


non_sensitive_docs = training_data[training_data["sensitivity"] == 0]
sensitive_docs = training_data[training_data["sensitivity"] == 1]

if args.strat == "s1":

    qrels = {}
    for _, doc in tqdm(non_sensitive_docs.iterrows(), total = len(non_sensitive_docs), desc = "generating qrels"):
        qid = doc.qid
        positive = [doc.medline_ui]
        negatives = non_sensitive_docs.sample(n = 1).medline_ui.tolist()
        qrels[qid] = positive + negatives
    
    with open("./pos_rns_neg_nrns_qrels.json", "w") as f:
        json.dump(qrels, f)


# In[ ]:

if args.strat == "s2":
    qrels = {}
    for _, doc in tqdm(non_sensitive_docs.iterrows(), total = len(non_sensitive_docs), desc = "generating qrels"):
        qid = doc.qid
        positive = [doc.medline_ui]
        negatives = sensitive_docs.sample(n = 1).medline_ui.tolist()
        qrels[qid] = positive + negatives
    
    with open("./pos_rns_neg_nrs_qrels.json", "w") as f:
        json.dump(qrels, f)


# In[ ]:


if args.strat == "s3":
    qrels = {}
    for _, doc in tqdm(training_data.iterrows(), total = len(training_data), desc = "generating qrels"):
        qid = doc.qid
        if doc.sensitivity == 1:
            negatives = [doc.medline_ui]
            positives = non_sensitive_docs.sample(n = 1).medline_ui.tolist()
            qrels[qid] = positive + negatives
        elif doc.sensitivity == 0:
            positive = [doc.medline_ui]
            negatives = sensitive_docs.sample(n = 1).medline_ui.tolist()
            qrels[qid] = positive + negatives
        else:
            print("error")
    
    with open("./no_pos_rs_neg_nrs.json", "w") as f:
        json.dump(qrels, f)

if args.strat == "s4":
    qrels = {}
    for _, doc in tqdm(non_sensitive_docs.iterrows(), total = len(non_sensitive_docs), desc = "generating qrels"):
        qid = doc.qid
        positive = [doc.medline_ui]
        negatives = training_data.sample(n = 1).medline_ui.tolist()
        qrels[qid] = positive + negatives
    
    with open("./pos_rns_neg_random_qrels.json", "w") as f:
        json.dump(qrels, f)