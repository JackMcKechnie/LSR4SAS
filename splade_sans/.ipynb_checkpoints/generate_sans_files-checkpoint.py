#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from tqdm import tqdm
from pyterrier_pisa import PisaIndex
import pyterrier as pt
import argparse
if not pt.started():
    pt.init()

parser = argparse.ArgumentParser(description='Generate SANS files')
parser.add_argument('--strategy')
args = parser.parse_args()

# In[2]:


training_file = pd.read_csv("./data/training_data_w_qids.csv")


# In[3]:

if args.strategy == "strat1":
    with open("./data/sans_strat1_7neg.tsv", "w") as f:
        sensitive_docs = training_file[training_file["sensitivity"] == 1]
        for _, row in tqdm(training_file.iterrows(), total = len(training_file), desc = "Generating negs"):
            for_sample = sensitive_docs[sensitive_docs["medline_ui"] != row.medline_ui]
            negatives = for_sample.sample(n=7, replace = True).medline_ui
            for negative in negatives:
                sample = row.qid, row.medline_ui, negative
                f.write(f"{sample[0]}\t{sample[1]}\t{sample[2]}\n")        


if args.strategy == "strat2":
    with open("./data/sans_strat2_7neg.tsv", "w") as f:
        sensitive_docs = training_file[training_file["sensitivity"] == 1]
        non_sensitive_docs = training_file[training_file["sensitivity"] == 0]
        for _, row in tqdm(training_file.iterrows(), total = len(training_file), desc = "Generating negs"):
            if row.sensitivity == 0:
                for_sample = training_file[training_file["medline_ui"] != row.medline_ui]
                negatives = for_sample.sample(n=7, replace = True).medline_ui
            else:
                for_sample = sensitive_docs[sensitive_docs["medline_ui"] != row.medline_ui]
                negatives = for_sample.sample(n=7, replace = True).medline_ui
                
            for negative in negatives:
                    sample = row.qid, row.medline_ui, negative
                    f.write(f"{sample[0]}\t{sample[1]}\t{sample[2]}\n")        

if args.strategy == "strat3":
    index_path = "./indices/pisa_index_v5"
    training_file = training_file.rename(columns = {"medline_ui" : "docno", "abstract_title" : "text"})
    training_file.docno = training_file.docno.astype(str)
    
    iter_indexer = PisaIndex(index_path, text_field = ["text"])
    indexref = iter_indexer.index(training_file.to_dict(orient="records"))
    threads = 1024
    index = PisaIndex(index_path, threads = threads)
    model = index.bm25(verbose = True) % 100
    model.search("Hello my name is Jack")
    
    
    # In[12]:
    
    
    training_file = training_file.rename(columns = {"docno" : "medline_ui", "text" : "abstract_title"})
    
    with open("./data/sans_strat3_7neg.tsv", "w") as f:
        sensitive_docs = training_file[training_file["sensitivity"] == 1]
        non_sensitive_docs = training_file[training_file["sensitivity"] == 0]
        for _, row in tqdm(training_file.iterrows(), total = len(training_file), desc = "Generating negs"):
            if row.sensitivity == 0:
                for_sample = training_file[training_file["medline_ui"] != row.medline_ui]
                if for_sample.empty:
                    continue
                negatives = for_sample.sample(n=7, replace = True).medline_ui
            else:
                bm25_res = model.search(row.best_query)
                if bm25_res.empty:
                    continue
                negatives = bm25_res.sample(n=7, replace = True).docno
                
            for negative in negatives:
                    sample = row.qid, row.medline_ui, negative
                    f.write(f"{sample[0]}\t{sample[1]}\t{sample[2]}\n")        
    
