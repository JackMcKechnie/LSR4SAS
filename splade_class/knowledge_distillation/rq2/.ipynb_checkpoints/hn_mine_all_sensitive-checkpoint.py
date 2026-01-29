#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import sys
sys.path.append("../../lsr_package")
from lsr.transformer import LSR
from pyterrier_pisa import PisaIndex, PisaToksIndexer
from tqdm import tqdm
import random
import json
import pyterrier as pt
if not pt.started():
    pt.init()


# In[2]:


model_path = "/nfs/primary/SPLADE/splade_sans/learned-sparse-retrieval-1.0.0/outputs/splade_max_strategy1_group8/model"


# In[3]:


model = LSR(model_path)


# In[4]:


full_df = pd.read_csv("/nfs/primary/SPLADE/splade_sans/data/training_data_w_qids.csv")
full_df


# In[5]:


lookup_dict = dict(zip(full_df["medline_ui"], full_df["text"]))


# In[6]:


def iterator():
    for _, doc in tqdm(full_df.iterrows(), total = len(full_df), desc = "indexing"):
        yield {"docno" : str(doc.medline_ui), "text" : doc.text}

# Index the corpus
# print("Starting indexing...")
# index_pipeline = model >> PisaToksIndexer("./indices/sans1_training_index")
# index = index_pipeline.index(iterator())
# print("Indexing completed!")

index = PisaIndex("./indices/sans1_training_index")


# In[7]:


def is_sensitive(docnos):
    full_df.medline_ui = full_df.medline_ui.astype(str)
    filtered_df = full_df[full_df['medline_ui'].isin(docnos)].sensitivity.sum()
    full_df.medline_ui = full_df.medline_ui.astype(int)
    return filtered_df 

def filter(run):
    original_len = len(run)
    full_df.medline_ui = full_df.medline_ui.astype(str)
    run = pd.merge(run, full_df, left_on = "docno", right_on = "medline_ui")
    assert len(run) == original_len, "merge error"
    run = run[run["sensitivity"] == 1]
    run = run.drop(columns= ["rank"])
    run = pt.model.add_ranks(run)
    full_df.medline_ui = full_df.medline_ui.astype(int)
    return run

def safe_sample(docnos, k):
    if len(docnos) == 0:
        return []
    if len(docnos) >= k:
        return random.sample(docnos, k)
    else:
        return random.choices(docnos, k=k)

full_df = full_df.rename(columns = {"qid" : "query_id_x"})


# In[ ]:


output_list = []
docnos_list = []

retr = model.query_encoder() >> index.quantized() >> pt.apply.generic(filter) % 50
for _, doc in tqdm(full_df.iterrows(), total = len(full_df), desc = "negative mining"):
    res = retr.search(str(doc.best_query))
    candidate_docnos = res.docno.tolist()
    filtered_docnos = [x for x in candidate_docnos if x != str(doc.medline_ui)]
    selected_docnos = safe_sample(filtered_docnos, k=8)
    if selected_docnos != []:
        output_list.append({"query" : str(doc.best_query), "docno_a" : lookup_dict[doc.medline_ui], "docno_b" : [lookup_dict[int(docno)] for docno in selected_docnos]})
        docnos_list.append({"query" : str(doc.query_id_x), "docno_a" : doc.medline_ui, "docno_b" : [docno for docno in selected_docnos]})


# In[ ]:


with open("hn_splade_sans1_all_sensitive_8_negs.jsonl", "w") as f:
    for item in output_list:
        f.write(json.dumps(item) + "\n")

with open("hn_splade_sans1_all_sensitive_8_negs_docnos.jsonl", "w") as f:
    for item in docnos_list:
        f.write(json.dumps(item) + "\n")


# In[ ]:


file_path = "hn_splade_sans1.jsonl"
OUTPUTS = ["true", "false"]

def iter_jsonl_with_index(filepath):
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            yield 'Query: ' + data["query"] + ' Document: ' + data["docno_a"] + ' Relevant:', OUTPUTS[0]
            for neg in data["docno_b"]:
                yield 'Query: ' + data["query"] + ' Document: ' + neg + ' Relevant:', OUTPUTS[1]
 
iter_jsonl_with_index(file_path)

