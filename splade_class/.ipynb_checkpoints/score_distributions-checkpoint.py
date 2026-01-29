#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd
from pyterrier_t5 import MonoT5ReRanker
import pyterrier as pt
if not pt.started():
    pt.init()


# In[20]:


training_docs = pd.read_pickle("/nfs/primary/sas_reranker/d2qmm_ohsumed_training_data.pkl")
training_docs["label"] = [1] * len(training_docs)


# In[27]:


def get_best_query(row):
    queries = row["querygen"].split("\n") if isinstance(row["querygen"], str) else []
    scores = list(row["querygen_score"])
    if queries and scores and len(queries) == len(scores):
        return max(zip(queries, scores), key=lambda x: x[1])[0]
    return ""  # or use "" if you prefer an empty string

training_docs["query"] = training_docs.apply(get_best_query, axis=1)
training_docs["qid"] = list(range(len(training_docs)))


# In[35]:


to_score = training_docs.sample(frac = 1, random_state=42)
to_score = to_score[["medline_ui", "text", "qid", "query", "sensitivity", "label"]]
to_score = to_score.rename(columns = {"medline_ui" : "docno"})
to_score


# In[40]:


shuffled = to_score[["docno", "text"]].sample(frac=1, random_state=42).reset_index(drop=True)

# Ensure the number of rows matches before assignment
to_score_shuffled = to_score.copy()
to_score_shuffled.loc[:, ["docno", "text"]] = shuffled.values  # Assign values safely
to_score_shuffled["label"] = [0] * len(to_score_shuffled)


# In[43]:


unscored = pd.concat([to_score, to_score_shuffled])
unscored


# In[ ]:


model_path = "/nfs/primary/sas_cross_encoder/trained_models/ohsumed_randomneg_synth_data_monot5_sas_best-1"
monot5_scorer = MonoT5ReRanker(model = model_path, batch_size = 128, verbose = True)
scored = monot5_scorer(unscored)


scored.to_csv("./all_random_score_distribution.csv", index = False)
pt.io.write_results(scored, "./all_random_score_distribution.run.gz")