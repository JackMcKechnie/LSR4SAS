import pandas as pd
import pyterrier as pt
import pandas as pd
import numpy as np
if not pt.started():
    pt.init()
from pyterrier.model import add_ranks
import uuid
from ir_measures import *

# In[7]:


ohsumed_docs = pd.read_pickle("/nfs/primary/sas_reranker/ohsumed_docs_w_t5base_sensitivity.pkl")
queries = pd.read_pickle("/nfs/primary/graph_adaptive_reranking/queries.pkl")
qrels = pd.read_pickle("/nfs/primary/graph_adaptive_reranking/qrels.pkl")
df = pd.merge(qrels, ohsumed_docs, on = "docno")


# In[145]:


def shuffle(run):
    shuffled_run = run.groupby('qid', group_keys=False).apply(
        lambda x: x.sample(frac=1).assign(score=lambda df: len(df) - df.reset_index().index)
    ).reset_index(drop=True)
    return add_ranks(shuffled_run)

pipeline = pt.Transformer.from_df(df) >> pt.apply.generic(shuffle)

def _sens_docs(qrels, run):
    run.doc_id = run.doc_id.astype(str)
    original_run_length = len(run)
    if "sensitivity" in run.columns:
        run = run.drop(columns = ["sensitivity"])
    merged = pd.merge(run, ohsumed_docs, left_on = "doc_id", right_on = "docno")
    assert len(merged) == original_run_length
    return merged.sensitivity.sum()
    
import ir_measures
sens_docs = ir_measures.define_byquery(
    _sens_docs, 
    name="sens_docs")

while True:
    pipeline_results = pipeline(queries)
    
    exp = pt.Experiment(
        [pipeline_results],
        queries,
        qrels,
        eval_metrics = [nDCG@10,sens_docs@10],
        names = [str(uuid.uuid4())],
    )
    
    ndcg_val = round(exp.iloc[0]["nDCG@10"], 4)
    sens_docs_val = round(exp.iloc[0]["sens_docs@10"], 4)
    tolerance = 1e-6
    valid_values = [round(x * 0.1, 1) for x in range(1, 10)]  # [0.1, 0.2, ..., 0.9]
    
    for x in valid_values:
        if abs(ndcg_val - x) < tolerance and abs(sens_docs_val - x) < tolerance:
            matched_value = x
            pt.io.write_results(f"./matched_{matched_value}")
