import pandas as pd
import pyterrier as pt
import numpy as np
from pyterrier_t5 import MonoT5ReRanker
import argparse
from pyterrier_pisa import PisaIndex

parser = argparse.ArgumentParser(description='Score corpus')
parser.add_argument('--model', help='Description for foo argument', required=True)
parser.add_argument('--name', help='Description for foo argument', required=True)

args = parser.parse_args()


training_docs = pd.read_pickle("/mnt/primary/sas_reranker/d2qmm_ohsumed_training_data.pkl")
training_docs = training_docs.rename(columns = {"medline_ui" : "docno"})
training_docs.docno = training_docs.docno.astype(str)

# args.start_qid = int(args.start_qid)
# args.end_qid = int(args.end_qid)

def get_best_query(row):
    queries = row['querygen'].split('\n')
    scores = row['querygen_score']
    if scores.size != 0:
        index = np.argmax(scores)
        best_query = queries[index]
    else:
        best_query = ""
    return best_query

training_docs['best_query'] = training_docs.apply(get_best_query, axis=1)

queries = pd.DataFrame({
    'qid': range(len(training_docs)),
    'query': training_docs['best_query']
})

# queries = queries.iloc[int(args.qid)]
# queries = pd.DataFrame(queries).T

queries.qid = queries.qid.astype(int)
# queries = queries[(queries['qid'] >= args.start_qid) & (queries['qid'] <= args.end_qid)]

# print(queries)
# print(args.start_qid, args.end_qid)

# index = PisaIndex("./indices/ohsumed_training.pisa/")
# bm25 = index.bm25(verbose = True, threads = 16)
# results = (bm25 % 100)(queries)

res = pt.io.read_results(f"./doc2query_score_files/bm25_to_rerank.run")
print(res.dtypes)
res.qid = res.qid.astype(int)

def get_text(run):
    new_run = pd.merge(training_docs, run, on = "docno")
    assert len(new_run) == len(run)
    return new_run

print(f"./doc2query_score_files/doc2query_monot5{args.name}_scored_top_100.run")
monot5 = MonoT5ReRanker(model = args.model, batch_size = 256)
pipeline = res >> pt.apply.generic(get_text) >> monot5
results = pipeline(queries)

pt.io.write_results(results, f"./doc2query_score_files/doc2query_monot5{args.name}_scored_top_100.run")