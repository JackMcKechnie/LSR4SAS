import pandas as pd
from pyterrier_pisa import PisaIndex
import pyterrier as pt

def ce_data_iterator():
    for _, doc in text.iterrows():
        yield {"docno" : doc.docno, "text" : doc.text}

queries = pd.read_csv("../data/training_queries.tsv", sep = "\t", names = ["qid", "query"])
text = pd.read_csv("../data/text.tsv", sep = "\t", names = ["docno", "text"])

# print("Indexing...")
index = PisaIndex("./ce_scoring_pisa_index", threads = 1)
# index.index(ce_data_iterator())
# print("Indexed!")

print("Running retrieval...")
bm25 = index.bm25(verbose = True)
retrieval_results = bm25(queries)
print("Retrieval done!")

pt.io.write_results(retrieval_results, "./bm25_all_training_data.run.gz")

