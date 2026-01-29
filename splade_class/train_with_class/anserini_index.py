import pandas as pd
import pyterrier as pt
from pyterrier_anserini import AnseriniIndex
from pyterrier_pisa import PisaIndex


training_docs = pd.read_pickle("/mnt/primary/sas_reranker/d2qmm_ohsumed_training_data.pkl")

def iterator():
    for _, doc in training_docs.iterrows():
        yield {"docno" : str(doc["medline_ui"]), "text" : doc["text"]}

index = PisaIndex('./indices/ohsumed_training.pisa')
indexer = index.indexer()
print("Indexing...")
indexer.index(iterator())
print("Indexed!")