import pyt_splade
import pyterrier_dr
import pyterrier as pt


splade = pyt_splade.Splade()
dataset = pt.get_dataset('irds:vaswani')
index = pyterrier_dr.FlexIndex('./indices/vaswani-splade')

# indexing
idx_pipeline = splade.doc_encoder(sparse=False) >> index
idx_pipeline.index(dataset.get_corpus_iter())

# retrieval

retr_pipeline = splade.query_encoder(sparse=False) >> index.np_retriever()
print(retr_pipeline(dataset.get_topics()))