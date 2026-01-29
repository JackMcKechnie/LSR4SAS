import pyterrier as pt
if not pt.started():
    pt.init()
    
import ir_datasets
from ir_measures import *
import sys
sys.path.append("../lsr_package")

from lsr.transformer import LSR
from pyterrier_pisa import PisaIndex, PisaToksIndexer
from tqdm import tqdm

lsr = LSR("../lsr_package/outputs/splade_msmarco_distil_kl_div/model") # load a trained LSR model

dataset = pt.get_dataset('irds:msmarco-passage/trec-dl-2019/judged')

print("Starting indexing...")
# index_pipeline = lsr >> PisaToksIndexer("./idx/")
index = PisaIndex("./idx/", threads=32)

# index = index_pipeline.index(dataset.get_corpus_iter())
print("Indexing completed!")

splade_retr = retr_pipeline = lsr.query_encoder() >> index.quantized()
pt.io.write_results(splade_retr(dataset.get_topics()), "splade_msmarco_distil_kl_div")

retrieval_results = pt.Experiment(
    [splade_retr],
    dataset.get_topics(),
    dataset.get_qrels(),
    eval_metrics=[nDCG@10, RR(rel=2), AP(rel=2)]
)

print("Retrieval results:")
print(retrieval_results)