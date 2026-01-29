import pyterrier as pt
pt.init()

import pyt_splade
from pyterrier_splade import PyTerrierSPLADE
from ir_measures import *

dataset = pt.get_dataset("irds:vaswani")
craig_splade = pyt_splade.SpladeFactory()
craig_splade_pipe = craig_splade.query() >> pt.BatchRetrieve('./indices/vaswani_craig_test/', wmodel='Tf')


jack_splade = PyTerrierSPLADE(
    model_name = "naver/splade-cocondenser-ensembledistil",
    verbose = True,
    index_location = "./indices/vaswani_sparse_test",
    index_name = "vaswani_sparse_test"
)

exp = pt.Experiment(
    [craig_splade_pipe, jack_splade],
    dataset.get_topics(),
    dataset.get_qrels(),
    eval_metrics=[nDCG@10, "mrt"],
    names = ["Craig SPLADE", "Jack SPLADE"]
)

print(exp)