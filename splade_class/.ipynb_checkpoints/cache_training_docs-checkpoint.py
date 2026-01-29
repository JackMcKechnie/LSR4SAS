import pandas as pd
from pyt_splade import Splade
import pickle
import sys
sys.path.append("./lsr_package")
from lsr.transformer import LSR

# splade = Splade().doc_encoder(verbose = True)

model_folder = "/nfs/primary/SPLADE/splade_class/lsr_package/outputs/splade_ohsumed_multiple_negative/model/"
index_folder = "/nfs/primary/SPLADE/splade_class/indices/sens_reg_exp/splade_ohsumed_multiple_negative"
splade = LSR(model_folder).doc_encoder()

training_docs = pd.read_pickle("/nfs/primary/sas_reranker/d2qmm_ohsumed_training_data.pkl")
texts = training_docs.text.tolist()
labels = training_docs.sensitivity.tolist()

encoded_inputs = list(splade(pd.DataFrame({
    'docno': list(range(len(texts))),
    'text': texts,
}))['toks'])

with open("encoded_training_docs_cache_ohsumed_8_negs.pkl", "wb") as fp:   #Pickling
    pickle.dump(encoded_inputs, fp)

with open("training_labels_cache.pkl", "wb") as fp:   #Pickling
    pickle.dump(labels, fp)