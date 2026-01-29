import sys
sys.path.append("./lsr_package")
from transformer_w_progress import LSR
import pandas as pd
import torch

print(torch.cuda.is_available())

model_folder = "/nfs/primary/SPLADE/splade_class/lsr_package/outputs/splade_ohsumed_multiple_negative/model/"
index_folder = "/nfs/primary/SPLADE/splade_class/indices/sens_reg_exp/splade_ohsumed_multiple_negative"
splade = LSR(model_folder).doc_encoder(sparse = False)

training_docs = pd.read_pickle("/nfs/primary/sas_reranker/d2qmm_ohsumed_training_data.pkl")
texts = training_docs.text.tolist()
labels = training_docs.sensitivity.tolist()

tensor = torch.stack([torch.tensor(arr) for arr in splade(training_docs).doc_vec.tolist()])
torch.save(tensor, "doc_vecs_ohsumed_8_negs.pt")