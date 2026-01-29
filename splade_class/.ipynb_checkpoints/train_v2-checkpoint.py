from collections import Counter
import torch
import pandas as pd
from typing import List
import pyterrier as pt
pt.init()
from pyt_splade import Splade
from transformers import BertTokenizer
from tqdm import tqdm
import pickle
import wandb

wandb.login(key="89144673f34dfdcbf35383f8c9040f5e7eb2fcb1")
wandb.init(project="splade_training", name="train_splade")


def train(inputs: List[str], labels: List[bool]):
    labels = torch.tensor(labels, dtype=torch.float32).cuda()
    print("Loading...")
    doc_vecs = torch.load("doc_vecs.pt").cuda()
    print("Loaded!")
    parameters = torch.nn.Parameter(torch.randn(len(doc_vecs[0]), device="cuda", requires_grad=True))

    optim = torch.optim.Adam([parameters], lr=0.01)
    for epoch in tqdm(range(25000), total = len(range(25000)), desc = "Training"):
        # Zero gradients at the start of each step
        optim.zero_grad()

        # Forward pass
        scores = torch.matmul(doc_vecs, torch.sigmoid(parameters))
        data_loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels)
        reg_loss = torch.sigmoid(parameters).sum()
        loss = data_loss + reg_loss

        # Backward pass and optimization
        loss.backward()
        optim.step()
        wandb.log({"epoch": epoch, "data_loss": data_loss.item(), "reg_loss": reg_loss.item(), "loss": loss.item()})

        with open("outputs.txt", "a") as f:
            f.write(f'Epoch {epoch}, Data Loss: {data_loss.item():.4f} Reg Loss: {reg_loss.item():.4f} Loss: {loss.item():.4f}\n')
    p = torch.sigmoid(parameters).detach().cpu().numpy()
    return p


def get_top_tokens(vector, tokenizer_name="bert-base-uncased", top_n=10):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    if not isinstance(vector, torch.Tensor):
        vector = torch.tensor(vector)
        
    top_indices = torch.topk(vector, top_n).indices.tolist()
    top_tokens = [tokenizer.convert_ids_to_tokens(idx) for idx in top_indices]
    return top_tokens

training_docs = pd.read_pickle("/nfs/primary/sas_reranker/d2qmm_ohsumed_training_data.pkl")

training_examples = pd.concat([
    training_docs[training_docs['sensitivity'] == 0].sample(n=5000, random_state=42),
    training_docs[training_docs['sensitivity'] == 1].sample(n=5000, random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)

p = train(training_docs.text.tolist(), training_docs.sensitivity.tolist())
print(get_top_tokens(p))

with open("outputs.txt", "a") as f:
    f.write(f'{get_top_tokens(p)}\n')
    f.write("---\n")
    
with open("trained_parameter_0.1_l1reg.pkl", "wb") as fp:   #Pickling
    pickle.dump(p, fp)

wandb.finish()