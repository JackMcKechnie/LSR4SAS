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
import argparse


# parser = argparse.ArgumentParser(description='Evaluate SPLADE model on OHSUMED')
# parser.add_argument('--alpha', type=str, help='Alpha value')
# args = parser.parse_args()

# alpha = float(args.alpha)

wandb.login(key="89144673f34dfdcbf35383f8c9040f5e7eb2fcb1")
wandb.init(project="splade_training", name=f"train_splade")

def train(inputs: List[str], labels: List[bool], batch_size=1024):
    labels = torch.tensor(labels, dtype=torch.float32).cuda()
    
    print("Loading...")
    # Load doc_vecs in CPU first
    doc_vecs = torch.load("doc_vecs_ohsumed_8_negs.pt", map_location="cpu")
    print("Loaded!")

    # Initialize parameters
    parameters = torch.nn.Parameter(torch.randn(len(doc_vecs[0]), device="cuda", requires_grad=True))
    optim = torch.optim.Adam([parameters], lr=0.01)

    num_batches = (len(doc_vecs) + batch_size - 1) // batch_size  # Compute total batches

    for epoch in range(500):
        epoch_data_loss = 0.0
        epoch_reg_loss = 0.0
        epoch_total_loss = 0.0

        # Shuffle indices to get different batches each epoch
        indices = torch.randperm(len(doc_vecs))

        for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch}"):
            batch_indices = indices[batch_idx * batch_size: (batch_idx + 1) * batch_size]

            # Load batch to GPU
            batch_vecs = doc_vecs[batch_indices].cuda()
            batch_labels = labels[batch_indices]

            optim.zero_grad()

            # Forward pass
            scores = torch.matmul(batch_vecs, parameters)
            data_loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, batch_labels)
            reg_loss = torch.sigmoid(parameters).abs().sum()
            loss = data_loss + reg_loss

            # Backward pass and optimization
            loss.backward()
            optim.step()

            # Track loss separately
            epoch_data_loss += data_loss.item()
            epoch_reg_loss += reg_loss.item()
            epoch_total_loss += loss.item()

        # Compute average loss per batch
        avg_data_loss = epoch_data_loss / num_batches
        avg_reg_loss = epoch_reg_loss / num_batches
        avg_total_loss = epoch_total_loss / num_batches

        # Log all losses separately to WandB
        wandb.log({"epoch": epoch, "data_loss": avg_data_loss, "reg_loss": avg_reg_loss, "loss": avg_total_loss})

        # Log losses to file
        with open("outputs.txt", "a") as f:
            f.write(f'Epoch {epoch}, Data Loss: {avg_data_loss:.4f}, Reg Loss: {avg_reg_loss:.4f}, Total Loss: {avg_total_loss:.4f}\n')

    # p = torch.sigmoid(parameters).detach().cpu().numpy()
    p = parameters.detach().cpu().numpy()
    return p


def get_top_tokens(vector, tokenizer_name="bert-base-uncased", top_n=10):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    if not isinstance(vector, torch.Tensor):
        vector = torch.tensor(vector)
        
    top_indices = torch.topk(vector, top_n).indices.tolist()
    top_tokens = [tokenizer.convert_ids_to_tokens(idx) for idx in top_indices]
    return top_tokens


# Load training data
training_docs = pd.read_pickle("/nfs/primary/sas_reranker/d2qmm_ohsumed_training_data.pkl")

# Balance and shuffle training set
training_examples = pd.concat([
    training_docs[training_docs['sensitivity'] == 0].sample(n=5000, random_state=42),
    training_docs[training_docs['sensitivity'] == 1].sample(n=5000, random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)

# Train model with batch processing
p = train(training_docs.text.tolist(), training_docs.sensitivity.tolist(), batch_size=1024)

# Get top tokens
def get_top_tokens(vector, tokenizer_name="bert-base-uncased", top_n=100):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    if not isinstance(vector, torch.Tensor):
        vector = torch.tensor(vector)

    # Get top N indices and values
    top_k = torch.topk(vector, top_n)
    top_indices = top_k.indices.tolist()
    top_values = top_k.values.tolist()

    # Convert token IDs to actual tokens
    top_tokens = [tokenizer.convert_ids_to_tokens(idx) for idx in top_indices]

    # Return list of tuples (token, value)
    top_tokens_with_values = list(zip(top_tokens, top_values))
    with open("top_tokens.txt", "a") as f:
        for token, value in top_tokens_with_values:
            f.write(f"{token}: {value:.8f}\n")
    return

# Save trained parameters
with open(f"trained_vanilla_ohsumed_8_negs.pkl", "wb") as fp:
    pickle.dump(p, fp)

wandb.finish()