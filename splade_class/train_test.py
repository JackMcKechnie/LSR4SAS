from collections import Counter
import torch
import pandas as pd
from typing import List
import pyterrier as pt
pt.init()
from pyt_splade import Splade
import pickle
from tqdm import tqdm

class TokenClassifier(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.p = torch.nn.Parameter(torch.randn(vocab_size))
    
    def forward(self, x):
        return torch.matmul(x, torch.sigmoid(self.p))

def train(inputs: List[str], labels: List[bool], batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading...")
    with open("./encoded_training_docs_cache.pkl", "rb") as fp:
        encoded_inputs = pickle.load(fp)
    with open("./training_labels_cache.pkl", "rb") as fp:
        labels = pickle.load(fp)
    with open("./vocab.pkl", "rb") as fp:
        vocab = pickle.load(fp)
    with open("./inv_vocab.pkl", "rb") as fp:
        inv_vocab = pickle.load(fp)
    with open("./training_docs_dense.pkl", "rb") as fp:
        dense_docs = pickle.load(fp)
    print("Loaded!")
        
    labels = torch.tensor(labels, dtype=torch.float32)
    
    # Create model instance
    model = TokenClassifier(len(vocab)).to(device)
    
    # Get dimensions from pre-loaded dense docs
    n_samples = len(dense_docs)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    # Create optimizer with model parameters
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    
    pbar = tqdm(range(50), desc='Training')

    for epoch in pbar:
        
        total_loss = 0
        total_data_loss = 0
        total_reg_loss = 0
        
        # Process each batch
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)
            
            # Use pre-loaded dense docs
            batch_dense = dense_docs[start_idx:end_idx].to(device)
            batch_labels = labels[start_idx:end_idx].to(device)
            
            optim.zero_grad()
            
            # Forward pass
            scores = model(batch_dense)
            data_loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, batch_labels)
            reg_loss = torch.sigmoid(model.p).sum()
            loss = data_loss + reg_loss
            
            # Backward pass and optimization
            loss.backward()
            optim.step()
            
            # Accumulate losses
            total_loss += loss.item() * (end_idx - start_idx)
            total_data_loss += data_loss.item() * (end_idx - start_idx)
            total_reg_loss += reg_loss.item() * (end_idx - start_idx)
            
            
            pbar.set_postfix({
            'loss': f'{data_loss:.4f}'
            })
            
            # Free up GPU memory
            del batch_dense
            del batch_labels
            torch.cuda.empty_cache()
        
        # Calculate average losses for the epoch
        avg_loss = total_loss / n_samples
        avg_data_loss = total_data_loss / n_samples
        avg_reg_loss = total_reg_loss / n_samples
        print(data_loss.item(), reg_loss.item(), loss.item())
        
    
    pbar.close()
    
    # Get final parameters
    p = torch.sigmoid(model.p).detach().cpu().numpy()
    for x, y in Counter(dict(zip(inv_vocab, p))).most_common():
        print(x, y)

    with open("trained_parameter.pkl", "wb") as fp:   #Pickling
        pickle.dump(p, fp)

# Example usage
train([
    'some sensitive document',
    'a document',
    'a third sensitive document',
    "a completely unrelated document"
], [1, 0, 1, 0])