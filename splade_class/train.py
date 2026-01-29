from collections import Counter
import torch
import pandas as pd
from typing import List
import pyterrier as pt
pt.init()
from pyt_splade import Splade

def train(inputs: List[str], labels: List[bool]):
    labels = torch.tensor(labels, dtype=torch.float32)
    splade = Splade().doc_encoder()
    
    encoded_inputs = list(splade(pd.DataFrame({
        'docno': list(range(len(inputs))),
        'text': inputs,
    }))['toks'])
    
    vocab = {}
    inv_vocab = []
    for doc in encoded_inputs:
        for token in doc.keys() - vocab:
            vocab[token] = len(vocab)
            inv_vocab.append(token)
            
    parameters = torch.nn.Parameter(torch.randn(len(vocab)))
    weight = torch.nn.Parameter(torch.tensor(1.0))  # Learnable weight

    docs_dense = torch.zeros(len(encoded_inputs), len(vocab))
    for i, doc in enumerate(encoded_inputs):
        for token, count in doc.items():
            docs_dense[i, vocab[token]] = count

    # Separate optimizers for parameters and weight
    optim_params = torch.optim.Adam([parameters], lr=0.01)
    optim_weight = torch.optim.Adam([weight], lr=0.01)

    for epoch in range(30000000):  # Reduce epochs for testing
        # Zero gradients
        optim_params.zero_grad()
        optim_weight.zero_grad()

        # Forward pass
        scores = torch.matmul(docs_dense, torch.sigmoid(parameters))
        data_loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels)
        reg_loss = torch.abs(parameters).sum()

        # Ensure weight is between 0 and 1
        weight_clamped = torch.clamp(weight, min = 1)
        loss = (data_loss * (1 - weight_clamped)) + (reg_loss * weight_clamped)

        # Backward pass and optimization
        loss.backward()
        optim_params.step()
        optim_weight.step()

        if epoch % 500 == 0:
            print(f'Epoch {epoch}, Data Loss: {data_loss.item():.4f}, Reg Loss: {reg_loss.item():.4f}, Loss: {loss.item():.4f}, Weight: {weight_clamped.item():.4f}')
    
    p = torch.sigmoid(parameters).detach().numpy()
    
    for x, y in Counter(dict(zip(inv_vocab, p))).most_common()[:10]:
        print(x, y)

    return p

# Sample Test
p = train([
    'this is a very sensitive document',
    'a document about shakespeare',
    'this text is also quite sensitive',
], [1, 0, 1])

