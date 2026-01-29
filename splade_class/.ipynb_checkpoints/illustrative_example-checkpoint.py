# from collections import Counter
# import torch
# import pandas as pd
# from typing import List
# import pyterrier as pt
# pt.init()
# from pyt_splade import Splade
# from transformers import BertTokenizer

# def train(inputs: List[str], labels: List[bool]):
#     labels = torch.tensor(labels, dtype=torch.float32)
#     splade = Splade().doc_encoder(sparse = False)
#     encoded_inputs = splade(pd.DataFrame({
#         'docno': list(range(len(inputs))),
#         'text': inputs,
#     }))
#     encoded_inputs = torch.tensor(encoded_inputs["doc_vec"].tolist(), dtype=torch.float)
#     print(encoded_inputs)
    
#     # parameters = torch.nn.Parameter(torch.randn(len(encoded_inputs[0])))
#     parameters = torch.nn.Parameter(torch.zeros(len(encoded_inputs[0])))
#     optim = torch.optim.Adam([parameters], lr=0.01)
#     for epoch in range(20000):
#         # Zero gradients at the start of each step
#         optim.zero_grad()

#         # Forward pass
#         scores = torch.matmul(encoded_inputs, torch.sigmoid(parameters))
#         data_loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels)
#         reg_loss = torch.sigmoid(parameters).abs().sum()
#         loss = data_loss + reg_loss

#         # Backward pass and optimization
#         loss.backward()
#         optim.step()

#         print(f'Epoch {epoch}, Data Loss: {data_loss.item():.4f} Reg Loss: {reg_loss.item():.4f} Loss: {loss.item():.4f}')
#     p = torch.sigmoid(parameters).detach().numpy()
#     return p

# def get_top_tokens(vector, tokenizer_name="bert-base-uncased", top_n=10):
#     tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

#     if not isinstance(vector, torch.Tensor):
#         vector = torch.tensor(vector)
        
#     top_indices = torch.topk(vector, top_n).indices.tolist()
#     top_tokens = [tokenizer.convert_ids_to_tokens(idx) for idx in top_indices]
#     return top_tokens

# p = train([
#     'this is a very sensitive document',
#     'a document about shakespeare',
#     'this text is also quite sensitive',
# ], [1, 0, 1])

# print(get_top_tokens(p))
# print(p.mean(), p.min(), p.max(), torch.count_nonzero(torch.tensor(p)))


from collections import Counter
import torch
import pandas as pd
from typing import List
import pyterrier as pt
pt.init()
from pyt_splade import Splade
from transformers import BertTokenizer

def train(inputs: List[str], labels: List[bool]):
    labels = torch.tensor(labels, dtype=torch.float32)
    splade = Splade().doc_encoder(sparse=False)
    encoded_inputs = splade(pd.DataFrame({
        'docno': list(range(len(inputs))),
        'text': inputs,
    }))
    encoded_inputs = torch.tensor(encoded_inputs["doc_vec"].tolist(), dtype=torch.float)

    # Initialize parameters
    parameters = torch.nn.Parameter(torch.zeros(len(encoded_inputs[0])))
    optim = torch.optim.Adam([parameters], lr=0.01)

    for epoch in range(20000):
        optim.zero_grad()

        # Forward pass: Do NOT apply sigmoid before matmul
        scores = torch.matmul(encoded_inputs, parameters)

        # Compute binary cross-entropy loss
        data_loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels)

        # Apply L1 regularization directly on parameters (NOT on sigmoid(parameters))
        reg_loss = torch.sigmoid(parameters).abs().sum() * 10  # Increase multiplier for stronger sparsity

        loss = data_loss + reg_loss

        # Backpropagation
        loss.backward()
        optim.step()

        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Data Loss: {data_loss.item():.4f}, Reg Loss: {reg_loss.item():.4f}, Loss: {loss.item():.4f}')
    
    # Extract final learned parameters and apply sigmoid
    p = torch.sigmoid(parameters).detach().numpy()
    return p

def get_top_tokens(vector, tokenizer_name="bert-base-uncased", top_n=10):
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
    for token, value in top_tokens_with_values:
        print(f"{token}: {value:.8f}")
    return

p = train([
    'this is a very sensitive document',
    'a document about shakespeare',
    'this text is also quite sensitive',
], [1, 0, 1])

get_top_tokens(p, top_n = 100)
print(p.mean(), p.min(), p.max(), torch.count_nonzero(torch.tensor(p)))
