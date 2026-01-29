#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer
from sklearn.metrics import precision_score, accuracy_score, f1_score, balanced_accuracy_score, recall_score

import pickle
import os
from tqdm import tqdm

# In[2]:

def tokenize_and_save(texts, tokenizer, save_path):
    """Tokenize texts, convert to frequency vectors, and save as a .npy file."""
    print("Tokenizing texts...")
    vocab_size = len(tokenizer.vocab)
    frequency_matrix = np.zeros((len(texts), vocab_size), dtype=int)

    for i, text in tqdm(enumerate(texts), total = len(texts), desc = "Tokenizing"):
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        frequency_matrix[i], _ = np.histogram(token_ids, bins=np.arange(vocab_size + 1))

    print(f"Saving tokenized representations to {save_path}...")
    np.save(save_path, frequency_matrix)
    print("Tokenized representations saved!")

def load_tokenized_data(load_path):
    """Load tokenized representations from a .npy file."""
    if os.path.exists(load_path):
        print(f"Loading tokenized data from {load_path}...")
        return np.load(load_path)
    else:
        raise FileNotFoundError(f"Tokenized data not found at {load_path}")


def count_tokens(text):
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # Initialize a frequency vector (size of BERT vocabulary)
    vocab_size = len(tokenizer.vocab)
    frequency_vector = np.zeros(vocab_size, dtype=int)
    
    # Use numpy's histogram to count occurrences of each token ID
    frequency_vector, _ = np.histogram(token_ids, bins=np.arange(vocab_size + 1))
    return frequency_vector

def get_coefficients(texts, labels, test_docs, tokenizer, train_save_path, test_save_path):
    # Load or tokenize training data
    # print("Loading")
    # if os.path.exists(train_save_path):
    #     X_vectorized = load_tokenized_data(train_save_path)
    # print("Loaded!")

    # frequency_matrix = np.array([count_tokens(text) for text in texts])
    print("Tokenizing")
    X_vectorized = np.array([count_tokens(text) for text in texts])
    print("Tokenized!")
    
    # Train SVM
    print("Training...")
    # svm = SVC(kernel='linear', random_state=42, verbose=True, max_iter=100)
    svm = LinearSVC()
    svm.fit(X_vectorized, labels)
    coefficients = svm.coef_.flatten()
    print("Training complete!")

    print("Loading...")
    # Load or tokenize test data
    if os.path.exists(test_save_path):
        y_vectorized = load_tokenized_data(test_save_path)
    print("Loaded!")

    print("Predicting")
    y_pred = svm.predict(y_vectorized)
    y_true = test_docs.sensitivity.tolist()
    print("Predicted!")
    
    # Compute metrics
    precision = precision_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    bac = balanced_accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    print(f"Precision: {precision}")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Recall: {recall}")
    print(f"Balanced Accuracy: {bac}")


    test_docs["svm_labels"] = y_pred
    test_docs.to_csv("ohsumed_w_svm_sensitivity.csv", index = False)

    return coefficients
    
# In[3]:

print("Initialising...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
training_docs = pd.read_pickle("/nfs/primary/sas_reranker/d2qmm_ohsumed_training_data.pkl")
test_docs = pd.read_pickle("/nfs/primary/sas_reranker/ohsumed_docs_w_t5base_sensitivity.pkl")
print("Initialised!")

df_0 = training_docs[training_docs['sensitivity'] == 0].sample(n=50000, random_state=42, replace = True)  # Select 50 rows with label 0
df_1 = training_docs[training_docs['sensitivity'] == 1].sample(n=50000, random_state=42, replace = True)  # Select 50 rows with label 1

# Combine the two samples
df_sampled = pd.concat([df_0, df_1]).reset_index(drop=True)


train_save_path = "training_docs_freq_vec.npy"
test_save_path = "test_docs_freq_vec.npy"

coefficients = get_coefficients(df_sampled.text.tolist(), df_sampled.sensitivity.tolist(), test_docs, tokenizer, train_save_path, test_save_path)
tokens = tokenizer.convert_ids_to_tokens(np.arange(len(tokenizer.vocab)))

# Map coefficients to their corresponding tokens
feature_importance = pd.DataFrame({
    'Token': tokens,
    'Coefficient': coefficients
})

feature_importance_sorted = feature_importance.sort_values(by='Coefficient', ascending=False)

print(feature_importance_sorted.head(20))

feature_importance_sorted.to_csv("./feature_importance.csv", index = False)
with open("./svm_coefficients_v4.pkl", "wb") as f:
    pickle.dump(coefficients, f)