#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer
from sklearn.metrics import precision_score, accuracy_score, f1_score, balanced_accuracy_score
import pickle
from tqdm import tqdm

# In[2]:


def count_tokens(text):
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # Initialize a frequency vector (size of BERT vocabulary)
    vocab_size = len(tokenizer.vocab)
    frequency_vector = np.zeros(vocab_size, dtype=int)
    
    # Use numpy's histogram to count occurrences of each token ID
    frequency_vector, _ = np.histogram(token_ids, bins=np.arange(vocab_size + 1))
    return frequency_vector

# def get_coefficients(texts, labels):
#     print("Tokenising...")
#     frequency_matrix = np.array([count_tokens(text) for text in texts])
#     X_vectorized = np.array([count_tokens(text) for text in texts])
#     print("Tokenised!")
#     # Train an SVM with a linear kernel

#     print("Training...")
#     svm = SVC(kernel='linear', random_state=42, verbose = True, max_iter = 100)
#     svm.fit(X_vectorized, labels)
#     coefficients = svm.coef_.flatten()
#     print("Trained!")
#     # Test the classifier
#     print("Tokenizing...")
#     y_vectorised = np.array([count_tokens(text) for text in test_docs.text.tolist()])
#     print("Tokenized!")

#     print("Predicting...")
#     y_pred = svm.predict(y_vectorised)
#     print("Predicted!")
    
#     y_true = test_docs.sensitivity.tolist()
    
#     precision = precision_score(y_true, y_pred)
#     accuracy = accuracy_score(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred) 
#     bac = balanced_accuracy_score(y_true, y_pred)
    
#     # Print the metrics
#     print(f"Precision: {precision}")
#     print(f"Accuracy: {accuracy}")
#     print(f"F1 Score: {f1}")
#     print(f"Balanced Accuracy: {bac}")
#     return coefficients

def get_coefficients(texts, labels):
    print(test_docs.docno.tolist())
    print("Tokenising...")
    X_vectorized = np.array([count_tokens(text) for text in texts])
    print("Tokenised!")

    # Train an SVM with probability estimation
    print("Training...")
    svm = SVC(kernel='linear', probability=True, random_state=42, verbose=True, max_iter=100)
    svm.fit(X_vectorized, labels)
    coefficients = svm.coef_.flatten()
    print("Trained!")

    # Test the classifier
    print("Tokenizing test set...")
    y_vectorized = np.array([count_tokens(text) for text in test_docs.text.tolist()])
    print("Tokenized!")

    print("Predicting...")
    y_pred = []
    y_proba = []

    for vec in tqdm(y_vectorized, desc="Inference Progress"):  # Progress bar for prediction
        pred = svm.predict(vec.reshape(1, -1))[0]
        prob = svm.predict_proba(vec.reshape(1, -1))[0, 1]
        y_pred.append(pred)
        y_proba.append(prob)

    print("Predicted!")

    y_true = test_docs.sensitivity.tolist()

    precision = precision_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    bac = balanced_accuracy_score(y_true, y_pred)

    # Print metrics
    print(f"Precision: {precision}")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Balanced Accuracy: {bac}")

    # Save predictions with probabilities
    results_df = pd.DataFrame({
        'docno': test_docs.docno.tolist(),
        'predicted_label': y_pred,
        'probability': y_proba
    })
    results_df.to_csv("./predictions_with_probabilities.csv", index=False)

    return coefficients



# In[3]:

print("Initialising...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
training_docs = pd.read_pickle("/nfs/primary/sas_reranker/d2qmm_ohsumed_training_data.pkl")
test_docs = pd.read_pickle("/nfs/primary/sas_reranker/ohsumed_docs_w_t5base_sensitivity.pkl")
print("Initialised!")

df_0 = training_docs[training_docs['sensitivity'] == 0].sample(n=50, random_state=42)  # Select 50 rows with label 0
df_1 = training_docs[training_docs['sensitivity'] == 1].sample(n=50, random_state=42)  # Select 50 rows with label 1

# Combine the two samples
df_sampled = pd.concat([df_0, df_1]).reset_index(drop=True)

coefficients = get_coefficients(df_sampled.text.tolist(), df_sampled.sensitivity.tolist())
tokens = tokenizer.convert_ids_to_tokens(np.arange(len(tokenizer.vocab)))

# Map coefficients to their corresponding tokens
feature_importance = pd.DataFrame({
    'Token': tokens,
    'Coefficient': coefficients
})

feature_importance_sorted = feature_importance.sort_values(by='Coefficient', ascending=False)

print(feature_importance_sorted.head(20))

# feature_importance_sorted.to_csv("./feature_importance.csv", index = False)
# with open("./svm_coefficients.pkl", "wb") as f:
#     pickle.dump(coefficients, f)