from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
import pandas as pd
import numpy as np

# Assuming you have train_df and test_df
# train_df = pd.DataFrame({'text': [...], 'sensitivity': [...]})
# test_df = pd.DataFrame({'text': [...], 'sensitivity': [...]})

def train_svm_and_get_top_words(train_df, test_df, top_n=100):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    clf = SVC(kernel='linear')
    
    # Create a pipeline
    model = make_pipeline(vectorizer, clf)
    
    # Train the model
    print("Training...")
    model.fit(train_df['text'], train_df['sensitivity'])
    print("Trained!")
    
    # Predictions
    y_pred = model.predict(test_df['text'])
    y_true = test_df['sensitivity']
    
    # Evaluation metrics
    accuracy = model.score(test_df['text'], y_true)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Balanced Accuracy: {balanced_acc:.4f}')
    
    # Extract feature importance
    feature_names = vectorizer.get_feature_names_out()
    # coef = np.abs(clf.coef_).sum(axis=0)  # Sum across classes if multi-class
    coef = np.abs(clf.coef_.copy()).sum(axis=0)  # Ensure a writable copy

    # Get top N words
    top_indices = np.argsort(coef)[-top_n:][::-1]
    top_words = [feature_names[i] for i in top_indices]
    
    return top_words

training_docs = pd.read_pickle("/nfs/primary/sas_reranker/d2qmm_ohsumed_training_data.pkl")
test_docs = pd.read_pickle("/nfs/primary/sas_reranker/ohsumed_docs_w_t5base_sensitivity.pkl")

top_words = train_svm_and_get_top_words(training_docs, test_docs)[0][0]
for word in top_words:
    print(word)