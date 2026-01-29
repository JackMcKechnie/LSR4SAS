import pyterrier as pt
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
import pyterrier_dr
from ir_measures import *
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, load_npz, vstack
import json
from pyterrier.model import add_ranks
import os

class PyTerrierSPLADE(pt.Transformer):

    def __init__(self, model_name, index_location, index_name, verbose = True, batch_size = 4, num_results = 1000):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces = True)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.verbose = verbose
        self.batch_size = batch_size
        self.index_location = index_location
        self.num_results = num_results
        self.index_name = index_name

    def doc_encode(self, inp):
        texts = inp.text.tolist()
        it = range(0, len(texts), self.batch_size)
        all_vecs = []

        if self.verbose:
            it = pt.tqdm(it, desc='Processing Batches', unit='batches')

        for start_idx in it:
            rng = slice(start_idx, start_idx + self.batch_size)
            batch_texts = texts[rng]

            with torch.no_grad():
                tokens = self.tokenizer(batch_texts, padding=True, truncation=True, return_attention_mask = True, return_tensors='pt')
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                output = self.model(**tokens)
                vecs = torch.max(torch.log(1 + torch.relu(output.logits)) * tokens["attention_mask"].unsqueeze(-1), dim=1)[0].squeeze().detach().cpu().numpy()
                tokens = {k: v.to("cpu") for k, v in tokens.items()}
                if len(vecs.shape) == 1: vecs = vecs[np.newaxis, :] # Handle if there is a single document in the last batch
                all_vecs.extend(list(vecs))
            
        inp["doc_vec"] = all_vecs
        return inp

    def query_encode(self, inp):
        text = inp["query"].tolist()
        all_vecs = []
        
        with torch.no_grad():
            tokens = self.tokenizer(text, padding=True, truncation=True, return_attention_mask = True, return_tensors='pt')
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            output = self.model(**tokens)
            vecs = torch.max(torch.log(1 + torch.relu(output.logits)) * tokens["attention_mask"].unsqueeze(-1), dim=1)[0].squeeze().detach().cpu().numpy()
            if len(vecs.shape) == 1: vecs = vecs[np.newaxis, :] # Handle if there is a single document in the last batch
            all_vecs.extend(list(vecs))
            
        inp["query_vec"] = all_vecs
        return inp

    def index(self, inp, batch_size=1000):
        inp = pt.tqdm(inp, desc='indexing', unit='dvec', disable=False)  # Leave tqdm as an iterator
        docno_dict = {}
        batch_counter = 0
        batch_docs = []

        if not os.path.isdir(self.index_location):
            os.mkdir(self.index_location)
        
        for i, doc in enumerate(inp):
            batch_docs.append(doc)
    
            # When the batch size is reached, process and save the batch
            if len(batch_docs) == batch_size:
                self._process_and_save_batch(batch_docs, batch_counter, docno_dict)
                batch_docs = []  # Reset the batch
                batch_counter += 1
    
        # Process any remaining documents in the last batch
        if batch_docs:
            self._process_and_save_batch(batch_docs, batch_counter, docno_dict)
    
        # Save the combined docno dictionary
        with open(f"{self.index_location}/{self.index_name}.json", 'w') as f:
            json.dump(docno_dict, f)
    
        # Combine all batch files into one index file
        self._combine_batches(batch_counter + 1)  # +1 because batch_counter starts at 0

    def _process_and_save_batch(self, batch_docs, batch_counter, docno_dict):
        # Convert the batch to DataFrame
        batch_df = pd.DataFrame(batch_docs)
        
        # Encode document vectors for the batch
        batch_df = self.doc_encode(batch_df)
        doc_vecs = np.array(batch_df['doc_vec'].tolist())
        sparse_matrix = csr_matrix(doc_vecs)
    
        # Save the sparse matrix for this batch
        batch_filename = f"{self.index_location}/{self.index_name}_batch_{batch_counter}.npz"
        save_npz(batch_filename, sparse_matrix)
    
        # Update docno_dict with the new documents
        for i, doc in enumerate(batch_docs):
            docno_dict[len(docno_dict)] = doc["docno"]

    def _combine_batches(self, num_batches):
        combined_matrix = None
        
        for batch_counter in range(num_batches):
            print(f"Processing batch : {batch_counter}")
            batch_filename = f"{self.index_location}/{self.index_name}_batch_{batch_counter}.npz"
            sparse_matrix = load_npz(batch_filename)
            
            if combined_matrix is None:
                combined_matrix = sparse_matrix
            else:
                combined_matrix = vstack([combined_matrix, sparse_matrix], format='csr')
            
            # Remove the batch file to save disk space
            os.remove(batch_filename)
    
        # Save the final combined index
        save_npz(f"{self.index_location}/{self.index_name}.npz", combined_matrix)
        
    def load_index(self):
        self.index = load_npz(f"{self.index_location}/{self.index_name}.npz")
        self.lookup = json.load(open(f"{self.index_location}/{self.index_name}.json"))

    def retrieve(self, inp):
        df_rows = []
        for query in inp.iterrows(): 
            query_vector = csr_matrix(query[1]["query_vec"]).T
            similarity_scores = self.index.dot(query_vector).toarray().ravel()
            top_indices = np.argsort(similarity_scores)[-self.num_results:][::-1]
            scores = similarity_scores[top_indices]

            for idx in top_indices:
                    docno = self.lookup[str(idx)]
                    score = similarity_scores[idx]
                    df_rows.append({
                        "qid": query[1]["qid"],
                        "query": query[1]["query"],
                        "query_vec": query[1]["query_vec"],
                        "docno" : docno,
                        "score" : score
                    })

        return add_ranks(pd.DataFrame(df_rows))
            
    def transform(self, inp):
        if "text" in inp.columns and "doc_vec" not in inp.columns:
            inp = self.doc_encode(inp)
            
        if "query" in inp.columns and "query_vec" not in inp.columns:
            inp = self.query_encode(inp)
            self.load_index()
            inp = self.retrieve(inp)
            
        return inp