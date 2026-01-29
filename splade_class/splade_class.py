import pyterrier as pt
from pyt_splade import Splade
import pandas as pd
from ir_measures import *
import argparse
from typing import Union, List, Literal, Dict
import torch
import numpy as np
import pickle
from pyterrier_pisa import PisaIndex
from tqdm import tqdm
import pyterrier_dr

if not pt.started():
    pt.init()

parser = argparse.ArgumentParser(description='Evaluate SPLADE model on OHSUMED')
parser.add_argument('--index_loc', type=str, help='Path where the index should be stored')
parser.add_argument('--run_output_path', type=str, help='Path to output the run file to')
parser.add_argument('--experiment_name', type=str, help='Name of the experiment')
parser.add_argument("--parameter", type = str, help = "Location to the pickle of the parameter")
parser.add_argument("--combination_method", type = str, help = "How to combine the parameter and the doc_vecs")
args = parser.parse_args()


class SpladeClass(Splade):
    def __init__(
        self,
        classification_vector,
        combination_method = "add",
    ):
        self.combination_method = combination_method
        self.classification_vector = pickle.load(open(classification_vector, "rb"))
        self.sparse = False
        super().__init__()

    def encode(
        self,
        texts: List[str],
        rep: Literal['d', 'q'] = 'd',
        format: Literal['dict', 'np', 'torch'] ='np',
        scale: float = 1.,
    ) -> Union[List[Dict[str, float]], List[np.ndarray], torch.Tensor]:
        """Encodes a batch of texts into their SPLADE representations.

        Args:
            texts: the list of texts to encode
            rep: 'q' for query, 'd' for document
            format: 'dict' for a dict of term frequencies, 'np' for a list of numpy arrays, 'torch' for a torch tensor
            scale: the scale to apply to the term frequencies
        """
        format = "np"
        self.combination_method = args.combination_method
        rtr = []
        with torch.no_grad():
            reps = self.model(**{rep + '_kwargs': self.tokenizer(
                texts,
                add_special_tokens=True,
                padding="longest",  # pad to max sequence length in batch
                truncation="longest_first",  # truncates to max model length,
                max_length=self.max_length,
                return_attention_mask=True,
                return_tensors="pt",
            ).to(self.device)})[rep + '_rep']
            reps = reps * scale
        if format == 'dict':
            reps = reps.cpu()
            for i in range(reps.shape[0]):
                # get the number of non-zero dimensions in the rep:
                col = torch.nonzero(reps[i]).squeeze(1).tolist()
                # now let's create the bow representation as a dictionary
                weights = reps[i, col].cpu().tolist()
                # if document cast to int to make the weights ready for terrier indexing
                if rep == "d":
                    weights = list(map(int, weights))
                sorted_weights = sorted(zip(col, weights), key=lambda x: (-x[1], x[0]))
                # create the dict removing the weights less than 1, i.e. 0, that are not helpful
                d = {self.reverse_voc[k]: v for k, v in sorted_weights if v > 0}
                rtr.append(d)
        elif format == 'np':
            reps = reps.cpu().numpy()
            for i in range(reps.shape[0]):
                if self.combination_method == "add":
                    classified = reps[i] + self.classification_vector
                    rtr.append(classified)
                if self.combination_method == "multiply":
                    classified = reps[i] * (1 - (self.classification_vector * 1000))
                    print(classified)
                    rtr.append(classified)
                if self.combination_method == "none":
                    rtr.append(reps[i])
        elif format == 'torch':
            rtr = reps
        return rtr

def ohsumed_iterator():
    for row in tqdm(docs.iterrows(), total = len(docs), desc = "Indexing"):
        row = row[1]
        yield {"docno" : row.docno, "text" : row.text}

def _sens_docs(qrels, run):
    if "sensitivity" in run.columns:
        run = run.drop(columns = ["sensitivity"])
    merged = pd.merge(run, docs, left_on = "doc_id", right_on = "docno")
    return merged.sensitivity.sum()
    
import ir_measures
sens_docs = ir_measures.define_byquery(
    _sens_docs, 
    name="sens_docs")

docs = pd.read_pickle("/mnt/primary/sas_reranker/ohsumed_docs_w_t5base_sensitivity.pkl")
queries = pd.read_pickle("/mnt/primary/graph_adaptive_reranking/queries.pkl")
qrels = pd.read_pickle("/mnt/primary/graph_adaptive_reranking/qrels.pkl")

splade = SpladeClass(args.parameter)
index = pyterrier_dr.FlexIndex(args.index_loc)
idx_pipeline = splade.doc_encoder(sparse = False) >> index
idx_pipeline.index(ohsumed_iterator())
retr_pipeline = splade.query_encoder(sparse = False) >> index.np_retriever()

res = retr_pipeline(queries)



print("Saving results...")
pt.io.write_results(res, args.run_output_path)
print("Results saved!")

retrieval_results = pt.Experiment(
    [res],
    queries,
    qrels,
    eval_metrics=[nDCG@10, sens_docs@10]
)

print("Retrieval results:")
print(retrieval_results)

with open("retrieval_test_out.txt", "a") as f:
    f.write(f"{args.experiment_name}\n")
    f.write(f"{str(retrieval_results.iloc[0]['nDCG@10'])}\n")
    f.write(f"{str(retrieval_results.iloc[0]['sens_docs@10'])}\n\n")