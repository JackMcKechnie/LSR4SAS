import json
import ir_datasets
import pandas as pd
import pyterrier as pt
pt.init()
from pyterrier.measures import *
from pyterrier_t5 import MonoT5ReRanker
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AdamW
from random import Random
import itertools
import wandb
from transformers import BitsAndBytesConfig
import argparse
import os


parser = argparse.ArgumentParser(description='monot5 trainer')
parser.add_argument('--model', help='Description for foo argument', required=True)
args = parser.parse_args()

# Initialize Weights & Biases
wandb.login(key="89144673f34dfdcbf35383f8c9040f5e7eb2fcb1")
wandb.init(project="knowledge_distillation", name=f"{args.model}_sans1")

BATCH_SIZE = 8

import torch
torch.manual_seed(0)

_logger = ir_datasets.log.easy()

OUTPUTS = ['true', 'false']
training_docs = pd.read_pickle("/nfs/primary/sas_reranker/d2qmm_ohsumed_training_data.pkl")

def get_best_query(row):
    scores = row['querygen_score']
    queries = row['querygen'].split('\n') if pd.notna(row['querygen']) else []
    
    if isinstance(scores, list) and scores:  # Ensure scores is a non-empty list
        return queries[np.argmax(scores)] if len(queries) == len(scores) else None
    return "error"

training_docs['best_query'] = training_docs.apply(get_best_query, axis=1)
non_sensitive_docs = training_docs[training_docs["sensitivity"] == 0]
sensitive_docs = training_docs[training_docs["sensitivity"] == 1]

def iter_train_samples():
    while True:
        for _, non_sensitive_doc in non_sensitive_docs.iterrows():
            yield 'Query: ' + non_sensitive_doc.best_query + ' Document: ' + non_sensitive_doc.text + ' Relevant:', OUTPUTS[0]
            sensitive_doc = sensitive_docs.sample(n=1).iloc[0]
            yield 'Query: ' + sensitive_doc.best_query + ' Document: ' + sensitive_doc.text + ' Relevant:', OUTPUTS[1]

train_iter = _logger.pbar(iter_train_samples(), desc='total train samples')

if args.model == "t5-base":
    model = T5ForConditionalGeneration.from_pretrained("t5-base").cuda()
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
elif args.model == "t5-large":
    model = T5ForConditionalGeneration.from_pretrained("t5-large").cuda()
    tokenizer = T5Tokenizer.from_pretrained("t5-large")
elif args.model == "t5-3b":
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = T5ForConditionalGeneration.from_pretrained(
        "t5-3b", quantization_config=quantization_config
    )
    tokenizer = T5Tokenizer.from_pretrained("t5-3b")
else:
    import sys
    sys.exit(1)


optimizer = AdamW(model.parameters(), lr=5e-5)

reranker = MonoT5ReRanker(verbose=False, batch_size=BATCH_SIZE)
reranker.REL = tokenizer.encode(OUTPUTS[0])[0]
reranker.NREL = tokenizer.encode(OUTPUTS[1])[0]

def build_validation_data():
    result = []
    dataset = ir_datasets.load('msmarco-passage/trec-dl-2019/judged')
    docs = dataset.docs_store()
    queries = {q.query_id: q.text for q in dataset.queries_iter()}
    for qrel in _logger.pbar(ir_datasets.load('msmarco-passage/trec-dl-2019/judged').scoreddocs, desc='dev data'):
        if qrel.query_id in queries:
            result.append([qrel.query_id, queries[qrel.query_id], qrel.doc_id, docs.get(qrel.doc_id).text])
    return pd.DataFrame(result, columns=['qid', 'query', 'docno', 'text'])

valid_data = build_validation_data()
valid_qrels = pt.get_dataset('irds:msmarco-passage/trec-dl-2019/judged').get_qrels()

epoch = 0

max_ndcg = 0.

ohsumed_docs = pd.read_pickle("/nfs/primary/sas_reranker/ohsumed_docs_w_t5base_sensitivity.pkl")
queries = pd.read_pickle("/nfs/primary/graph_adaptive_reranking/queries.pkl")
qrels = pd.read_pickle("/nfs/primary/graph_adaptive_reranking/qrels.pkl")

def _sens_docs(qrels, run):
    if "sensitivity" in run.columns:
        run = run.drop(columns = ["sensitivity"])
    merged = pd.merge(run, ohsumed_docs, left_on = "doc_id", right_on = "docno")
    return merged.sensitivity.sum()
    
import ir_measures
sens_docs = ir_measures.define_byquery(
    _sens_docs, 
    name="sens_docs")


def get_text(run):
    l = len(run)
    run = pd.merge(run, ohsumed_docs, on = "docno")
    assert len(run) == l
    return run
    
while True:
    with _logger.pbar_raw(desc=f'train {epoch}', total=16384 // BATCH_SIZE) as pbar:
        model.train()
        total_loss = 0
        count = 0
        for _ in range(16384 // BATCH_SIZE):
        # for _ in range(10):
            inp, out = [], []
            for i in range(BATCH_SIZE):
                i, o = next(train_iter)
                inp.append(i)
                out.append(o)
            if args.model != "t5-3b":
                inp_ids = tokenizer(inp, return_tensors='pt', padding=True, max_length=512, truncation=True).input_ids.cuda()
                out_ids = tokenizer(out, return_tensors='pt', padding=True, max_length=512, truncation=True).input_ids.cuda()
            else:
                inp_ids = tokenizer(inp, return_tensors='pt', padding=True, max_length=512, truncation=True).input_ids
                out_ids = tokenizer(out, return_tensors='pt', padding=True, max_length=512, truncation=True).input_ids       
            loss = model(input_ids=inp_ids, labels=out_ids).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss = loss.item()
            count += 1
            pbar.update(1)
            pbar.set_postfix({'loss': total_loss/count})
            wandb.log({"train_loss": total_loss / count})
            inp_ids = inp_ids.to("cpu")
            out_ids = out_ids.to("cpu")
            torch.cuda.empty_cache()
            if count >= 200000:
                final_eval()
        wandb.log({"epoch": epoch})
    with _logger.duration(f'valid {epoch}'):
        reranker.model = model
        reranker.verbose = True
        res = reranker(valid_data)
        reranker.verbose = False
        metrics = {'epoch': epoch, 'loss': total_loss / count}
        metrics.update(pt.Utils.evaluate(res, valid_qrels, [nDCG, RR(rel=2)]))
        _logger.info(metrics)
        wandb.log(metrics)
        
        eval_results = pt.Experiment(
            [
            pt.io.read_results("/nfs/primary/standard_run_files/ohsumed/bm25_terrier_1000.run") >>
            pt.apply.generic(get_text) >>
            reranker
            ]
            ,
            queries,
            qrels,
            eval_metrics=[nDCG@10, sens_docs@10],
            names = ["BM25 >> monoT5"]
        )
        
        wandb.log({
            "ohsumed_ndcg@10" : eval_results.iloc[0]["nDCG@10"],
            "ohsumed_sens_docs@10" : eval_results.iloc[0]["sens_docs@10"]
        })
        
        with open('log.jsonl', 'at') as f:
            f.write(json.dumps(metrics) + '\n')
            f.write(json.dumps({
            "ohsumed_ndcg@10": eval_results.iloc[0]["nDCG@10"],
            "ohsumed_sens_docs@10": eval_results.iloc[0]["sens_docs@10"]
            }) + '\n')
        
        if metrics['nDCG'] > max_ndcg:
            _logger.info('new best nDCG')
            model.save_pretrained(f'./mymodel-best-{epoch}')
            max_ndcg = metrics['nDCG']
        epoch += 1