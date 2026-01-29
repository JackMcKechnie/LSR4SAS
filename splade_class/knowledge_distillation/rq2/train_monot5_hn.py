import numpy as np
from torch.nn import functional as F
import wandb
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW
import json
import ir_datasets
import pandas as pd
import pyterrier as pt
pt.init()
from pyterrier_pisa import PisaIndex
from pyterrier_t5 import MonoT5ReRanker
from ir_measures import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--negs', type=int, default=1)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

rng = np.random.RandomState(args.seed)

# Initialize Weights & Biases
wandb.login(key="89144673f34dfdcbf35383f8c9040f5e7eb2fcb1")
wandb.init(project="knowledge_distillation", name=f"hn_splade_sans1")

import torch
torch.manual_seed(0)

_logger = ir_datasets.log.easy()

bm25 = PisaIndex.from_dataset('msmarco_passage').bm25()

OUTPUTS = ['true', 'false']

file_path = "hn_splade_sans1_all_sensitive_32_negs.jsonl"
OUTPUTS = ["true", "false"]

def iter_jsonl_with_index(filepath):
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            yield 'Query: ' + data["query"] + ' Document: ' + data["docno_a"] + ' Relevant:', OUTPUTS[0]
            for neg in data["docno_b"]:
                yield 'Query: ' + data["query"] + ' Document: ' + neg + ' Relevant:', OUTPUTS[1]
 

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
    
train_iter = _logger.pbar(iter_jsonl_with_index(file_path), desc='total train samples')

model = T5ForConditionalGeneration.from_pretrained("/nfs/primary/sas_cross_encoder/trained_models/ohsumed_1bm25_proportionate_neg_synth_data_monot5_sas_random_negs_best-31").cuda()
# model = T5ForConditionalGeneration.from_pretrained("t5-base").cuda()
tokenizer = T5Tokenizer.from_pretrained("t5-base")
optimizer = AdamW(model.parameters(), lr=5e-5)

OUT_IDS = [tokenizer(t)['input_ids'][0] for t in OUTPUTS]

# model.train()
# for epoch in range(100):
#     total_loss = 0
#     correct = 0
#     count = 0
#     for _ in range(1024*16):
#       inp, out = [], []
#       for i in range(args.negs+1):
#         i, o = next(train_iter)
#         inp.append(i)
#         out.append(o)
#       inp_ids = tokenizer(inp, return_tensors='pt', padding=True).input_ids.cuda()
#       out_ids = tokenizer(out, return_tensors='pt', padding=True).input_ids.cuda()
#       model_out = model(input_ids=inp_ids, decoder_input_ids=torch.full_like(inp_ids[:,:1], model.config.decoder_start_token_id)).logits
#       logprobs = F.log_softmax(model_out[:, :, OUT_IDS], dim=2)[:, 0, :]
#       loss = -1 * (logprobs[0, 0] + logprobs[1:, 1].sum())
#       loss.backward()
#       optimizer.step()
#       optimizer.zero_grad()
#       total_loss = loss.item()
#       count += 1
#       c = (logprobs[0] > logprobs[1:]).sum() / args.negs
#       correct += c
#       total_loss += loss.item()
#       count += 1
#       wandb.log({'loss': loss.item(), "acc": c})
#     model.save_pretrained(f'data/t5-base-{args.negs}-{args.seed}-{epoch}')

epoch = 0
BATCH_SIZE = 1
reranker = MonoT5ReRanker(verbose=False, batch_size=BATCH_SIZE)
max_ndcg = 0.0 

while True:
  with _logger.pbar_raw(desc=f'train {epoch}', total=16384 // BATCH_SIZE) as pbar:
    model.train()
    total_loss = 0
    count = 0
    correct = 0
    for _ in range(16384 // BATCH_SIZE):
      inp, out = [], []
      for i in range(args.negs+1):
        i, o = next(train_iter)
        inp.append(i)
        out.append(o)
      inp_ids = tokenizer(inp, return_tensors='pt', padding=True, truncation=True, max_length=512).input_ids.cuda()
      out_ids = tokenizer(out, return_tensors='pt', padding=True, truncation=True, max_length=512).input_ids.cuda()
      model_out = model(input_ids=inp_ids, decoder_input_ids=torch.full_like(inp_ids[:,:1], model.config.decoder_start_token_id)).logits
      logprobs = F.log_softmax(model_out[:, :, OUT_IDS], dim=2)[:, 0, :]
      loss = -1 * (logprobs[0, 0] + logprobs[1:, 1].sum())
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      total_loss = loss.item()
      count += 1
      c = (logprobs[0] > logprobs[1:]).sum() / args.negs
      correct += c
      total_loss += loss.item()
      count += 1        
      pbar.update(1)
      pbar.set_postfix({'loss': total_loss/count})
      wandb.log({"loss" : loss})
  with _logger.duration(f'valid {epoch}'):
    reranker.model = model
    reranker.verbose = True
    res = reranker(valid_data)
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
    reranker.verbose = False
    with open('log.jsonl', 'at') as f:
        f.write(json.dumps(metrics) + '\n')
        f.write(json.dumps({
            "ohsumed_ndcg@10": eval_results.iloc[0]["nDCG@10"],
            "ohsumed_sens_docs@10": eval_results.iloc[0]["sens_docs@10"]
        }) + '\n')
        
    if metrics['nDCG'] > max_ndcg:
      _logger.info('new best nDCG')
      model.save_pretrained(f'./hn_splade_sans1_retrain_16_negs-{epoch}')
      max_ndcg = metrics['nDCG']
  epoch += 1