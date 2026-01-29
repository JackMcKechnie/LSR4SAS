#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyterrier as pt
import pandas as pd
import ir_datasets
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from data import MsMarcoTripletDataset, MsMarcoCollator, SPLADEArgs, OHSUMEDDataset
from model import SPLADEModel
from trainer import IRTrainer

ds = OHSUMEDDataset()

args = SPLADEArgs("./ohsumed_training_dir/")

args.tokenizer_name_or_path = "distilbert-base-uncased"
args.model_name_or_path = "distilbert-base-uncased" 
args.shared_weights = True
args.n_negatives = 1
args.splade_doc = False
args.model_q = None
args.dense = False
args.full_determinism = False, 
args.seed = 123
args.lr = 2e-5
args.seed = 123
args.gradient_accumulation_steps = 1
args.weight_decay = 0.01
args.validation_metrics = [ "MRR@10", "recall@100", "recall@200", "recall@500" ]
args.pretrained_no_yamlconfig = False
args.nb_iterations = 150000
args.per_device_train_batch_size = 32  # number of gpus needs to divide this
args.per_device_eval_batch_size = 32
args.index_retrieve_batch_size = 500
args.record_frequency = 10000
args.train_monitoring_freq = 500
args.warmup_steps = 6000
args.max_length = 256
args.fp16 = True
args.matching_type = "splade"
args.monitoring_ckpt = "MRR@10"  # or e.g. MRR@10
args.lexical_type = "none"
args.training_loss = "contrastive"
args.T_d = 0
args.T_q = 0
args.l0d = 5e-4
args.l0q = 5e-4
args.top_d = -1
args.top_q = -1
args.evaluation_strategy = "step"
args.do_eval = True
args._flops = 1 # fix
args.num_train_epochs = 100

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
collator = MsMarcoCollator(tokenizer = tokenizer)

model = SPLADEModel(args.model_name_or_path,shared_weights=args.shared_weights,n_negatives=args.n_negatives, tokenizer=tokenizer, splade_doc=args.splade_doc, model_q=args.model_q)

trainer = IRTrainer(model = model,                         
                    args = args,                  
                    train_dataset = ds,
                    data_collator = collator.torch_call,
                    tokenizer = tokenizer,
                    shared_weights = args.shared_weights,  
                    splade_doc = args.splade_doc,          
                    n_negatives = args.n_negatives,         
                    dense = args.dense,
                   )
trainer.train()
model.save("./ohsumed_splade_model_100_epoch", tokenizer = tokenizer)