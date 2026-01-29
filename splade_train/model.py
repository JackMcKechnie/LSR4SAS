import torch
from transformers import AutoModelForMaskedLM, AutoModel
from transformers.trainer import  logger
from transformers import PreTrainedModel
import os
from typing import Dict, List

class SPLADEModel(torch.nn.Module):
    
    @staticmethod
    def splade_max(output, attention_mask):
        output = output.logits
        relu = torch.nn.ReLU(inplace=False)
        values, _ = torch.max(torch.log(1 + relu(output)) * attention_mask.unsqueeze(-1), dim=1)
        return values
    
    def __init__(self, model_type_or_dir, tokenizer=None, shared_weights=True, n_negatives=-1, splade_doc=False, model_q=None, **kwargs):
        super().__init__()        
        self.shared_weights = shared_weights       
        self.doc_encoder = AutoModelForMaskedLM.from_pretrained(model_type_or_dir)
        self.output_dim = self.doc_encoder.config.vocab_size
        self.n_negatives = n_negatives
        self.splade_doc = splade_doc
        self.doc_activation = self.splade_max
        self.query_activation = self.splade_max if not self.splade_doc else self.passthrough
        self.query_encoder = self.doc_encoder

    def forward(self, **tokens):
        representations = self.doc_activation(self.doc_encoder(**tokens),attention_mask=tokens["attention_mask"]) #TODO This should separate docs and queries and use their separate activations, for now is not a problem because they will always be the same if we are here.
        output = representations.view(-1,self.n_negatives+2,representations.size(1))
        queries_result = output[:,:1,:]
        docs_result = output[:,1:,:]
        return queries_result, docs_result
    
    def save(self, output_dir, tokenizer):
        model_dict = self.doc_encoder.state_dict()
        torch.save(model_dict, os.path.join(output_dir,  "pytorch_model.bin"))
        self.doc_encoder.config.save_pretrained(output_dir)
    
        if tokenizer:
            tokenizer.save_pretrained(output_dir)