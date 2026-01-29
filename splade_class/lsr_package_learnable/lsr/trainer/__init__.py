import torch
import os
import transformers
import logging
from lsr.models import DualSparseEncoder
from lsr.models import DualSparseEncoder
from collections import defaultdict
import wandb

logger = logging.getLogger(__name__)

LOSS_NAME = "loss.pt"

class HFTrainer(transformers.trainer.Trainer):
    """Customized Trainer from Huggingface's Trainer"""

    def __init__(self, *args, loss=None, **kwargs) -> None:
        super(HFTrainer, self).__init__(*args, **kwargs)
        self.loss = loss
        self.customed_log = defaultdict(lambda: 0.0)
        self.tokenizer = self.data_collator.tokenizer

    def evaluate(self, ignore_keys=None):
        from lsr.transformer import LSR
        from pyterrier_pisa import PisaIndex, PisaToksIndexer
        import pandas as pd
        from tqdm import tqdm
        import pyterrier as pt
        if not pt.started():
            pt.init()
        from ir_measures import nDCG
        docs = pd.read_pickle("/nfs/primary/sas_reranker/ohsumed_docs_w_t5base_sensitivity.pkl")
        queries = pd.read_pickle("/nfs/primary/graph_adaptive_reranking/queries.pkl")
        qrels = pd.read_pickle("/nfs/primary/graph_adaptive_reranking/qrels.pkl")
        
        # Util functions
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

        import os
        import uuid 
        folder_name = f"../in_training_eval/{uuid.uuid1()}"
        os.makedirs(folder_name)
        self.save_model(folder_name)
        lsr = LSR(folder_name) # load a trained LSR model
        
        # Index the corpus
        print("Starting indexing...")
        index_pipeline = lsr >> PisaToksIndexer(folder_name)
        index = index_pipeline.index(ohsumed_iterator())
        print("Indexing completed!")
        
        splade_retr = retr_pipeline = lsr.query_encoder() >> index.quantized()
        
        print("Starting retrieval...")
        run_df = splade_retr(queries)
        print("Retrieval completed!\n")
        
        retrieval_results = pt.Experiment(
            [run_df],
            queries,
            qrels,
            eval_metrics=[nDCG@10, sens_docs@10]
        )
        
        print("Retrieval results:")
        print(retrieval_results)

        wandb.log({
            "nDCG@10" : retrieval_results.iloc[0]["nDCG@10"],
            "sens_docs@10" : retrieval_results.iloc[0]["sens_docs@10"]
        })

        import shutil
        shutil.rmtree(folder_name, ignore_errors = True)


    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval
    ):
        if self.control.should_log:
            log = {}
            for metric in self.customed_log:
                log[metric] = (
                    self._nested_gather(self.customed_log[metric]).mean().item()
                )
                log[metric] = round(
                    (
                        log[metric]
                        / (self.state.global_step - self._globalstep_last_logged)
                        / self.args.gradient_accumulation_steps
                    ),
                    4,
                )
            self.log(log)
            for metric in self.customed_log:
                self.customed_log[metric] -= self.customed_log[metric]
            self.control.should_log = True
        super()._maybe_log_save_evaluate(
            tr_loss, None, model, trial, epoch, ignore_keys_for_eval
        )

    def _load_optimizer_and_scheduler(self, checkpoint):
        super()._load_optimizer_and_scheduler(checkpoint)
        if checkpoint is None:
            return
        if os.path.join(checkpoint, LOSS_NAME):
            self.loss.load_state_dict(torch.load(os.path.join(checkpoint, LOSS_NAME)))

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss
        """
        loss_output, q_reg, d_reg, to_log = model(self.loss, **inputs)
        sens_reg = to_log["sensitivity reg"]
        for log_metric in to_log:
            self.customed_log[log_metric] += to_log[log_metric]
        wandb.log({
            "weight_val" : model.encoder.sens_weight.data,
            "sens_weight grad" : model.encoder.sens_weight.data.grad,
            "ce_loss grad" : loss_output.grad,
            "q_reg grad" : q_reg.grad,
            "d_reg grad" : d_reg.grad
        })
        return loss_output + q_reg + d_reg + model.encoder.sens_weight * sens_reg

    def save_model(self, model_dir=None, _internal_call=False):
        """Save model checkpoint"""
        logger.info("Saving model checkpoint to %s", model_dir)
        if model_dir is None:
            model_dir = os.path.join(self.args.output_dir, "model")
        self.model.save_pretrained(model_dir)
        if self.tokenizer is not None:
            tokenizer_path = os.path.join(model_dir, "tokenizer")
            self.tokenizer.save_pretrained(tokenizer_path)
        loss_path = os.path.join(model_dir, LOSS_NAME)
        logger.info("Saving loss' state to %s", loss_path)
        torch.save(self.loss.state_dict(), loss_path)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Load from a checkpoint to continue training"""
        # Load model from checkpoint
        logger.info("Loading model's weight from %s", resume_from_checkpoint)
        self.model.load_state_dict(
            DualSparseEncoder.from_pretrained(resume_from_checkpoint).state_dict()
        )
