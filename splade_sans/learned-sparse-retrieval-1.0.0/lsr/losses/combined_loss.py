from torch import nn
import torch
from lsr.losses import Loss, cross_dot_product, num_non_zero


class CombinedLoss(Loss):

    def __init__(self, classification_token_idx=0, q_regularizer=None, d_regularizer=None, alpha=0.5):
        super(CombinedLoss, self).__init__(q_regularizer, d_regularizer)
        self.ce = nn.CrossEntropyLoss(reduction="mean")
        self.classifier_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.88/0.12))
        self.alpha = alpha
        self.classification_token_idx = classification_token_idx
        self.historic_classification_loss = torch.tensor([]).cuda()
        self.historic_ranking_loss = torch.tensor([]).cuda()
        self.epsilon = 1e-8 
        
    def forward(self, q_reps, d_reps, sensitivity_labels=None):
        # Calculate similarity matrix
        sim_matrix = cross_dot_product(q_reps, d_reps)
        # cross_dot_product(q_reps, d_reps)
        reg_q_output = (
            torch.tensor(0.0, device=q_reps.device)
            if (self.q_regularizer is None)
            else self.q_regularizer(q_reps)
        )
        reg_d_output = (
            torch.tensor(0.0, device=d_reps.device)
            if (self.d_regularizer is None)
            else self.d_regularizer(d_reps)
        )
        
        if not self.q_regularizer is None:
            self.q_regularizer.step()
        if not self.d_regularizer is None:
            self.d_regularizer.step()
        q_num = q_reps.size(0)
        d_num = d_reps.size(0)
        assert d_num % q_num == 0
        doc_group_size = d_num // q_num
        labels = torch.arange(0, d_num, doc_group_size, device=sim_matrix.device)
        ce_loss = self.ce(sim_matrix, labels)

        classification_token_reps = d_reps[:, self.classification_token_idx]
        class_labels = sensitivity_labels.ravel()

        classification_loss = self.classifier_loss(classification_token_reps, class_labels.float())

        self.historic_classification_loss = torch.cat((self.historic_classification_loss, torch.tensor([classification_loss.item()]).cuda()))
        self.historic_ranking_loss = torch.cat((self.historic_ranking_loss, torch.tensor([ce_loss.item()]).cuda()))
        
        ce_loss = (ce_loss / (self.historic_ranking_loss.mean() + self.epsilon))
        classification_loss = (classification_loss / (self.historic_classification_loss.mean() + self.epsilon))

        combined_loss = ce_loss + classification_loss.squeeze()

        
        # Log additional info
        to_log = {
            "query reg": reg_q_output.detach(),
            "doc reg": reg_d_output.detach(),
            "query length": num_non_zero(q_reps),
            "doc length": num_non_zero(d_reps),
            "classification loss": classification_loss.detach(),
            "retrieval loss": ce_loss.detach(),
            "combined loss" : combined_loss,
        }

        return combined_loss, reg_q_output, reg_d_output, to_log