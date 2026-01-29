from torch import nn
import torch
import numpy as np
import pickle


class Regularizer(nn.Module):
    """Base class for sparse regularizer.
    Attributes
    ----------
    weight : float
        the weight of regularizer in the loss
    T : int
        the weight get exponentially increased over T steps
    """

    def __init__(self, weight=0.0, T=1000) -> None:
        """
        Intializing the regularizer with weight and decaying steps
        Parameters
        ----------
            weight: float
                the regularizer's weight in the loss
            T: int
                warming up steps
        """
        super().__init__()
        self.weight_T = weight
        self.weight_t = 0
        self.T = T
        self.t = 0
        self.k = 100
        
    def step(self):
        """
        Perform a warming up step.
        The weight starts with zero and get expoentially increased step by step until the T-th step.
        """
        if self.t >= self.T:
            pass
        else:
            self.t += 1
            self.weight_t = self.weight_T * (self.t / self.T) ** 2

    def forward(self, reps):
        """
        reps: batch representation
        """
        raise NotImplementedError("This is an abstract regularizer only.")


class FLOPs(Regularizer):
    """
    Implementation of the FLOPs regularizer which is a mooth approximation for number of term overlap between a query and a document.
    Paper: https://arxiv.org/abs/2004.05665
    """

    def forward(self, reps):
        return (torch.abs(reps).mean(dim=0) ** 2).sum() * self.weight_t


class L1(Regularizer):
    """
    Implementation of the L1 regularizer
    """

    def forward(self, reps):
        return torch.abs(reps).sum(dim=1).mean() * self.weight_t

class SensReg(Regularizer):
    """
    Implementation of the custom sensitivity regulariser that uses 
    fixed top-k indices from SVM coefficients
    """
    def __init__(self, weight=0.0, T=1000, k=100) -> None:
        super().__init__(weight, T)  # Only pass weight and T to parent
        self.k = k  # Set k locally
        file = open("/nfs/primary/SPLADE/splade_class/svm_coefficients_v4.pkl", "rb")
        svm_vec = pickle.load(file)
        file.close()
        
        # Get the indices of the top-k values in the SVM vector
        self.top_k_indices = np.argsort(svm_vec)[-self.k:]
        # Convert to PyTorch tensor
        self.top_k_indices = torch.tensor(self.top_k_indices, dtype=torch.long)
    
    def forward(self, reps):        
        # Select only the top-k indices from each representation
        # We need to index the second dimension (features)
        if self.top_k_indices.device != reps.device:
            self.top_k_indices = self.top_k_indices.to(reps.device)
            
        # Select only the specific indices from each representation
        selected_reps = torch.index_select(reps, dim=1, index=self.top_k_indices)
        # Compute the regularization term using only these selected values
        return torch.abs(selected_reps).sum(dim=1).mean() * self.weight_t