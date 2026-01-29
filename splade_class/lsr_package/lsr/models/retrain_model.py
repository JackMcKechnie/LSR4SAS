from lsr.models import TransformerMLMSparseEncoder, TransformerMLMConfig
from transformers import AutoConfig, AutoModelForMaskedLM
from safetensors.torch import load_file
from lsr.models.sparse_encoder import SparseEncoder
from lsr.utils import functional
from lsr.utils.functional import FunctionalFactory
from lsr.utils.pooling import PoolingFactory
from lsr.utils.sparse_rep import SparseRep
import torch

from lsr.models.mlm import TransformerMLMSparseEncoder, TransformerMLMConfig 
from transformers import AutoConfig, AutoModelForMaskedLM
from safetensors.torch import load_file
from lsr.models.sparse_encoder import SparseEncoder
from lsr.utils import functional
from lsr.utils.functional import FunctionalFactory
from lsr.utils.pooling import PoolingFactory
from lsr.utils.sparse_rep import SparseRep


class TransformerMLMSparseEncoderManualBackbone(TransformerMLMSparseEncoder):
    def __init__(self, config: TransformerMLMConfig = TransformerMLMConfig(), backbone=None):
        """
        Args:
            config (TransformerMLMConfig): Your model config.
            backbone (PreTrainedModel, optional): If provided, sets self.model manually.
        """
        
        # don't call the super class __init__ that loads pretrained backbone automatically
        super(TransformerMLMSparseEncoder, self).__init__(config)
        print("In TransformerMLMSparseEncoderManualBackbone()")
        self.model = TransformerMLMSparseEncoder().from_pretrained("/nfs/primary/SPLADE/splade_sans/learned-sparse-retrieval-1.0.0/outputs/splade_max_strategy1_group8/model/shared_encoder/")
        print(self.model.config)
        
    def forward(self, **kwargs):
        return self.model.forward(**kwargs)