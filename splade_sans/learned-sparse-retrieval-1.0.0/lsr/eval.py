from pprint import pprint
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.core.hydra_config import HydraConfig
from pprint import pprint
import logging
import wandb
import os
from pathlib import Path
from datetime import datetime
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="configs", config_name="config")
def eval(conf: DictConfig):
    print("** Evaluating**\n" * 100)


if __name__ == "__main__":
    eval()