import logging
from pathlib import Path
from typing import Union


import torch
from torch import nn
from torch.optim import Optimizer

# setup logging
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter
formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
log.addHandler(ch)

def save_model_checkpoint(epoch: int, device: str, model: nn.Module, optimizer: Optimizer, output_dir: Union[str, Path], prefix = None, suffix = None, overwrite=False):
    # consider abstracting the prefix and suffix string building into a separate function
    if overwrite == False:
        filepath = f'{output_dir}/{prefix + "_" if prefix is not None else ""}model_checkpoint_{epoch}{"_" + suffix if suffix is not None else ""}.pt'
    else:
        filepath = f'{output_dir}/{prefix + "_" if prefix is not None else ""}model_checkpoint{"_" + suffix if suffix is not None else ""}.pt'
    log.info(f"Saving model checkpoint {filepath}")
    torch.save({'epoch': epoch,
        'device': device,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        filepath)

def load_model(checkpoint_path: str, model: nn.Module):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    return model

