"""This module provides utility function for training."""
import os
import re
from typing import Any, Dict
import torch
from torch import nn

#from .options import arguments

#opt = arguments()


def saver(state: Dict[str, float], save_dir: str, epoch: int) -> None:
    torch.save(state, save_dir + "net_" + str(epoch) + ".pt")


def latest_checkpoint(checkpoints_dir: str) -> int:
    """Returns latest checkpoint."""
    if os.path.exists(checkpoints_dir):
        all_chkpts = "".join(os.listdir(checkpoints_dir))
        if len(all_chkpts) > 0:
            latest = max(map(int, re.findall(r"\d+", all_chkpts)))
        else:
            latest = None
    else:
        latest = None
    return latest

# Weight initialization for conv layers and fc layers
def weights_init(param: Any) -> None:
    """Initializes weights of Conv and fully connected."""

    if isinstance(param, nn.Conv2d):
        torch.nn.init.normal_(param.weight.data, mean=0.0, std=0.01)
        
    elif isinstance(param, nn.Linear):
        torch.nn.init.xavier_uniform_(param.weight.data)
        torch.nn.init.constant_(param.bias.data, 0.0)
        