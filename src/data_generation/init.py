import random

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf


def read_config(fname):
    """
    Read config from yaml file and print it
    """
    with open(fname, "r") as stream:
        cfg = yaml.safe_load(stream)

    return OmegaConf.create(cfg)


def set_seed(seed=0):
    """
    Don't set true seed to be nearby values. Doesn't give best randomness
    """
    rng = np.random.default_rng(seed)
    true_seed = int(rng.integers(2**30))

    random.seed(true_seed)
    np.random.seed(true_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(true_seed)
    torch.cuda.manual_seed_all(true_seed)
