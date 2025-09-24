import torch.nn as nn
import torch

def make_model():
    torch.manual_seed(42)
    model=nn.Sequential(
        nn.Linear(11,8),
        nn.ReLU(),
        nn.Linear(8,1)
    )
    return model