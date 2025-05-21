import torch
import torch.nn as nn

from .model_utils import ModelOutput


class LinearModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.linear = nn.Linear(config.model.input_dim, config.model.output_dim)
        
    def forward(self, x):
        return ModelOutput(out=self.linear(x))