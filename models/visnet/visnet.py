'''
Adapted from https://github.com/microsoft/AI2BMD
'''

import re
from typing import Tuple, Optional, List

import torch
from torch import nn
from torch import Tensor
from torch_scatter import scatter

from . import priors, output_modules
from .visnet_block import ViSNetBlock
from ..model_utils import ModelOutput


class ViSNet(nn.Module):
    def __init__(
        self,
        config,
        prior_model=None,
        mean=None,
        std=None,
    ):
        super(ViSNet, self).__init__()

        self.config = config
        self.use_phi_module = self.config.model.use_phi_module

        self.representation_model = ViSNetBlock(config)
        # self.output_model = getattr(
        #     output_modules, "Equivariant" + config.model.output_model
        # )(config.model.embedding_dimension, config.model.activation)
        self.output_model = getattr(
            output_modules, "Equivariant" + config.model.output_model
        )(config.model.hidden_channels, config.model.activation)

        self.prior_model = prior_model
        if not self.output_model.allow_prior_model and prior_model is not None:
            self.prior_model = None

        self.reduce_op = config.model.reduce_op
        self.derivative = config.training.predict_forces

        if self.use_phi_module:
            self.electrostatic_offset = nn.Parameter(torch.tensor(1.0))
            self.electrostatic_bias = nn.Parameter(torch.tensor(0.0))

        mean = torch.scalar_tensor(0) if mean is None else mean
        self.register_buffer("mean", mean)
        std = torch.scalar_tensor(1) if std is None else std
        self.register_buffer("std", std)

        self.reset_parameters()

        self.epoch = None

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()
        if self.prior_model is not None:
            self.prior_model.reset_parameters()

    # def forward(self, data: dict[str, Tensor]) -> Tuple[Tensor, Optional[Tensor]]:
    def forward(self, batch) -> Tuple[Tensor, Tensor]:
        z = batch.z 
        pos = batch.pos 
        batch = batch.batch

        if self.derivative:
            with torch.enable_grad():
                pos.requires_grad_(True) # data['pos'].requires_grad_(True)

        if self.use_phi_module:
            x, v, phi, rho, pde_res = self.representation_model(z, pos, batch)
        else:
            x, v = self.representation_model(z, pos, batch)

        x = self.output_model.pre_reduce(x, v, z, pos, batch)
        x = x * self.std

        if self.prior_model is not None:
            x = self.prior_model(x, z)

        out = scatter(x, batch, dim=0, reduce=self.reduce_op)
        out = self.output_model.post_reduce(out)

        out = out + self.mean

        if self.use_phi_module:
            # Compute electrostatic energy term
            if self.epoch >= self.config.training.pde_warmup_epochs:
                self.electrostatic_term = 0.5 * (phi * rho).sum()
                out = out + self.electrostatic_offset * self.electrostatic_term + self.electrostatic_bias
            else:
                self.electrostatic_term = torch.tensor([0.0], dtype=torch.float32)

        # compute gradients with respect to coordinates
        if self.derivative:
            grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(out)]
            dy = torch.autograd.grad(
                [out],
                [pos],
                grad_outputs=grad_outputs,
                create_graph=True,
            )[0]
            if dy is None:
                raise RuntimeError(
                    "Autograd returned None for the force prediction."
                )
            return ModelOutput(out=out, pde_residual=pde_res, forces=-dy) 
        
        return ModelOutput(out=out, pde_residual=pde_res)