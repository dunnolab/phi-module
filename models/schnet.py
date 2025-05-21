from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add

from torch_geometric.nn.models import SchNet
from torch_geometric.utils import get_laplacian

from .model_utils import ModelOutput
from .phi_module_utils import laplacian_matvec, block_diag_sparse, AlphaNet


class SchNetBase(SchNet):
    def __init__(self, config, *args, **kwargs):
        self.config = config
        self.use_phi_module = config.model.use_phi_module

        assert not self.config.model.use_pbc, 'PBC is not supported for this model yet'
 
        super().__init__(hidden_channels=config.model.hidden_features,
                         num_filters=config.model.num_filters,
                         num_interactions=config.model.num_interactions,
                         num_gaussians=config.model.num_gaussians,
                         cutoff=config.model.radius_cutoff,
                         max_num_neighbors=config.model.max_num_neighbors, 
                         *args,
                         **kwargs)
        
        if self.use_phi_module:
            self.electrostatic_offset = nn.Parameter(torch.tensor(1.0))
            self.electrostatic_bias = nn.Parameter(torch.tensor(0.0))

            self.alpha_model = AlphaNet(in_channels=self.hidden_channels, k=self.config.training.k_eigenvalues)

    def forward(self, batch):
        z = batch.z.long() 
        pos = batch.pos 
        batch = torch.zeros_like(z) if batch.batch is None else batch.batch

        if self.config.training.predict_forces:
            pos.requires_grad_(True)

        h = self.embedding(z)
        edge_index, edge_weight = self.interaction_graph(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)

        if self.use_phi_module:
            # Eigenbasis projection
            edge_index_L, Ls = get_laplacian(edge_index=edge_index, edge_weight=edge_weight, normalization='rw')
            sparse_diag_block_L = block_diag_sparse(edge_index_L, Ls, batch)
            evals, U = torch.lobpcg(sparse_diag_block_L, k=self.config.training.k_eigenvalues, largest=False)

        for i, interaction in enumerate(self.interactions):
            message = interaction(h, edge_index, edge_weight, edge_attr)
            h = h + message

            if self.use_phi_module:
                alpha = self.alpha_model(h) 

                # Perform spectral projection to accumulate potential and charges
                if i == 0:
                    phi = U @ alpha
                    rho = (U * evals) @ alpha
                else:
                    phi_step = U @ alpha
                    rho_step = (U * evals) @ alpha

                    phi += phi_step
                    rho += rho_step

        if self.use_phi_module:
            # Compute PDE residual
            L_phi = laplacian_matvec(Ls, phi, edge_index_L)
            pde_res = (L_phi - rho).pow(2).mean()

            # Apply optional constraint on net zero charge
            net_charge = torch.abs(scatter_add(rho, batch, dim=0)).sum()
            pde_res += self.config.training.net_charge_lambda * net_charge
        else:
            pde_res = None

        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        if self.dipole:
            # Get center of mass.
            mass = self.atomic_mass[z].view(-1, 1)
            M = self.sum_aggr(mass, batch, dim=0)
            c = self.sum_aggr(mass * pos, batch, dim=0) / M
            h = h * (pos - c.index_select(0, batch))

        if not self.dipole and self.mean is not None and self.std is not None:
            h = h * self.std + self.mean

        if not self.dipole and self.atomref is not None:
            h = h + self.atomref(z)

        out = self.readout(h, batch, dim=0)

        if self.dipole:
            out = torch.norm(out, dim=-1, keepdim=True)

        if self.scale is not None:
            out = self.scale * out

        if self.use_phi_module:
            # Compute electrostatic energy term
            if self.epoch >= self.config.training.pde_warmup_epochs:
                self.electrostatic_term = 0.5 * (phi * rho).sum()
                out += self.electrostatic_offset * self.electrostatic_term + self.electrostatic_bias
            else:
                self.electrostatic_term = torch.tensor([0.0], dtype=torch.float32)

        if self.config.training.predict_forces:
            grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(out)]
            forces = torch.autograd.grad(
                [out],
                [pos],
                grad_outputs=grad_outputs,
                create_graph=True,
            )[0]

            if forces is None:
                raise RuntimeError(
                    "Autograd returned None for the force prediction."
                )
            return ModelOutput(out=out, forces=-forces, pde_residual=pde_res)

        return ModelOutput(out=out, pde_residual=pde_res)
    