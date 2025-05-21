from typing import List, Union

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn.models.schnet import GaussianSmearing
from torch_geometric.utils import get_laplacian
from torch_scatter import scatter, scatter_add

from .model_utils import ModelOutput
from .p3m_utils import FNO3d
from .p3m_utils import add_cell_to_data, get_nonpbc_mesh_atom_graph
from .p3m_utils import (get_distances, get_distances_pbc,
                           radius_determinstic, radius_graph_determinstic,
                           radius_graph_pbc, radius_pbc)
from .p3m_utils import InteractionBlock, MultiheadAttention, Scalar

from.phi_module_utils import block_diag_sparse, AlphaNet, laplacian_matvec


class SchNet_P3M(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.use_phi_module = self.config.model.use_phi_module

        self.regress_forces = self.config.training.predict_forces
        self.use_pbc = self.config.model.use_pbc
        self.num_layers = self.config.model.num_interactions
        self.num_rbf = self.config.model.num_gaussians
        self.num_filters = self.config.model.num_filters
        self.hidden_channels = self.config.model.hidden_features
        self.max_z = self.config.model.max_z
        self.atom_cutoff = self.config.model.radius_cutoff
        self.max_a2a_neighbors = self.config.model.max_num_neighbors
        self.grid_cutoff = self.config.model.p3m_grid_cutoff
        self.max_a2m_neighbors = self.config.model.p3m_max_a2m_neighbors
        self.long_type = self.config.model.long_type

        assert not self.use_pbc, 'PBC is not supported for this model yet'
        num_grids = self.config.model.p3m_num_grids

        if isinstance(num_grids, int):
            self.num_grids = [num_grids, num_grids, num_grids]
        else:
            self.num_grids = num_grids
            
        self.total_num_grids = self.num_grids[0] * self.num_grids[1] * self.num_grids[2]
        
        self.embedding = nn.Embedding(self.max_z, self.hidden_channels)
        self.a2a_distance_expansion = GaussianSmearing(0.0, self.atom_cutoff, self.num_rbf)
        self.a2m_distance_expansion = GaussianSmearing(0.0, self.grid_cutoff, self.num_rbf)
        self.m2a_distance_expansion = GaussianSmearing(0.0, self.grid_cutoff, self.num_rbf)
        
        self.sl_block = nn.ModuleList()
        for _ in range(self.num_layers):
            a2m_mp = InteractionBlock(self.hidden_channels, self.num_rbf, self.num_filters, self.grid_cutoff)
            m2a_mp = InteractionBlock(self.hidden_channels, self.num_rbf, self.num_filters, self.grid_cutoff)
            short_mp = InteractionBlock(self.hidden_channels, self.num_rbf, self.num_filters, self.atom_cutoff)
            if self.long_type == 'FNO':
                long_mp = FNO3d(
                    *self.num_grids,
                    hidden_channels=self.hidden_channels // 2, 
                    in_channels=self.hidden_channels, 
                    out_channels=self.hidden_channels, 
                    n_layers=1,
                    lifting_channels=self.hidden_channels // 2,
                    projection_channels=self.hidden_channels // 2,
                    non_linearity=nn.SiLU(),
                )
            
            elif self.long_type == 'MHA': 
                long_mp = MultiheadAttention(self.hidden_channels, self.hidden_channels, 8)
            else:
                raise ValueError(f'Unknown long range interaction type: {self.long_type}')
            self.sl_block.append(
                ShortLongMixLayer(
                    self.hidden_channels,
                    self.num_grids,
                    a2m_mp,
                    m2a_mp,
                    short_mp,
                    long_mp,
                )
            )
        self.out_a_norm = nn.LayerNorm(self.hidden_channels)
        self.out_m_norm = nn.LayerNorm(self.hidden_channels)
        self.a_output = Scalar(self.hidden_channels)
        self.m_output = Scalar(self.hidden_channels)

        if self.use_phi_module:
            self.electrostatic_offset = nn.Parameter(torch.tensor(1.0))
            self.electrostatic_bias = nn.Parameter(torch.tensor(0.0))

            self.alpha_model = AlphaNet(in_channels=self.hidden_channels, k=self.config.training.k_eigenvalues)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.embedding.reset_parameters()
        for block in self.sl_block:
            block.reset_parameters()
        self.out_a_norm.reset_parameters()
        self.out_m_norm.reset_parameters()
        self.a_output.reset_parameters()
        self.m_output.reset_parameters()
    
    def forward(self, data):
        data_cell = add_cell_to_data(data)
        mesh_atom_graph = get_nonpbc_mesh_atom_graph(data_cell, expand_size=self.config.model.p3m_expand_size, 
                                                     num_grids=self.config.model.p3m_num_grids) # sget_mesh_atom_graph(data)
        
        if self.regress_forces:
            mesh_atom_graph['atom'].pos.requires_grad_(True)
        
        bs = mesh_atom_graph.num_graphs if hasattr(mesh_atom_graph['atom'], 'batch') else 1
        
        a_pos = mesh_atom_graph['atom'].pos
        m_pos = mesh_atom_graph['mesh'].pos
        
        if not self.use_pbc:
            a2a_edge_index, _ = radius_graph_determinstic(
                mesh_atom_graph['atom'], 
                self.atom_cutoff, 
                self.max_a2a_neighbors,
                symmetrize=True,
            )
            a2a_edge_weights = get_distances(a2a_edge_index, a_pos, return_distance_vec=False)
            a2m_edge_index = radius_determinstic(
                mesh_atom_graph['atom'],
                mesh_atom_graph['mesh'],
                self.grid_cutoff,
                self.max_a2m_neighbors,
            )
            a2m_edge_weights = get_distances(a2m_edge_index, a_pos, m_pos, return_distance_vec=False)
            m2a_edge_index = a2m_edge_index.flip(0)
            m2a_edge_weights = get_distances(m2a_edge_index, m_pos, a_pos, return_distance_vec=False)
        else:
            cell = mesh_atom_graph['atom'].cell
            a2a_edge_index, a2a_cell_offsets, a2a_neighbors = radius_graph_pbc(
                mesh_atom_graph['atom'], 
                self.atom_cutoff, 
                self.max_a2a_neighbors,
                symmetrize=True,
            )
            a2a_edge_weights = get_distances_pbc(
                a2a_edge_index, 
                cell, 
                a2a_cell_offsets, 
                a2a_neighbors, 
                a_pos, 
                return_distance_vec=False
            )
            
            a2m_edge_index, a2m_cell_offset, a2m_neighbors = radius_pbc(
                mesh_atom_graph['atom'],
                mesh_atom_graph['mesh'],
                self.grid_cutoff,
                self.max_a2m_neighbors,
            )
            
            a2m_edge_weights = get_distances_pbc(
                a2m_edge_index, 
                cell, 
                a2m_cell_offset, 
                a2m_neighbors, 
                a_pos, 
                m_pos,
                return_distance_vec=False
            )
            
            m2a_edge_index = a2m_edge_index.flip(0)
            m2a_cell_offset = -1 * a2m_cell_offset
            m2a_neighbors = a2m_neighbors
            
            m2a_edge_weights = get_distances_pbc(
                m2a_edge_index,
                cell,
                m2a_cell_offset,
                m2a_neighbors,
                m_pos,
                a_pos,
                return_distance_vec=False
            )
        
        a2a_edge_attr = self.a2a_distance_expansion(a2a_edge_weights)
        a2m_edge_attr = self.a2m_distance_expansion(a2m_edge_weights)
        m2a_edge_attr = self.m2a_distance_expansion(m2a_edge_weights)
        
        # N_atoms, F
        a_x = self.embedding(mesh_atom_graph['atom'].z)
        # NonTrainable Message Passing For Initial Mesh Embedding
        a_x_j = torch.index_select(a_x, 0, a2m_edge_index[0])
        m_x = scatter(a_x_j, a2m_edge_index[1], dim=0, reduce='mean', dim_size=self.total_num_grids * bs)

        if self.use_phi_module:
            # Eigenbasis projection 
            edge_index_L, Ls = get_laplacian(edge_index=a2a_edge_index, edge_weight=a2a_edge_weights, normalization='rw')
            sparse_diag_block_L = block_diag_sparse(edge_index_L, Ls, data.batch)
            evals, U = torch.lobpcg(sparse_diag_block_L, k=self.config.training.k_eigenvalues, largest=False)
        
        for i in range(self.num_layers):
            a_x, m_x = self.sl_block[i](
                a_x, 
                m_x, 
                a2a_edge_index, 
                a2m_edge_index, 
                m2a_edge_index,
                a2a_edge_weights,
                a2m_edge_weights,
                m2a_edge_weights,
                a2a_edge_attr,
                a2m_edge_attr,
                m2a_edge_attr,
            )

            if self.use_phi_module:
                alpha = self.alpha_model(a_x) 

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
            net_charge = torch.abs(scatter_add(rho, data.batch, dim=0)).sum()
            pde_res += self.config.training.net_charge_lambda * net_charge
        else:
            pde_res = None
            
        out_a_x = self.out_a_norm(a_x)
        out_m_x = self.out_m_norm(m_x)
        
        output_a_x = self.a_output(out_a_x)
        
        energy_a = scatter(
            output_a_x, 
            data.batch, # mesh_atom_graph['atom'].batch if hasattr(mesh_atom_graph['atom'], 'batch') else torch.zeros_like(mesh_atom_graph['atom'].z),
            dim=0, 
            reduce='sum'
        )

        output_m_x = self.m_output(out_m_x)
        
        energy_m = torch.sum(output_m_x.reshape(bs, -1), dim=-1, keepdim=True)
        energy = energy_a + energy_m

        if self.use_phi_module:
            # Compute electrostatic energy term
            if self.epoch >= self.config.training.pde_warmup_epochs:
                self.electrostatic_term = 0.5 * (phi * rho).sum()
                energy += self.electrostatic_offset * self.electrostatic_term + self.electrostatic_bias
            else:
                self.electrostatic_term = torch.tensor([0.0], dtype=torch.float32)
        
        if self.regress_forces:
            forces = -1 * (
                torch.autograd.grad(
                    energy,
                    mesh_atom_graph['atom'].pos,
                    grad_outputs=torch.ones_like(energy),
                    create_graph=True,
                )[0]
            )
            return ModelOutput(out=energy, forces=forces, pde_residual=pde_res) # energy, forces
        else:
            return ModelOutput(out=energy, pde_residual=pde_res) # energy, None
      

class ShortLongMixLayer(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_grids: List[int],
        a2m_mp: nn.Module,
        m2a_mp: nn.Module,
        short_mp: nn.Module,
        long_mp: nn.Module,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.a2m_mp = a2m_mp
        self.m2a_mp = m2a_mp
        self.short_mp = short_mp
        self.long_mp = long_mp
        self.num_grids = num_grids
        self.a2m_layernorm = nn.LayerNorm(hidden_channels)
        self.m2a_layernorm = nn.LayerNorm(hidden_channels)
        self.short_layernorm = nn.LayerNorm(hidden_channels)
        self.long_layernorm = nn.LayerNorm(hidden_channels)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.a2m_mp.reset_parameters()
        self.m2a_mp.reset_parameters()
        self.short_mp.reset_parameters()
        self.long_mp.reset_parameters()
        self.a2m_layernorm.reset_parameters()
        self.m2a_layernorm.reset_parameters()
        self.short_layernorm.reset_parameters()
        self.long_layernorm.reset_parameters()
    
    def forward(
        self, 
        a_x, 
        m_x, 
        a2a_edge_index,
        a2m_edge_index,
        m2a_edge_index,
        a2a_edge_weights,
        a2m_edge_weights,
        m2a_edge_weights,
        a2a_edge_attr,
        a2m_edge_attr,
        m2a_edge_attr,
    ):
        
        delta_a_x, delta_m_x = a_x, m_x
        
        # N_atoms, F
        a_x = self.short_layernorm(a_x)
        a_x = self.short_mp(a_x, a2a_edge_index, a2a_edge_weights, a2a_edge_attr, dim_size=a_x.shape[0])
        
        # N_meshs, F
        m_x = self.long_layernorm(m_x)
        if isinstance(self.long_mp, MultiheadAttention):
            m_x = m_x.reshape(-1, torch.prod(torch.tensor(self.num_grids)), self.hidden_channels)
            m_x = self.long_mp(m_x)
            m_x = m_x.reshape(-1, self.hidden_channels)
        else:
            m_x = m_x.reshape(-1, self.num_grids[0], self.num_grids[1], self.num_grids[2], self.hidden_channels).permute(0, 4, 1, 2, 3)
            m_x = self.long_mp(m_x).permute(0, 2, 3, 4, 1).reshape(-1, self.hidden_channels)
        
        # N_meshs, F
        a2m_message = self.a2m_mp(a_x, a2m_edge_index, a2m_edge_weights, a2m_edge_attr, dim_size=m_x.shape[0])
        a2m_message = self.a2m_layernorm(a2m_message)
        
        # N_atoms, F
        m2a_message = self.m2a_mp(m_x, m2a_edge_index, m2a_edge_weights, m2a_edge_attr, dim_size=a_x.shape[0])
        m2a_message = self.m2a_layernorm(m2a_message)
        
        return a_x + m2a_message + delta_a_x, m_x + a2m_message + delta_m_x