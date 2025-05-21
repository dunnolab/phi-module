from typing import List, Union

import torch
import torch.nn as nn
from torch_geometric.utils import get_laplacian
from torch_geometric.nn.models.schnet import GaussianSmearing
from torch_scatter import scatter, scatter_add
from torch_sparse import SparseTensor

from ..p3m_utils import FNO3d
from ..p3m_utils import InteractionBlock, MultiheadAttention

from ..p3m_utils import (get_distances, get_distances_pbc,
                            radius_determinstic, radius_graph_determinstic,
                            radius_graph_pbc, radius_pbc)
from ..p3m_utils import Scalar, add_cell_to_data, get_nonpbc_mesh_atom_graph

from ..model_utils import ModelOutput
from .utils import inner_product_normalized, ragged_range, repeat_blocks
from .initializers import he_orthogonal_init
from .layers.base_layers import Dense
from .layers.efficient import EfficientInteractionDownProjection
from .layers.embedding_block import AtomEmbedding, EdgeEmbedding
from .layers.interaction_block import InteractionBlockTripletsOnly
from .layers.radial_basis import RadialBasis
from .layers.spherical_basis import CircularBasisLayer

from ..phi_module_utils import laplacian_matvec, AlphaNet, block_diag_sparse


class GemNetT_P3M(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.use_phi_module = self.config.model.use_phi_module

        assert not self.config.model.use_pbc, 'PBC is not supported for this model yet'

        self.regress_forces = self.config.training.predict_forces
        self.use_pbc = self.config.model.use_pbc
        self.num_layers = self.config.model.num_blocks
        self.num_rbf = self.config.model.p3m_num_rbf
        self.num_filters = self.config.model.p3m_num_filters
        self.max_z = self.config.model.max_z
        self.atom_cutoff = self.config.model.cutoff
        self.max_a2a_neighbors = self.config.model.max_neighbors
        self.grid_cutoff = self.config.model.p3m_grid_cutoff
        self.max_a2m_neighbors = self.config.model.p3m_max_a2m_neighbors

        num_grids = self.config.model.p3m_num_grids
        num_radial = self.config.model.num_radial
        num_spherical = self.config.model.num_spherical
        rbf = self.config.model.rbf
        cbf = self.config.model.cbf
        emb_size_rbf = self.config.model.emb_size_rbf
        emb_size_cbf = self.config.model.emb_size_cbf
        envelope = self.config.model.envelope
        emb_size_atom = self.config.model.emb_size_atom
        emb_size_edge = self.config.model.emb_size_edge
        emb_size_trip = self.config.model.emb_size_trip
        emb_size_bil_trip = self.config.model.emb_size_bil_trip
        activation = self.config.model.activation
        long_type = self.config.model.long_type

        num_before_skip = self.config.model.num_before_skip
        num_after_skip = self.config.model.num_after_skip
        num_concat = self.config.model.num_concat
        num_atom = self.config.model.num_atom
        
        if isinstance(num_grids, int):
            self.num_grids = [num_grids, num_grids, num_grids]
        else:
            self.num_grids = num_grids
            
        self.total_num_grids = self.num_grids[0] * self.num_grids[1] * self.num_grids[2]

        # GemNet variants
        self.direct_forces = self.config.model.direct_forces
        self.extensive = self.config.model.extensive

        ### ---------------------------------- Basis Functions ---------------------------------- ###
        self.radial_basis = RadialBasis(
            num_radial=num_radial,
            cutoff=self.atom_cutoff,
            rbf=rbf,
            envelope=envelope,
        )
        radial_basis_cbf3 = RadialBasis(
            num_radial=num_radial,
            cutoff=self.atom_cutoff,
            rbf=rbf,
            envelope=envelope,
        )
        self.cbf_basis3 = CircularBasisLayer(
            num_spherical,
            radial_basis=radial_basis_cbf3,
            cbf=cbf,
            efficient=True,
        )
        
        self.a2m_distance_expansion = GaussianSmearing(0.0, self.grid_cutoff, self.num_rbf)
        self.m2a_distance_expansion = GaussianSmearing(0.0, self.grid_cutoff, self.num_rbf)
        
        ### ------------------------------- Share Down Projections ------------------------------ ###
        # Share down projection across all interaction blocks
        self.mlp_rbf3 = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            bias=False,
        )
        self.mlp_cbf3 = EfficientInteractionDownProjection(
            num_spherical, num_radial, emb_size_cbf
        )

        # Share the dense Layer of the atom embedding block accross the interaction blocks
        self.mlp_rbf_h = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            bias=False,
        )
        ### ------------------------------------------------------------------------------------- ###

        # Embedding block
        self.atom_emb = AtomEmbedding(emb_size_atom, self.max_z)
        self.edge_emb = EdgeEmbedding(
            emb_size_atom, num_radial, emb_size_edge, activation=activation
        )

        self.sl_block = nn.ModuleList()
        for i in range(self.num_layers):
            a2m_mp = InteractionBlock(emb_size_atom, self.num_rbf, self.num_filters, self.grid_cutoff)
            m2a_mp = InteractionBlock(emb_size_atom, self.num_rbf, self.num_filters, self.grid_cutoff)  
            short_mp = InteractionBlockTripletsOnly(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_trip=emb_size_trip,
                    emb_size_rbf=emb_size_rbf,
                    emb_size_cbf=emb_size_cbf,
                    emb_size_bil_trip=emb_size_bil_trip,
                    num_before_skip=num_before_skip,
                    num_after_skip=num_after_skip,
                    num_concat=num_concat,
                    num_atom=num_atom,
                    activation=activation,
                    name=f"IntBlock_{i+1}",
                )   
            if long_type == 'FNO':
                long_mp = FNO3d(
                    *self.num_grids,
                    hidden_channels=emb_size_atom // 2, 
                    in_channels=emb_size_atom, 
                    out_channels=emb_size_atom, 
                    n_layers=1,
                    lifting_channels=emb_size_atom // 2,
                    projection_channels=emb_size_atom // 2,
                    non_linearity=nn.SiLU(),
                )
            elif long_type == 'MHA':
                long_mp = MultiheadAttention(emb_size_atom, emb_size_atom, 8)
            else:
                raise ValueError(f'Unknown long range interaction type: {long_type}')
            self.sl_block.append(
                ShortLongMixLayer(
                    emb_size_atom,
                    emb_size_edge,
                    self.num_grids,
                    a2m_mp,
                    m2a_mp,
                    short_mp,
                    long_mp,
                )
            )
        
        self.a_mlp_rbf_out = Dense(num_radial, emb_size_edge, activation=None, bias=False)    
        
        self.out_edge_norm = nn.LayerNorm(emb_size_edge)
        self.out_m_norm = nn.LayerNorm(emb_size_atom)   
        self.a_output = Scalar(emb_size_edge)
        self.m_output = Scalar(emb_size_atom)
        
        if self.regress_forces and self.direct_forces:
            self.m_mlp_rbf_out = Dense(self.num_rbf, emb_size_atom, activation=None, bias=False)
            self.a_out_forces = Scalar(emb_size_edge)
            self.m_out_forces = Scalar(emb_size_atom)
            
        self.shared_parameters = [
            (self.mlp_rbf3.linear.weight, self.num_layers),
            (self.mlp_cbf3.weight, self.num_layers),
            (self.mlp_rbf_h.linear.weight, self.num_layers),
        ]

        if self.use_phi_module:
            self.electrostatic_offset = nn.Parameter(torch.tensor(1.0))
            self.electrostatic_bias = nn.Parameter(torch.tensor(0.0))

            self.alpha_model = AlphaNet(in_channels=config.model.emb_size_atom, k=self.config.training.k_eigenvalues)

        self.reset_parameters()
    
    def reset_parameters(self):
        self.atom_emb.reset_parameters()
        self.edge_emb.reset_parameters()
        self.mlp_rbf3.reset_parameters()
        self.mlp_cbf3.reset_parameters()
        self.mlp_rbf_h.reset_parameters()
        for layer in self.sl_block:
            layer.reset_parameters()
        self.a_mlp_rbf_out.reset_parameters()
        self.out_edge_norm.reset_parameters()
        self.out_m_norm.reset_parameters()
        self.a_output.reset_parameters()
        self.m_output.reset_parameters()
        if self.regress_forces and self.direct_forces:
            self.m_mlp_rbf_out.reset_parameters()
            self.a_out_forces.reset_parameters()
            self.m_out_forces.reset_parameters()
        
    def get_triplets(self, edge_index, num_atoms):
        """
        Get all b->a for each edge c->a.
        It is possible that b=c, as long as the edges are distinct.

        Returns
        -------
        id3_ba: torch.Tensor, shape (num_triplets,)
            Indices of input edge b->a of each triplet b->a<-c
        id3_ca: torch.Tensor, shape (num_triplets,)
            Indices of output edge c->a of each triplet b->a<-c
        id3_ragged_idx: torch.Tensor, shape (num_triplets,)
            Indices enumerating the copies of id3_ca for creating a padded matrix
        """
        idx_s, idx_t = edge_index  # c->a (source=c, target=a)

        value = torch.arange(
            idx_s.size(0), device=idx_s.device, dtype=idx_s.dtype
        )
        # Possibly contains multiple copies of the same edge (for periodic interactions)
        adj = SparseTensor(
            row=idx_t,
            col=idx_s,
            value=value,
            sparse_sizes=(num_atoms, num_atoms),
        )
        adj_edges = adj[idx_t]

        # Edge indices (b->a, c->a) for triplets.
        id3_ba = adj_edges.storage.value()
        id3_ca = adj_edges.storage.row()

        # Remove self-loop triplets
        # Compare edge indices, not atom indices to correctly handle periodic interactions
        mask = id3_ba != id3_ca
        id3_ba = id3_ba[mask]
        id3_ca = id3_ca[mask]

        # Get indices to reshape the neighbor indices b->a into a dense matrix.
        # id3_ca has to be sorted for this to work.
        num_triplets = torch.bincount(id3_ca, minlength=idx_s.size(0))
        id3_ragged_idx = ragged_range(num_triplets)

        return id3_ba, id3_ca, id3_ragged_idx

    def generate_interaction_graph(self, mesh_atom_graph):
        num_atoms = mesh_atom_graph['atom'].z.size(0)
        a_pos = mesh_atom_graph['atom'].pos
        if not self.use_pbc:
            a2a_edge_index, a2a_neighbors = radius_graph_determinstic(
                    mesh_atom_graph['atom'], 
                    self.atom_cutoff, 
                    self.max_a2a_neighbors,
                    symmetrize=True,
                )
            a2a_edge_weights, a2a_normed_vec = get_distances(a2a_edge_index, a_pos, return_distance_vec=True)
        else:
            cell = mesh_atom_graph['atom'].cell
            a2a_edge_index, a2a_cell_offsets, a2a_neighbors = radius_graph_pbc(
                mesh_atom_graph['atom'], 
                self.atom_cutoff, 
                self.max_a2a_neighbors,
                symmetrize=True,
            )
            a2a_edge_weights, a2a_normed_vec = get_distances_pbc(
                a2a_edge_index, 
                cell, 
                a2a_cell_offsets, 
                a2a_neighbors, 
                a_pos, 
                return_distance_vec=True
            )

        # These vectors actually point in the opposite direction.
        # But we want to use col as idx_t for efficient aggregation.
        a2a_normed_vec = -a2a_normed_vec

        # Indices for swapping c->a and a->c (for symmetric MP)
        block_sizes = a2a_neighbors // 2
        id_swap = repeat_blocks(
            block_sizes,
            repeats=2,
            continuous_indexing=False,
            start_idx=block_sizes[0],
            block_inc=block_sizes[:-1] + block_sizes[1:],
            repeat_inc=-block_sizes,
        )

        id3_ba, id3_ca, id3_ragged_idx = self.get_triplets(
            a2a_edge_index, num_atoms=num_atoms
        )

        return (
            a2a_edge_index,
            a2a_neighbors,
            a2a_edge_weights,
            a2a_normed_vec,
            id_swap,
            id3_ba,
            id3_ca,
            id3_ragged_idx,
        )

    def forward(self, data):
        data_cell = add_cell_to_data(data)
        mesh_atom_graph = get_nonpbc_mesh_atom_graph(data_cell, expand_size=self.config.model.p3m_expand_size, 
                                                     num_grids=self.config.model.p3m_num_grids)
        
        if self.regress_forces and not self.direct_forces:
            mesh_atom_graph['atom'].pos.requires_grad_(True)
        
        bs = max(data.batch) + 1 # mesh_atom_graph.num_graphs if hasattr(mesh_atom_graph['atom'], 'batch') else 1
        batch  = data.batch # mesh_atom_graph['atom'].batch if hasattr(mesh_atom_graph['atom'], 'batch') else torch.zeros_like(mesh_atom_graph['atom'].atomic_numbers)
        
        a_pos = mesh_atom_graph['atom'].pos
        m_pos = mesh_atom_graph['mesh'].pos
        
        (   
            a2a_edge_index,
            neighbors,
            a2a_edge_weights,
            a2a_normed_vec,
            id_swap,
            id3_ba,
            id3_ca,
            id3_ragged_idx,
        ) = self.generate_interaction_graph(mesh_atom_graph)
        idx_s, idx_t = a2a_edge_index
        
        if not self.use_pbc:
            a2m_edge_index = radius_determinstic(
                    mesh_atom_graph['atom'],
                    mesh_atom_graph['mesh'],
                    self.grid_cutoff,
                    self.max_a2m_neighbors,
                )
            a2m_edge_weights = get_distances(a2m_edge_index, a_pos, m_pos, return_distance_vec=False)
            
            m2a_edge_index = a2m_edge_index.flip(0)
            m2a_edge_weights, m2a_normed_vec = get_distances(m2a_edge_index, m_pos, a_pos, return_distance_vec=True)
            m2a_normed_vec = -m2a_normed_vec
        else:
            cell = mesh_atom_graph['atom'].cell
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
            
            m2a_edge_weights, m2a_normed_vec = get_distances_pbc(
                m2a_edge_index,
                cell,
                m2a_cell_offset,
                m2a_neighbors,
                m_pos,
                a_pos,
                return_distance_vec=True
            )
            m2a_normed_vec = -m2a_normed_vec
        
        a2m_edge_attr = self.a2m_distance_expansion(a2m_edge_weights)
        m2a_edge_attr = self.m2a_distance_expansion(m2a_edge_weights)

        # Calculate triplet angles
        cosφ_cab = inner_product_normalized(a2a_normed_vec[id3_ca], a2a_normed_vec[id3_ba])
        rad_cbf3, cbf3 = self.cbf_basis3(a2a_edge_weights, cosφ_cab, id3_ca)

        rbf = self.radial_basis(a2a_edge_weights)

        # Embedding block
        a_x = self.atom_emb(mesh_atom_graph['atom'].z)  # (nAtoms, emb_size_atom
        a_x_j = torch.index_select(a_x, 0, a2m_edge_index[0])
        m_x = scatter(a_x_j, a2m_edge_index[1], dim=0, reduce='mean', dim_size=self.total_num_grids * bs)
        
        m = self.edge_emb(a_x, rbf, idx_s, idx_t)  # (nEdges, emb_size_edge)
        
        rbf3 = self.mlp_rbf3(rbf)
        cbf3 = self.mlp_cbf3(rad_cbf3, cbf3, id3_ca, id3_ragged_idx)
        rbf_h = self.mlp_rbf_h(rbf)

        if self.use_phi_module:
            # Eigenbasis projection
            edge_index_L, Ls = get_laplacian(edge_index=a2a_edge_index, edge_weight=a2a_edge_weights, normalization='rw')
            sparse_diag_block_L = block_diag_sparse(edge_index_L, Ls, batch)
            evals, U = torch.lobpcg(sparse_diag_block_L, k=self.config.training.k_eigenvalues, largest=False)
        
        for i in range(self.num_layers):
            # Interaction block
            a_x, m_x, m = self.sl_block[i](
                a_x=a_x,
                m_x=m_x,
                m=m,
                rbf3=rbf3,
                cbf3=cbf3,
                id3_ragged_idx=id3_ragged_idx,
                id_swap=id_swap,
                id3_ba=id3_ba,
                id3_ca=id3_ca,
                rbf_h=rbf_h,
                idx_s=idx_s,
                idx_t=idx_t,
                a2m_edge_index=a2m_edge_index, 
                m2a_edge_index=m2a_edge_index,
                a2m_edge_weights=a2m_edge_weights,
                m2a_edge_weights=m2a_edge_weights,
                a2m_edge_attr=a2m_edge_attr,
                m2a_edge_attr=m2a_edge_attr,
            ) 

            if self.use_phi_module:
                # Compute eigenbasis coefficients "alpha"
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
            net_charge = torch.abs(scatter_add(rho, batch, dim=0)).sum()
            pde_res += self.config.training.net_charge_lambda * net_charge
        else:
            pde_res = None
            
        out_a_edge = self.out_edge_norm(m)
        out_a_edge = self.a_mlp_rbf_out(rbf) * out_a_edge
        out_a_x = scatter(
            out_a_edge,
            idx_t,
            dim=0,
            dim_size=mesh_atom_graph['atom'].z.size(0),
            reduce="add",
        )
        out_m_x = self.out_m_norm(m_x)
        
        E_t = self.a_output(out_a_x)
        if self.extensive:
            energy_a = scatter(
                E_t, batch, dim=0, dim_size=bs, reduce="add"
            )  # (nMolecules, num_targets)
        else:
            energy_a = scatter(
                E_t, batch, dim=0, dim_size=bs, reduce="mean"
            )  # (nMolecules, num_targets)
        output_m_x = self.m_output(out_m_x)
        energy_m = torch.sum(output_m_x.reshape(bs, -1), dim=-1, keepdim=True)
        
        energy = energy_a + energy_m

        if self.use_phi_module:
            # Compute electrostatic energy term
            if self.epoch >= self.config.training.pde_warmup_epochs:
                self.electrostatic_term = 0.5 * (phi * rho).sum()
                E_t = E_t + self.electrostatic_offset * self.electrostatic_term + self.electrostatic_bias
            else:
                self.electrostatic_term = torch.tensor([0.0], dtype=torch.float32)
        
        if self.regress_forces:
            if self.direct_forces:
                # short forces
                a_F_st = self.a_out_forces(out_a_edge) # (nEdges, 1)
                a_F_st_vec = a_F_st[:, :, None] * a2a_normed_vec[:, None, :]
                a_F_t = scatter(
                    a_F_st_vec,
                    idx_t,
                    dim=0,
                    dim_size=mesh_atom_graph['atom'].atomic_numbers.size(0),
                    reduce="add",
                )  # (nAtoms, num_targets, 3)
                
                # long forces
                out_m_x_j = torch.index_select(out_m_x, 0, m2a_edge_index[0])
                m_F_st = self.m_mlp_rbf_out(m2a_edge_attr) * out_m_x_j
                m_F_st = self.m_out_forces(m_F_st) # (n_m2a_Edges, 1)
                
                m_F_st_vec = m_F_st[:, :, None] * m2a_normed_vec[:, None, :]
                m_F_t = scatter(
                    m_F_st_vec,
                    m2a_edge_index[1],
                    dim=0,
                    dim_size=mesh_atom_graph['atom'].atomic_numbers.size(0),
                    reduce="add",
                )
                forces = (a_F_t + m_F_t).squeeze(1)
            else:
                forces = -1 * (
                    torch.autograd.grad(
                        energy,
                        mesh_atom_graph['atom'].pos,
                        grad_outputs=torch.ones_like(energy),
                        create_graph=True,
                    )[0]
                )

            return ModelOutput(out=energy, forces=forces, pde_residual=pde_res) # energy, forces  # (nMolecules, num_targets), (nAtoms, 3)
        else:
            return ModelOutput(out=energy, pde_residual=pde_res) # (nMolecules, num_targets), None


class ShortLongMixLayer(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        edge_hidden_channels: int,
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
        self.m2a_edge_layernorm = nn.LayerNorm(edge_hidden_channels)
        self.short_layernorm = nn.LayerNorm(hidden_channels)
        self.m2a_edge_combine = nn.Linear(2 * hidden_channels, edge_hidden_channels)
        self.m2a_edge_act = nn.SiLU()
        self.short_edge_layernorm = nn.LayerNorm(edge_hidden_channels)
        self.long_layernorm = nn.LayerNorm(hidden_channels)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.a2m_mp.reset_parameters()
        self.m2a_mp.reset_parameters()
        self.short_mp.reset_parameters()
        self.long_mp.reset_parameters()
        self.a2m_layernorm.reset_parameters()
        self.m2a_edge_layernorm.reset_parameters()
        self.short_layernorm.reset_parameters()
        he_orthogonal_init(self.m2a_edge_combine.weight)
        self.m2a_edge_combine.bias.data.fill_(0)
        self.short_edge_layernorm.reset_parameters()
        self.long_layernorm.reset_parameters()
    
    def forward(
        self, 
        a_x,
        m_x,
        m,
        rbf3,
        cbf3,
        id3_ragged_idx,
        id_swap,
        id3_ba,
        id3_ca,
        rbf_h,
        idx_s,
        idx_t,
        a2m_edge_index, 
        m2a_edge_index,
        a2m_edge_weights,
        m2a_edge_weights,
        a2m_edge_attr,
        m2a_edge_attr,
    ):
        
        delta_m_x = m_x
        
        # N_atoms, F
        a_x = self.short_layernorm(a_x)
        m = self.short_edge_layernorm(m)
        a_x, m = self.short_mp(a_x, m, rbf3, cbf3, id3_ragged_idx, id_swap, id3_ba, id3_ca, rbf_h, idx_s, idx_t)        
        
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
        
        m2a_message_j = m2a_message.index_select(0, idx_s)
        m2a_message_i = m2a_message.index_select(0, idx_t)
        m2a_edge_message = self.m2a_edge_act(self.m2a_edge_combine(torch.cat([m2a_message_j, m2a_message_i], dim=-1)))
        m2a_edge_message = self.m2a_edge_layernorm(m2a_edge_message)
        
        return a_x, m_x + a2m_message + delta_m_x , m + m2a_edge_message
 