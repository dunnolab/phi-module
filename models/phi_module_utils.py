import torch
from torch import nn
from torch_scatter import scatter_add


class AlphaNet(nn.Module):
    def __init__(self, in_channels, k):
        super().__init__()

        self.embed_dim = 64
        
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.SiLU(),
            nn.Conv1d(in_channels=32, out_channels=self.embed_dim, kernel_size=5, stride=1, padding=2),
            nn.SiLU()
        )

        self.global_pool = nn.AdaptiveAvgPool1d(k) 
        self.final_fc = nn.Linear(self.embed_dim, 1)

        self.k = k

    def forward(self, x):
        x = x.T.unsqueeze(0) # [N, F] -> [1, F, N]

        x = self.net(x) # [1, F, N] -> [1, E, N]
        x = self.global_pool(x) # [1, E, N] -> [1, E, K]

        x = x.squeeze(0).T # [1, E, K] -> [K, N]
        alpha = self.final_fc(x) # [K, N] -> [K, 1]

        return alpha
    

def laplacian_matvec(L, phi, edge_indices):
    """
    Sparse mat-vec multiplication: L @ phi

    Parameters
    ----------
    L : torch.Tensor
        A 1D tensor of shape (E,) containing the Laplacian values 
        for each edge in the graph. Each L[e] corresponds to 
        the (row, col) = (edge_indices[0,e], edge_indices[1,e]) 
        entry in the Laplacian.
    phi : torch.Tensor
        A 1D tensor of shape (N,) representing the vector we want 
        to multiply by L. N is the total number of nodes.
    edge_indices : Tuple[torch.Tensor, torch.Tensor]
        A pair (row, col) of shape (E,) each, where row[e] indicates 
        the row index (in [0..N-1]) and col[e] indicates the column 
        index for the e-th edge or Laplacian entry.

    Returns
    -------
    torch.Tensor
        A 1D tensor of shape (N, 1) containing the product L @ phi.
    """

    row, col = edge_indices
    
    L_phi = L.unsqueeze(-1) * phi[col]
    L_phi = scatter_add(L_phi, row, dim=0)

    return L_phi
    

def block_diag_sparse(edge_indices, L, batch):
    """
    Compute block-diagonal sparse Laplacian from vectorized Laplacian values and edge indices

    Parameters
    ----------
    edge_indices : Tuple[torch.Tensor, torch.Tensor]
        A pair (src, dst) of shape (E,) each, indicating 
        edge (src[i], dst[i]). Typically, E is the total 
        number of edges before filtering by graph.
    L : torch.Tensor
        A 1D tensor of shape (E,) containing the edge weight 
        or Laplacian contribution for each edge. Must match 
        the length of src and dst.
    batch : torch.Tensor
        A 1D tensor of shape (N,) where N is the total number of nodes 
        across the batch. batch[i] indicates which graph node i belongs to.

    Returns
    -------
    torch.sparse_coo_tensor
        A coalesced sparse COO tensor (N x N) containing block-diagonal 
        Laplacian entries. Cross-graph edges are excluded. Each block 
        corresponds to one graph in the batch.

    """

    N = batch.size(0)

    src, dst = edge_indices
    same_graph_mask = (batch[src] == batch[dst])
    src = src[same_graph_mask]
    dst = dst[same_graph_mask]
    w = L[same_graph_mask]
    
    # For an undirected Laplacian, each edge contributes 4 entries:
    #   (i, j) += -w, (j, i) += -w, (i, i) += +w, (j, j) += +w
    i2 = torch.cat([src,   dst,   src,   dst], dim=0)
    j2 = torch.cat([dst,   src,   src,   dst], dim=0)
    v2 = torch.cat([-w,    -w,     w,     w ], dim=0)
    
    indices_2d = torch.stack([i2, j2], dim=0)
    L = torch.sparse_coo_tensor(indices_2d, v2, size=(N, N))

    L = L.coalesce()

    return L