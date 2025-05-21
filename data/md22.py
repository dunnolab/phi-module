import numpy as np

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class MD22Dataset(Dataset):
    def __init__(self, npz_path, data_indices, mean=None, std=None):
        trajectories = np.load(npz_path)
        
        self.z = torch.from_numpy(trajectories['z']).long()
        self.pos = torch.from_numpy(trajectories['R']).float()[data_indices]
        self.energy = torch.from_numpy(trajectories['E'])[data_indices]
        self.force = torch.from_numpy(trajectories['F']).float()[data_indices]

        if mean is not None or std is not None:
            self.energy = (self.energy - mean) / std

    def __getitem__(self, idx):
        data = Data(z=self.z, pos=self.pos[idx], y=self.energy[idx], forces=self.force[idx])

        return data

    def __len__(self):
        return len(self.pos)