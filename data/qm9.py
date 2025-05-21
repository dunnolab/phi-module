import numpy as np

from torch.utils.data import Dataset
from torch_geometric.datasets import QM9


targets_dict = {
            'dipole': 0,
            'alpha': 1,
            'homo': 2,
            'lumo': 3,
            'gap': 4,
            'r2': 5,
            'zpve': 6,
            'u0': 7,
            'u': 8,
            'h': 9,
            'g': 10,
            'c_v': 11,
            'u0_atom': 12,
            'u_atom': 13,
            'h_atom': 14,
            'g_atom': 15,
            'a': 16,
            'b': 17,
            'c': 18
        }

normalization_dict = {
            'dipole': [2.6729531, 1.503479],
            'alpha': [75.28118, 8.173830],
            'homo': [-6.536452, 0.5977412],
            'lumo': [0.32204, 1.27485],
            'gap': [6.8584918, 1.284168],
            'r2': [1189.41064, 280.478149],
            'zpve': [4.0569372, 0.9017231],
            'u0': [-11178.9667, 1085.57873],
            'u': [-11178.7353, 1085.572753],
            'h': [-11178.70996, 1085.57275],
            'g': [-11179.87695, 1085.5924072],
            'c_v': [31.6203632, 4.067580],
            'u0_atom': [-76.116012, 10.323753],
            'u_atom': [-76.580490, 10.415176],
            'h_atom': [-77.018257, 10.4892702],
            'g_atom': [-70.836662, 9.498342],
            'a': [9.9660215, 1830.4630126],
            'b': [1.406728, 1.6008282],
            'c': [1.1273994, 1.107471]
        }


class QM9Dataset(Dataset):
    def __init__(self, root, split='train', target_property='alpha', mean=None, std=None):
        self.qm9 = QM9(root=root)

        self._random_state = np.random.RandomState(seed=999)
        all_idx = self._random_state.permutation(np.arange(self.qm9.__len__()))

        if split == 'train':
            self.qm9 = self.qm9[all_idx[:100000]]
        elif split == 'val':
            self.qm9 = self.qm9[all_idx[100000:100000 + 17748]]
        elif split == 'test':
            self.qm9 = self.qm9[all_idx[100000 + 17748:]]
        else:
            raise ValueError('Unknown data split')

        self.target_property = target_property
        self.mean = mean
        self.std = std
    
    def __getitem__(self, idx):
        sample = self.qm9[idx]
        sample.y = sample.y[:, targets_dict[self.target_property]]

        if self.mean is not None and self.std is not None:
            sample.y = (sample.y - self.mean) / self.std

        return sample
    
    def __len__(self):
        return self.qm9.__len__()
    



