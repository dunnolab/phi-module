import json
import lmdb
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset


class OE62Dataset(Dataset):
    def __init__(self, lmdb_path, mean=None, std=None):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with self.env.begin() as txn:
            self.length = txn.stat()['entries']
            self.keys = [key.decode() for key, _ in txn.cursor()]

            self.all_values = [pickle.loads(value).y for _, value in txn.cursor()]
        
        self.mean, self.std = mean, std

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            key = f'{self.keys[idx]}'.encode('ascii')
            data = pickle.loads(txn.get(key))

            if self.mean is not None and self.std is not None:
                data.y = (data.y - self.mean) / self.std

        return data

    def __len__(self):
        return self.length