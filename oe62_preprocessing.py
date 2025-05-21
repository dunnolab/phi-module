import os
import lmdb
import json
import pickle
from tqdm import tqdm
from collections import Counter
from io import StringIO
import pandas as pd
from sklearn.linear_model import LinearRegression

import torch
import numpy as np
from ase.io import read
from torch_geometric.data import Data


def build_lmdb(data: pd.DataFrame, data_path: str, dataset_name: str, split_name: str):
    ''' Generate LMDB datafile for selected dataset and path configuration '''

    os.makedirs(os.path.join(data_path, dataset_name, split_name), exist_ok=True)

    db = lmdb.open(os.path.join(data_path, dataset_name, split_name, f'pbe0_{split_name}.mdb'),
                   map_size=1099511627776 * 2, subdir=True, meminit=False, map_async=False)

    with db.begin(write=True) as txn:
        for idx, row in tqdm(data.iterrows(), total=len(data)):
            atom_string = StringIO(row["xyz_pbe_relaxed"])

            xyz_data = read(atom_string, format='xyz')
            atomic_numbers = torch.Tensor(xyz_data.get_atomic_numbers())
            pos = torch.Tensor(xyz_data.get_positions()) 

            n_atoms = pos.shape[0]
            fixed = torch.zeros(n_atoms, dtype=torch.float32)
            structure_id = idx

            refcode_csd = row['refcode_csd']
            target_relaxed = targets_relaxed[idx]

            data_sample = Data(pos=pos, z=atomic_numbers, n_atoms=n_atoms, structure_id=structure_id, 
                            refcode_csd=refcode_csd, fixed=fixed, y=target_relaxed)
            
            key_sample = f"{idx}".encode("ascii")
            value_sample = pickle.dumps(data_sample, protocol=-1)
            
            txn.put(key_sample, value_sample)

    db.sync()
    db.close()


if __name__ == '__main__':
    oe62 = pd.read_json('../data/OE62/df_62k.json', orient='split')
    oe62 = oe62.reset_index(drop=True)

    # Retrieve all atom types
    all_symbols = []
    for _, row in tqdm(oe62.iterrows(), total=len(oe62)):
        atom_string = StringIO(row["xyz_pbe_relaxed"])
        xyz_data = read(atom_string, format='xyz')
        symbols = xyz_data.get_chemical_symbols()

        for s in symbols:
            if s not in all_symbols:
                all_symbols.append(s)

    symbol_map = dict(zip(all_symbols, range(len(all_symbols))))

    # Modify target energies by removing offset using linear regression
    regression_samples = []
    targets = []
    for _, row in tqdm(oe62.iterrows(), total=len(oe62)):
        atom_string = StringIO(row["xyz_pbe_relaxed"])
        xyz_data = read(atom_string, format='xyz')

        symbols = xyz_data.get_chemical_symbols()
        symbol_counts = Counter(symbols)
        regression_sample = [0] * 16

        for key, val in symbol_counts.items():
            regression_sample[symbol_map[key]] = val

        regression_samples.append(regression_sample)
        targets.append(-row['total_energy_pbe0_vac_tier2']) 

    regression_samples = np.array(regression_samples, dtype=float)
    targets = np.array(targets)

    offset_model = LinearRegression(positive=True).fit(regression_samples, targets)
    targets_relaxed = -targets + offset_model.predict(regression_samples)

    # Build LMBDs
    data_path = '../data/OE62/'
    dataset_name = 'total_energy_lincorr_pbe0'
    os.makedirs(os.path.join(data_path, dataset_name), exist_ok=True)

    oe62 = oe62.sample(frac=1, random_state=42)
    train_size = 50000
    val_size = 6000
    
    print('Building LMDB data splits...')
    build_lmdb(data=oe62[:train_size], data_path=data_path, dataset_name=dataset_name, split_name='train')
    build_lmdb(data=oe62[train_size:train_size + val_size], data_path=data_path, dataset_name=dataset_name, split_name='val')
    build_lmdb(data=oe62[train_size + val_size:], data_path=data_path, dataset_name=dataset_name, split_name='test')

    # Save offset model info
    reg_info = {"Atom to Coefficient Mapping": dict(zip(all_symbols, range(len(all_symbols)))),
                "Regression Coefficients": list(offset_model.coef_),
                "Regression Intercept": float(offset_model.intercept_)}
    
    with open(os.path.join(data_path, dataset_name, "offset_fitting_params_pbe0.json"), "w") as f:
        json.dump(reg_info, f)
    



        




