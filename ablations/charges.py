import os
import sys 
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GraphDataLoader

# import psi4
import numpy as np
import pandas as pd
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem

import seaborn as sns
import matplotlib.pyplot as plt

from trainer import load_config, set_seed
from data import OE62Dataset
from models.dpp.dimenetpp import DimeNetPlusPlusBase
from models.painn.painn import PaiNN
from models.schnet import SchNetBase
from models.gemnet.gemnet import GemNetT


symbols_map = {
    1: 'H',
    3: 'Li',
    6: 'C',
    7: 'N',
    8: 'O',
    9: 'F',
    5: 'B',
    14: 'Si',
    15: 'P',
    16: 'S',
    17: 'Cl',
    33: 'As',
    34: 'Se',
    35: 'Br',
    52: 'Te',
    53: 'I',
}


def ase_to_pyg_data(ase_atoms):
    atomic_numbers = ase_atoms.get_atomic_numbers()   
    positions = ase_atoms.get_positions()             
    
    z = torch.tensor(atomic_numbers, dtype=torch.long)
    pos = torch.tensor(positions, dtype=torch.float)
    batch = torch.zeros(len(ase_atoms), dtype=torch.long) 
    
    data = Data(z=z, pos=pos, batch=batch)

    return data


def rdkit_to_ase(rdmol):
    """Convert an RDKit molecule with 3D coordinates to an ASE Atoms object."""
    atoms = []
    conf = rdmol.GetConformer()
    for atom in rdmol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        atoms.append((atom.GetSymbol(), (pos.x, pos.y, pos.z)))
    symbols, positions = zip(*atoms)

    return Atoms(symbols=symbols, positions=positions)


def mol2xyz(mol, multiplicity=1):
    charge = Chem.GetFormalCharge(mol)
    xyz_string = "\n{} {}\n".format(charge, multiplicity)
    for atom in mol.GetAtoms():
        pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
        xyz_string += "{} {} {} {}\n".format(atom.GetSymbol(), pos.x, pos.y, pos.z)
    return xyz_string


def clone_mol_data(data):
    N = 6

    data.z = data.z.repeat(N)
    data.pos = data.pos.repeat(N, 1)
   
    batch_clones = [data.batch + i for i in range(N)]
    data.batch = torch.cat(batch_clones)

    print(batch_clones)

    return data


def compute_and_save_charge_data(method='phi-module'):
    model_configs = {
        'schnet': 'configs/schnet/schnet_oe62.yml',
        'dpp': 'configs/dimenetpp/dimenetpp_oe62.yml',
        'painn': 'configs/painn/painn_oe62.yml',
        'gemnet': 'configs/gemnet/gemnet_t_oe62.yml'
    }
    checkpoint_paths = {
        'schnet': '../weights/schnet.ckpt',
        'dpp': '../dpp-ablations-phi-module/last.ckpt',
        'painn': '../weights/painn.ckpt',
        'gemnet': '../weights/gemnet.ckpt'
    }

    model_config = load_config(model_configs[method])

    checkpoint_path = checkpoint_paths[method]

    model_config['training']['k_eigenvalues'] = 9  
    model_config['model']['cutoff'] = 6.0
    model_config['model']['use_phi_module'] = True
    if method == 'painn':
        model_config['model']['scale_file'] = 'scaling_factors_oe62/painn_baseline.pt'
    elif method =='gemnet':
        model_config['model']['scale_file'] = 'scaling_factors_oe62/gemnet_baseline.json'

    set_seed(model_config['seed'])

    if method == 'schnet':
        model = SchNetBase(model_config)
    elif method == 'dpp':
        model = DimeNetPlusPlusBase(model_config)
    elif method == 'painn':
        model = PaiNN(model_config)
    elif method == 'gemnet':
        model = GemNetT(model_config)

    model.epoch = 0
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['model_state_dict'], strict=False)
    model.eval()

    test_dataset = OE62Dataset('../../datasets/OE62/total_energy_lincorr_pbe0/test/pbe0_test.mdb', 
                               mean=0.0036029790876720654, 
                               std=1.742015096746636)   
    test_dataloader = GraphDataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=True)
    
    atomic_charges = {
        '1': 0,
        '6': 0,
        '7': 0,
        '8': 0,
        '9': 0,
    }
    counts = {
        '1': 0,
        '6': 0,
        '7': 0,
        '8': 0,
        '9': 0,
    }

    # Get Hirshfeld data
    oe62 = pd.read_json('../../datasets/OE62/df_62k.json', orient='split')
    oe62 = oe62.reset_index(drop=True)

    refcodes = oe62['refcode_csd']
    hirshfeld_charges = oe62['hirshfeld_pbe']
    hirshfeld_charges_by_code = {oe62['refcode_csd'][x]: oe62['hirshfeld_pbe'][x] for x in range(len(oe62))}

    # Gather atomic charges and charges sum
    net_rho_values = []
    for i, data in enumerate(tqdm(test_dataloader)):
        if method != 'hirshfeld':
            with torch.no_grad():
                out, rho = model(data) # if you want to retrive charges, please add "rho" to the return of the selected model
        else:
            rho = torch.tensor(hirshfeld_charges_by_code[data['refcode_csd'][0]], dtype=torch.float32) # ensure bs == 1
    
        for atom_idx in atomic_charges.keys():
            if not torch.isnan(rho[data.z.view(-1) == float(atom_idx)].mean()):
                atomic_charges[atom_idx] += rho[data.z.view(-1) == float(atom_idx)].mean()

            counts[atom_idx] += rho[data.z.view(-1) == float(atom_idx)].shape[0]

        net_rho_values.append(rho.sum().item())

    print(atomic_charges)
    print(counts)
    
    # Get average values
    for atom_idx in atomic_charges.keys():
        atomic_charges[atom_idx] /= counts[atom_idx]
        atomic_charges[atom_idx] = atomic_charges[atom_idx].item()

    print(atomic_charges)

    # Get Pearson Correlation against electronegativity
    electronegativity = {
        "H": 2.20, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98, 
    }
    en_values = list(electronegativity.values()) 
    en_charge_corr = np.corrcoef(en_values, list(atomic_charges.values()))[0, 1]
    print(en_charge_corr)

    # Plots
    avg_charge = np.array(list(atomic_charges.values()))  

    np.save(f'ablations/results/avg_charge_{method}', np.array(avg_charge))
    np.save(f'ablations/results/all_charges_{method}', np.array(net_rho_values))


def get_charge_plots(model='schnet'):
    elements = ["H", "C", "N", "O", "F",] 
    en = np.array([2.20, 2.55, 3.04, 3.44, 3.98,])

    avg_charge_hirshfeld = np.load('ablations/results/avg_charge_hirshfeld.npy')
    net_rho_hirshfeld_values = np.load('ablations/results/all_charges_hirshfeld.npy')
    
    avg_charge_phi_module = np.load(f'ablations/results/avg_charge_{model}.npy')
    net_rho_phi_module_values = np.load(f'ablations/results/all_charges_{model}.npy')

    palette = ["#2f3677ff", "#9fa4d4ff", "#d56131ff",  "#4049a3ff", "#676fbcff", "#4b4b4bff"]

    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 20, # 13,
        "axes.linewidth": 1.2,
        "axes.labelsize": 20, # 14,
        "axes.titlesize": 20, # 15,
        "xtick.labelsize": 17, # 12,
        "ytick.labelsize": 17, # 12,
        "legend.fontsize": 17, # 12,
        "grid.linewidth": 0.5,
        "lines.linewidth": 2,
    })

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    (ax1, ax2), (ax3, ax4) = axes

    # (a) Hirshfeld: EN vs Avg Charge
    ax1.scatter(en, avg_charge_hirshfeld, color=palette[3], edgecolor=palette[5], s=130, zorder=3)
    for i, label in enumerate(elements):
        ax1.text(en[i], avg_charge_hirshfeld[i], label, fontsize=18, ha='right', va='bottom', color=palette[5])
    ax1.set_xlabel("Electronegativity (Pauling)")
    ax1.set_ylabel("Average Hirshfeld Charge")
    ax1.set_title("(a) Hirshfeld", pad=30)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # (b) Phi_module: EN vs Avg Charge
    ax2.scatter(en, avg_charge_phi_module, color=palette[0], edgecolor=palette[5], s=130, zorder=3)
    for i, label in enumerate(elements):
        ax2.text(en[i], avg_charge_phi_module[i], label, fontsize=17, ha='right', va='bottom', color=palette[5])
    ax2.set_xlabel("Electronegativity (Pauling)")
    ax2.set_ylabel("Average Predicted Charge")
    # ax2.set_title("(b) Phi-DimeNet++", pad=10)
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax2.yaxis.offsetText.set_fontsize(17)
    ax2.yaxis.offsetText.set_x(-0.1)
    ax2.set_title(r'(b) $\boldsymbol{\Phi}$-GemNet-T', pad=30)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # (c) Hirshfeld: Net Charge Histogram
    ax3.hist(net_rho_hirshfeld_values, bins=20, color=palette[4], edgecolor=palette[5], linewidth=1)
    ax3.set_xlabel("Net Charge")
    ax3.set_ylabel("Number of Molecules")
    # ax3.set_title("(c) Hirshfeld Charge Conservation", pad=10)
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3f}"))
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # (d) Phi_module: Net Charge Histogram
    ax4.hist(net_rho_phi_module_values, bins=20, color=palette[1], edgecolor=palette[5], linewidth=1)
    ax4.set_xlabel("Net Predicted Charge")
    ax4.set_ylabel("Number of Molecules")
    # ax4.set_title(r'(d) $\boldsymbol{\Phi}$-DimeNet++ Charge Conservation', pad=10)
    ax4.grid(True, linestyle='--', alpha=0.5)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"plots/charge_comparison_{model}.pdf", format='pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # To compute charge statistics
    compute_and_save_charge_data(method='schnet')

    # To get visualizations
    # get_charge_plots(model='schnet')

