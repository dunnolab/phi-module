import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gc
from ase import Atoms
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch_geometric.data import Data

from trainer import load_config, set_seed
from models.schnet import SchNetBase
from models.schnet_ewald import SchNetEwald
from models.schnet_p3m import SchNet_P3M


def make_dummy_chain_graph(N, spacing=1.5):
    pos = torch.tensor([[i * spacing, 0.0, 0.0] for i in range(N)], dtype=torch.float)
    z = torch.full((N,), fill_value=6, dtype=torch.long)

    edge_index = []
    for i in range(N - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i]) 

    return Data(z=z, pos=pos, y_relaxed=torch.zeros_like(z), batch=torch.zeros_like(z).long())


def compute_memory_consumption(model_type):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_type == "baseline":
        model_config = load_config("configs/schnet/schnet_oe62.yml")
        model_config["model"]["use_phi_module"] = False

        model = SchNetBase(model_config).to(device)
    elif model_type == "phi-module":
        model_config = load_config("configs/schnet/schnet_oe62.yml")

        model_config["training"]["k_eigenvalues"] = 9  
        model_config["model"]["cutoff"] = 6.0
        model_config["model"]["use_phi_module"] = True

        model = SchNetBase(model_config).to(device)
        model.epoch = 1
    elif model_type == "ewald":
        model_config = load_config("configs/schnet/schnet_ewald_oe62.yml")
        model_config["model"]["use_phi_module"] = False

        model = SchNetEwald(model_config).to(device)
    elif model_type == 'p3m':
        model_config = load_config("configs/schnet/schnet_p3m_oe62.yml")
        model_config["model"]["use_phi_module"] = False

        model = SchNet_P3M(model_config).to(device)

    memory_consumption = []
    print(f"Model Type = {model_type}")
    for n in range(1000, 100000, 1000):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        polymer = make_dummy_chain_graph(N=n)
        polymer = polymer.to(device)

        out = model(polymer)
        memory_consumption.append(torch.cuda.max_memory_allocated() / 1024**2)

        print(f"Number of atoms: {len(polymer.z)}, Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MiB")

        del polymer
        gc.collect()

    print(memory_consumption)


def visualize_memory_consumtion():
    custom_palette = ["#2f3677ff", "#9fa4d4ff", "#4049a3ff", "#717171ff", "#676fbcff", "#4b4b4bff"]

    N_atoms = np.arange(1000, 100000, 1000)

    # Precomputed values 
    baseline_memory = [178.2265625, 446.7548828125, 715.14208984375, 982.998046875, 1250.03662109375, 1517.41552734375, 1784.62744140625, 2031.61328125, 
                       2276.8515625, 2538.765625, 2800.8779296875, 3063.4697265625, 3327.56884765625, 3589.66748046875, 3853.0712890625, 4121.0234375, 
                       4388.5009765625, 4657.11865234375, 4924.86181640625, 5191.529296875, 5457.94921875, 5706.7509765625, 5952.13916015625, 6214.68994140625, 
                       6476.48876953125, 6738.943359375, 7001.87939453125, 7264.23388671875, 7528.90087890625, 7795.4091796875, 8061.77734375, 8330.75146484375, 
                       8598.50927734375, 8867.3330078125, 9134.45703125, 9380.57373046875, 9626.9541015625, 9888.93359375, 10150.82177734375, 10416.0625, 10677.97265625, 
                       10939.6416015625, 11202.84716796875, 11468.9345703125, 11736.2177734375, 12003.07763671875, 12271.84765625, 12540.330078125, 12804.37646484375, 
                       13053.662109375, 13301.75146484375, 13565.93505859375, 13828.18603515625, 14090.35107421875, 14353.77490234375, 14617.357421875, 14881.4052734375, 
                       15144.3193359375, 15410.9443359375, 15678.2509765625, 15946.5498046875, 16214.07373046875, 16479.6494140625, 16730.4951171875, 16979.29638671875, 
                       17241.15576171875, 17501.44921875, 17764.86474609375, 18028.64794921875, 18290.95068359375, 18553.119140625, 18818.779296875, 19084.7958984375, 
                       19351.31591796875, 19619.3349609375, 19887.5244140625, 20153.59765625, 20421.1484375, 20670.6298828125, 20914.60546875, 21177.744140625, 
                       21440.12744140625, 21702.6513671875, 21964.3251953125, 22227.8056640625, 22492.5517578125, 22758.29296875, 23026.63232421875, 23294.3125, 
                       23562.08740234375, 23829.78857421875, 24097.38232421875, 24345.623046875, 24591.56689453125, 24852.7861328125, 25115.921875, 25379.53955078125, 
                       25640.7001953125, 25903.33154296875]

    phi_module_memory = [182.6201171875, 457.9365234375, 731.982421875, 1007.41455078125, 1283.08056640625, 1560.18505859375, 1835.47900390625, 2085.01708984375, 
                         2340.1494140625, 2609.02392578125, 2876.37744140625, 3150.220703125, 3424.1923828125, 3692.78662109375, 3961.76904296875, 4235.279296875, 
                         4512.43505859375, 4784.95458984375, 5060.08837890625, 5332.08984375, 5607.2509765625, 5862.10302734375, 6116.98486328125, 6383.748046875, 
                         6653.1259765625, 6925.7373046875, 7192.93603515625, 7459.5341796875, 7731.77001953125, 8009.18359375, 8286.04345703125, 8556.39208984375, 
                         8831.7978515625, 9107.4765625, 9381.1259765625, 9636.94775390625, 9891.455078125, 10160.607421875, 10429.2724609375, 10700.48876953125, 
                         10973.55029296875, 11242.1220703125, 11510.84619140625, 11782.5380859375, 12057.01806640625, 12333.43359375, 12606.884765625, 12881.263671875, 
                         13154.599609375, 13411.6318359375, 13667.23095703125, 13934.650390625, 14204.91845703125, 14473.828125, 14743.60205078125, 15016.2548828125, 
                         15289.2734375, 15557.24853515625, 15828.236328125, 16104.9892578125, 16381.052734375, 16655.314453125, 16929.85791015625, 17185.68115234375, 
                         17437.18408203125, 17708.92822265625, 17980.80712890625, 18248.97119140625, 18518.6171875, 18788.87353515625, 19059.021484375, 19332.845703125, 
                         19608.5634765625, 19880.24658203125, 20152.08984375, 20424.92041015625, 20698.1669921875, 20973.57763671875, 21231.798828125, 21485.251953125, 
                         21753.4296875, 22024.1279296875, 22294.86572265625, 22565.43603515625, 22833.60009765625, 23104.36328125, 23377.38330078125, 23653.5966796875, 
                         23928.82470703125, 24202.16552734375, 24476.05517578125, 24753.2294921875, 25008.9462890625, 25261.10595703125, 25530.533203125, 25800.11083984375, 
                         26068.77734375, 26335.98681640625, 26605.4912109375]

    ewald_memory = [577.818359375, 1532.263671875, 2475.86572265625, 3436.59130859375, 4389.02392578125, 5335.884765625, 6294.79052734375, 7263.7080078125, 8211.947265625, 
                    9153.9580078125, 10116.3974609375, 11068.04150390625, 12012.408203125, 12974.59521484375, 13930.005859375, 14874.970703125, 15835.10302734375, 
                    16800.14013671875, 17745.6591796875, 18692.52685546875, 19659.31591796875, 20615.66796875, 21559.47119140625, 22517.63037109375, 23468.748046875, 
                    24421.595703125, 25384.65625, 26330.5, 27278.7197265625, 28236.04248046875, 29200.56005859375, 30151.10595703125, 31094.8818359375, 32057.93115234375, 
                    33007.560546875, 33954.9677734375, 34916.31640625, 35866.10107421875, 36813.95361328125, 37772.455078125, 38735.90771484375, 39682.27685546875, 
                    40630.82763671875, 41595.9248046875, 42548.89990234375, 43492.64599609375, 44454.33154296875, 45405.955078125, 46353.21875, 47314.802734375, 48266.8154296875, 
                    49214.1513671875, 50172.91357421875, 51135.40625, 52083.76171875, 53033.8359375, 53994.326171875, 54942.5478515625, 55890.94140625, 56853.38525390625,
                    57805.5068359375, 58754.0048828125, 59710.751953125, 60673.490234375, 61620.98828125, 62569.8447265625, 63532.228515625, 64483.060546875, 65430.44091796875, 
                    66389.859375, 67341.85400390625, 68290.599609375, 69248.47021484375, 70203.583984375, 71150.95849609375, 72108.36328125]
    
    neural_p3m_memory = [324.63134765625, 752.55712890625, 1356.70166015625, 2205.53466796875, 3231.8642578125, 4445.7109375, 5847.58837890625, 7425.41357421875, 9190.21142578125, 
                         11136.23779296875, 13272.2294921875, 15580.666015625, 18080.59033203125, 20764.3720703125, 23618.892578125, 26671.8779296875, 29895.3056640625,
                           33306.548828125, 36903.10595703125, 40680.90087890625, 44639.31298828125, 48784.32568359375, 53112.85205078125, 57624.1884765625, 62317.2744140625, 67195.6455078125]

    N_atoms_ewald = N_atoms[:len(ewald_memory)]
    N_atoms_p3m = N_atoms[:len(neural_p3m_memory)]

    plt.figure(figsize=(10, 6))

    plt.plot(N_atoms, baseline_memory, label="Baseline", color=custom_palette[0], linewidth=4)
    plt.plot(N_atoms, phi_module_memory, label=r"$\boldsymbol{\Phi}\mathbf{-Module}$", color=custom_palette[3], linewidth=4.5)
    plt.plot(N_atoms_ewald, ewald_memory, label="Ewald", color=custom_palette[1], linewidth=4, linestyle='--')
    plt.plot(N_atoms_p3m, neural_p3m_memory, label="Neural P3M", color=custom_palette[2], linewidth=4, linestyle='--')

    # Highlight the OOM termination points
    plt.scatter(N_atoms[len(ewald_memory)-1], ewald_memory[-1],
            color=custom_palette[1], edgecolor='black', s=350, marker='X', zorder=5, label="OOM Point (Ewald)")
    plt.scatter(N_atoms[len(neural_p3m_memory)-1], neural_p3m_memory[-1],
            color=custom_palette[2], edgecolor='black', s=350, marker='X', zorder=5, label="OOM Point (Neural P3M)")

    plt.xlabel("Number of Atoms", fontsize=18)
    plt.ylabel("CUDA Memory Consumption (MB)", fontsize=18)
    plt.legend(fontsize=15, loc='upper right', bbox_to_anchor=(1, 0.85))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tick_params(axis='both', which='major', labelsize=17)

    ax = plt.gca()        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig("plots/memory.pdf", dpi=300)

    plt.show()



if __name__ == '__main__':
    model_type = "p3m" # baseline, phi-module, ewald, p3m
    # compute_memory_consumption(model_type)

    visualize_memory_consumtion()

