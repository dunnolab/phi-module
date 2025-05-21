import argparse

import torch
import torch.multiprocessing as mp

from data import load_config
from trainer import Trainer, set_seed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the configuration file")
    parser.add_argument("--seed", required=False, help='Random seed')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = int(args.seed)

    set_seed(config.seed)
    trainer = Trainer(config)
    trainer.run()
    