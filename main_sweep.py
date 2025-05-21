import wandb
import random
import argparse

from data import load_config
from trainer import Trainer, set_seed


def run(config, sweep_config):
    config['logging']['run_name'] = f"sweep_{random.randrange(1000000, 9999999)}"
    
    # Replace config values with sweep parameters
    for param in sweep_config.keys():
        if param == 'config_file_name':
            continue

        if not (param in config.training or param in config.model):
            raise ValueError('Sweep parameter not found in original config or the names do not match')
        
        if param in config.training:
            config['training'][param] = sweep_config[param]
        elif param in config.model:
            config['model'][param] = sweep_config[param]

    set_seed(config.seed)
    trainer = Trainer(config)
    trainer.run()


if __name__ == '__main__':
    wandb.login()
    wandb.init()

    sweep_config = wandb.config
    config = load_config(sweep_config['config_file_name'])

    run(config, sweep_config)
    

    