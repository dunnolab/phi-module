import wandb
import argparse

from data import load_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the configuration file")
    parser.add_argument("--sweep_config", required=True, help="Path to the sweep configuration file (optional)")
    args = parser.parse_args()

    config = load_config(args.config)
    sweep_config = load_config(args.sweep_config)

    sweep_id = wandb.sweep(sweep_config, project=config.logging.project_name)
    print(sweep_id)