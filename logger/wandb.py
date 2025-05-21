import wandb


class WandbLogger():
    def __init__(self, project_name, run_name=None, group=None, dir=None, config=None):
        self.project_name = project_name
        wandb.init(project=project_name, name=run_name, config=config, group=group, dir=dir, reinit=True)

    def log(self, metrics: dict):
        wandb.log(metrics)

    def finish_wandb(self):
        wandb.finish()
        print("Wandb run finished.")

    