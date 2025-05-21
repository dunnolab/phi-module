from torch.optim import Optimizer
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler

import numpy as np
import inspect
from bisect import bisect


def boltzmann_sampling(energies, num_train, num_val, T=300):
    ''' MD Trajectories Sampling with Boltzmann Distribution '''

    k_B = 1.38e-23

    energy_shift = np.min(energies) 
    rescaled_energies = energies - energy_shift

    boltzmann_weights = np.exp(-rescaled_energies / (k_B * T)).squeeze(1) + 1e-12
    boltzmann_weights /= np.sum(boltzmann_weights) 
    
    train_indices = np.random.choice(energies.shape[0], size=num_train, replace=False, p=boltzmann_weights)

    remaining_indices = list(set(range(energies.shape[0])) - set(train_indices))
    remaining_weights = boltzmann_weights[remaining_indices]
    remaining_weights /= np.sum(remaining_weights) 
    val_indices = np.random.choice(remaining_indices, size=num_val, replace=False, p=remaining_weights)

    test_indices = list(set(remaining_indices) - set(val_indices))

    return train_indices, val_indices, test_indices


class EarlyStopping:
    def __init__(self, patience, delta=0.001):
        self.patience = patience
        self.delta = delta

        self.best_val_loss = float('inf')
        self.counter = 0

        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss + self.delta < self.best_val_loss:
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True


class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer: Optimizer, base_scheduler: _LRScheduler, warmup_epochs: int = 5, warmup_steps: int = None, last_epoch: int = -1):
        self.base_scheduler = base_scheduler
        self.warmup_epochs = warmup_epochs
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.in_warmup = True

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.in_warmup:
            if self.warmup_steps is not None:
                warmup_factor = (self.current_step + 1) / self.warmup_steps
            else:
                warmup_factor = (self.last_epoch + 1) / self.warmup_epochs

            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            return self.base_scheduler.get_last_lr()

    def step(self, epoch=None):
        if self.in_warmup:
            if self.warmup_steps is not None:
                self.current_step += 1
                if self.current_step >= self.warmup_steps:
                    self.in_warmup = False
                    self.base_scheduler.step(epoch)
            else:
                if self.last_epoch + 1 >= self.warmup_epochs:
                    self.in_warmup = False
                    self.base_scheduler.step(epoch)
        else:
            self.base_scheduler.step(epoch)

        self.last_epoch += 1
        self._set_lr(self.optimizer, self.get_lr())

    @staticmethod
    def _set_lr(optimizer, lrs):
        for param_group, lr in zip(optimizer.param_groups, lrs):
            param_group['lr'] = lr

    def state_dict(self):
        state = super().state_dict()

        state['base_scheduler'] = self.base_scheduler.state_dict()
        state['current_step'] = self.current_step
        state['in_warmup'] = self.in_warmup

        return state

    def load_state_dict(self, state_dict):
        self.current_step = state_dict.pop('current_step', 0)
        self.in_warmup = state_dict.pop('in_warmup', True)
        self.base_scheduler.load_state_dict(state_dict.pop('base_scheduler', {}))

        super().load_state_dict(state_dict)
