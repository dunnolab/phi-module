import os
import time
import wandb
import random
import gc
import datetime
import numpy as np
import warnings
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Subset, random_split
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import optim
from torch.utils.data import DistributedSampler
from torch_geometric.loader import DataLoader as GraphDataLoader
from torch_geometric.nn.models import SchNet

from models import MODEL_REGISTRY 
from logger import BasicLogger, WandbLogger
from data import load_config, QM9Dataset, OE62Dataset, MD22Dataset
from training_utils import WarmupScheduler, EarlyStopping, boltzmann_sampling

warnings.simplefilter('always', UserWarning)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    

class Trainer():
    def __init__(self, config):
        self.setup_group()
        self.device = torch.device(f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu')

        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Device is {self.device}')
         
        if config.task == 'nbody':
            # N-body problem will be added later

            raise NotImplementedError
        elif config.task == 'qm9':
            if self.world_size > 1:
                raise ValueError('Multi-GPU training is not supported for QM9. Use a single GPU, man, the dataset is kinda small')

            train_dataset = QM9Dataset(root=config.data.root, split='train', 
                                       target_property=config.data.target_property,
                                       mean=self.config.data.mean, std=self.config.data.std)
            val_dataset = QM9Dataset(root=config.data.root, split='val', 
                                     target_property=config.data.target_property,
                                     mean=self.config.data.mean, std=self.config.data.std)
            test_dataset = QM9Dataset(root=config.data.root, split='test', 
                                      target_property=config.data.target_property,
                                      mean=self.config.data.mean, std=self.config.data.std)
            
            self.train_dataloader = GraphDataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
            self.val_dataloader = GraphDataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)
            self.test_dataloader = GraphDataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)
            
            self.loss_function = F.l1_loss
            self.metric_function = None

            self.model = self.build_model()
        elif config.task == 'oe62':
            train_dataset = OE62Dataset(self.config.data.train_path, mean=self.config.data.mean, std=self.config.data.std)
            val_dataset = OE62Dataset(self.config.data.val_path, mean=self.config.data.mean, std=self.config.data.std)
            test_dataset = OE62Dataset(self.config.data.test_path, mean=self.config.data.mean, std=self.config.data.std)
           
            self.train_dataloader = GraphDataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
            self.val_dataloader = GraphDataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)
            self.test_dataloader = GraphDataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)

            self.loss_function = F.l1_loss
            self.metric_function = None

            self.model = self.build_model()
        elif config.task == 'md22':
            
            molecule_sizes = {
                'Ac-Ala3-NHMe': [85109, 6000],
                'DHA': [69753, 8000],
                'stachyose': [27272, 8000],
                'AT-AT': [20001, 3000],
                'AT-AT-CG-CG': [10153, 2000],
                'buckyball-catcher': [6102, 600],
                'dw_nanotube': [5032, 800]
            }

            mol_trajectories = np.load(os.path.join(self.config.data.npz_path, f'md22_{self.config.data.molecule}.npz'))
            mol_energies = mol_trajectories['E']
    
            # Split train, val, test according to original MD22 paper
            md22_num_train = int(molecule_sizes[self.config.data.molecule][1] * 0.95)
            md22_num_val = molecule_sizes[self.config.data.molecule][1] - md22_num_train
            md22_num_test = molecule_sizes[self.config.data.molecule][0] - md22_num_train - md22_num_val

            # md22_train_indices, md22_val_indices, md22_test_indices = boltzmann_sampling(mol_energies, md22_num_train, md22_num_val, T=300)
            md22_train_indices, md22_val_indices, md22_test_indices = random_split(range(molecule_sizes[self.config.data.molecule][0]), \
                                                                          [md22_num_train, md22_num_val, md22_num_test], \
                                                                          generator=torch.Generator().manual_seed(self.config.seed))

            npz_file_path = os.path.join(self.config.data.npz_path, f'md22_{self.config.data.molecule}.npz')

            train_dataset = MD22Dataset(npz_file_path, data_indices=md22_train_indices, mean=self.config.data.mean, std=self.config.data.std)
            val_dataset = MD22Dataset(npz_file_path, data_indices=md22_val_indices, mean=self.config.data.mean, std=self.config.data.std)
            test_dataset = MD22Dataset(npz_file_path, data_indices=md22_test_indices, mean=self.config.data.mean, std=self.config.data.std)

            train_sampler = DistributedSampler(train_dataset, shuffle=True) if self.world_size > 1 else None
            val_sampler = DistributedSampler(val_dataset, shuffle=False) if self.world_size > 1 else None
            test_sampler = DistributedSampler(test_dataset, shuffle=False) if self.world_size > 1 else None

            self.train_dataloader = GraphDataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=(train_sampler is None), sampler=train_sampler)
            self.val_dataloader = GraphDataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False, sampler=val_sampler)
            self.test_dataloader = GraphDataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False, sampler=test_sampler)

            self.loss_function = F.l1_loss
            self.force_loss_function = F.l1_loss
            self.metric_function = None

            self.model = self.build_model()
            if self.world_size > 1:
                self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
        else:
            raise NotImplementedError
        
        print(f'Effective batch size: {self.config.training.batch_size} * {self.world_size} = {self.config.training.batch_size * self.world_size}')
        print(f'Total number of parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        
        if config.training.weight_decay is not None:
            self.optimizer = optim.AdamW(self.model.parameters(), lr=config.training.lr, 
                                    weight_decay=config.training.weight_decay)
        else:
            self.optimizer = optim.AdamW(self.model.parameters(), lr=config.training.lr)
        
        # Create scheduler
        if config.task in ['qm9', 'oe62', 'md22']:
            if self.config.training.warmup_epochs is not None:
                base_scheduler =  optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.training.epochs, eta_min=1e-7)
                # base_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=5000, T_mult=2)
                self.scheduler = WarmupScheduler(optimizer=self.optimizer, base_scheduler=base_scheduler, 
                                                 warmup_epochs=self.config.training.warmup_epochs)
            else:
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.training.epochs, eta_min=1e-7)
        else:
            self.scheduler = None
        
        # Create early stopping tracker
        if self.config.training.early_stopping_patience is not None:
            self.early_stopping = EarlyStopping(patience=self.config.training.early_stopping_patience, 
                                           delta=self.config.training.early_stopping_delta)
        else:
            self.early_stopping = EarlyStopping(patience=float('inf'),
                                           delta=self.config.training.early_stopping_delta)

        # Setup logging
        if self.local_rank == 0:
            if config.logging.logger == 'basic':
                self.logger = BasicLogger(name=config.logging.run_name)
            elif config.logging.logger == 'wandb':
                self.logger = WandbLogger(project_name=config.logging.project_name, run_name=config.logging.run_name, 
                                        group=config.logging.group, dir=config.logging.output_dir, config=config)
            else:
                raise ValueError('Unknown Logger Type')
        else:
            self.logger = None
        
        os.makedirs(self.config.logging.checkpoints_path, exist_ok=True)

    def setup_group(self):
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))

        if self.world_size > 1:
            dist.init_process_group(backend='nccl', timeout=datetime.timedelta(minutes=30))
            torch.cuda.set_device(self.local_rank)

    def cleanup_group(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def build_model(self):
        if self.config.model.type not in MODEL_REGISTRY:
            raise ValueError(f'Model type {self.config.model.type } is not supported')
        
        return MODEL_REGISTRY[self.config.model.type](self.config).to(self.device)
        
    def run(self):
        best_val_loss = float('inf')
        
        for epoch in range(self.config.training.epochs): 
            self.current_epoch = epoch
        
            if self.world_size > 1:  
                self.model.module.epoch = epoch
            else:
                self.model.epoch = epoch

            train_loss = self.train()
            val_loss = self.validate()

            if self.local_rank == 0:
                self.logger.log({'epoch': epoch})

            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                
                if self.local_rank == 0:
                    torch.save({
                        'epoch': epoch,                        
                        'model_state_dict': self.model.state_dict(),  
                        'optimizer_state_dict': self.optimizer.state_dict(),  
                        'train_loss': train_loss,
                        'val_loss': val_loss,                          
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    }, os.path.join(self.config.logging.checkpoints_path, 'best.ckpt'))

            if self.local_rank == 0:
                torch.save({
                    'epoch': epoch,                        
                    'model_state_dict': self.model.state_dict(),  
                    'optimizer_state_dict': self.optimizer.state_dict(),  
                    'train_loss': train_loss,
                    'val_loss': val_loss,                          
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }, os.path.join(self.config.logging.checkpoints_path, 'last.ckpt'))

            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print(f'Early stopping has happened at epoch {epoch}')
                break
            
        test_loss = self.test()
        print(f'Test loss {test_loss}')

        del self.model
        del self.optimizer
        del self.train_dataloader
        del self.val_dataloader
        del self.test_dataloader
        torch.cuda.empty_cache()
        gc.collect()

        if self.local_rank == 0:
            if self.config.logging.logger == 'wandb':
                self.logger.finish_wandb()

        self.cleanup_group()

    def prepare_batch(self, batch):    
        if self.config.task in ['qm9', 'oe62', 'md22']:
            if self.config.model.type == 'linear':
                one_hot = F.one_hot(batch.z.long() - 1, num_classes=self.config.model.input_dim).float().to(self.device) 
                bag_of_atoms = torch.zeros((len(torch.unique(batch.batch)), self.config.model.input_dim), dtype=torch.float, device=self.device)

                bag_of_atoms.index_add_(0, batch.batch.to(self.device), one_hot)

                return bag_of_atoms
            else:
                batch.z = batch.z.to(self.device) 
                batch.pos = batch.pos.to(self.device) 
                batch.batch = batch.batch.to(self.device) 

                if self.config.task == 'md22': 
                    batch.n_atoms = torch.tensor([len(batch.batch[batch.batch == 0]) for _ in range(len(batch.y))], dtype=torch.long).to(self.device)

                return batch
        else:
            raise NotImplementedError
    
    def forward_pass(self, batch):
        if self.config.model.type == 'linear':
            bag_of_atoms = self.prepare_batch(batch)

            model_out = self.model(bag_of_atoms)
        else:
            model_out = self.model(self.prepare_batch(batch))

        if model_out.out.ndim == 2 and model_out.out.shape[1] == 1:
            model_out.out = model_out.out.squeeze(1)
        elif model_out.out.ndim == 2 and model_out.out.shape[0] == 1:
            model_out.out = model_out.out.squeeze(0)

        return model_out
            
    def compute_loss(self, model_out, label, label_forces=None):     
        loss = self.loss_function(model_out.out, label)
        
        if model_out.pde_residual is not None:
            loss = loss + self.config.training.pde_lambda * model_out.pde_residual

        if model_out.forces is not None:
            if label_forces is None:
                warnings.warn('True forces were not passed to compute_loss() method. Now, only energy loss is computed')

                return loss

            loss_forces = self.force_loss_function(model_out.forces, label_forces)

            # For further logging
            self.energy_loss = loss.item()
            self.forces_loss = loss_forces.item()
            
            if self.config.training.energy_lambda is None or self.config.training.force_lambda is None:
                raise ValueError("E or F scaling parameter is None")
            
            loss = self.config.training.energy_lambda * loss + self.config.training.force_lambda * loss_forces

        return loss

    def train(self):
        self.model.train()

        if self.world_size > 1:
            self.train_dataloader.sampler.set_epoch(self.current_epoch)

        total_loss = 0.0
        total_norm = 0.0
        
        for step, batch in enumerate(tqdm(self.train_dataloader)): 
            self.step = step
            if 'forces' in batch.keys():
                label_forces = batch.forces.to(self.device)
            else:
                label_forces = None
    
            self.optimizer.zero_grad()

            label = batch.y.to(self.device)
            model_out = self.forward_pass(batch)
            
            train_loss = self.compute_loss(model_out, label, label_forces=label_forces) 
            train_loss.backward()

            if self.config.data.mean is not None and self.config.data.std is not None and self.config.task != 'md22':
                train_loss = train_loss * self.config.data.std + self.config.data.mean

            if self.config.training.clipping is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.clipping)

            batch_grad_norm = self.get_grad_norm()
            total_norm += batch_grad_norm

            self.optimizer.step()

            total_loss += train_loss.item()

            if self.local_rank == 0:
                if not isinstance(self.logger, BasicLogger):
                    self.logger.log({'train_step': step, 'train_loss': train_loss.item(), 'train_loss': train_loss.item(),
                                    'grad_norm': batch_grad_norm})
                    
                    if self.config.task == 'md22':
                        self.logger.log({'train_energy_loss': self.energy_loss, 'train_forces_loss': self.forces_loss})
                    
                    if self.config.model.use_phi_module:
                        if self.world_size > 1:
                            res_offset = self.model.module.electrostatic_offset.item()
                            res_bias = self.model.module.electrostatic_bias.item()
                            res_E = self.model.module.electrostatic_term.item()
                        else:
                            res_offset = self.model.electrostatic_offset.item()
                            res_bias = self.model.electrostatic_bias.item()
                            res_E = self.model.electrostatic_term.item()

                        self.logger.log({'res_offset': res_offset, 
                                        'res_bias': res_bias, 
                                        'res_E': res_E})
                elif isinstance(self.logger, BasicLogger) and step % self.config.logging.log_interval_steps == 0:
                    self.logger.log({'train_step': step, 'train_loss': train_loss.item(), 'train_loss': train_loss.item(),
                                    'grad_norm': batch_grad_norm, 'learning_rate': self.optimizer.param_groups[0]['lr']})
                    
                    if self.config.model.use_phi_module:
                        if self.world_size > 1:
                            res_offset = self.model.module.electrostatic_offset.item()
                            res_bias = self.model.module.electrostatic_bias.item()
                            res_E = self.model.module.electrostatic_term.item()
                        else:
                            res_offset = self.model.electrostatic_offset.item()
                            res_bias = self.model.electrostatic_bias.item()
                            res_E = self.model.electrostatic_term.item()

                        self.logger.log({'res_offset': res_offset, 
                                        'res_bias': res_bias, 
                                        'res_E': res_E})
            
        if self.scheduler is not None:
            self.scheduler.step()
        
        if self.local_rank == 0:
            if not isinstance(self.logger, BasicLogger):
                self.logger.log({'learning_rate': self.optimizer.param_groups[0]['lr']})
        
        return train_loss / len(self.train_dataloader)
            
    def validate(self):
        self.model.eval()
        total_val_loss = 0.0
        total_val_metric = 0.0
        
        for step, batch in enumerate(tqdm(self.val_dataloader)):
            label = batch.y.to(self.device)
            if 'forces' in batch.keys():
                label_forces = batch.forces.to(self.device)
            else:
                label_forces = None

            if 'forces' in batch.keys():
                model_out = self.forward_pass(batch)
            else:
                with torch.no_grad():
                    model_out = self.forward_pass(batch)
        
            val_loss = self.compute_loss(model_out, label, label_forces=label_forces)

            if self.config.data.mean is not None and self.config.data.std is not None and self.config.task != 'md22':
                val_loss = val_loss * self.config.data.std + self.config.data.mean

            if torch.isnan(val_loss):
                raise ValueError('NaN val loss encountered!')
            
            total_val_loss += val_loss.item()

            if self.metric_function is not None:
                metric = self.metric_function(model_out.out.cpu().detach().numpy(), label.cpu().detach().numpy())
                total_val_metric += metric

        if self.local_rank == 0:
            if self.metric_function is not None:
                self.logger.log({'val_loss': total_val_loss / len(self.val_dataloader), 'val_metric': total_val_metric / len(self.val_dataloader)})
            else:
                self.logger.log({'val_loss': total_val_loss / len(self.val_dataloader)})

            if self.config.task == 'md22':
                self.logger.log({'val_energy_loss': self.energy_loss, 'val_forces_loss': self.forces_loss})

        return total_val_loss / len(self.val_dataloader)
    
    def test(self):
        self.model.eval()
        total_test_loss = 0.0
        total_test_metric = 0.0
        
        for step, batch in enumerate(tqdm(self.test_dataloader)):
            label = batch.y.to(self.device)
            if 'forces' in batch.keys():
                label_forces = batch.forces.to(self.device)
            else:
                label_forces = None
            
            if 'forces' in batch.keys():
                model_out = self.forward_pass(batch)
            else:
                with torch.no_grad():
                    model_out = self.forward_pass(batch)

            test_loss = self.compute_loss(model_out, label, label_forces=label_forces) 

            if self.config.data.mean is not None and self.config.data.std is not None and self.config.task != 'md22':
                test_loss = test_loss * self.config.data.std + self.config.data.mean

            total_test_loss += test_loss.item()

            if self.metric_function is not None:
                metric = self.metric_function(model_out.cpu().detach().numpy(), label.cpu().detach().numpy())
                total_test_metric += metric

        if self.local_rank == 0:
            if self.metric_function is not None:
                if not isinstance(self.logger, BasicLogger):
                    self.logger.log({'test_loss': total_test_loss / len(self.test_dataloader), 'test_metric': total_test_metric / len(self.test_dataloader)})
                elif isinstance(self.logger, BasicLogger) and step % self.config.logging.log_interval_steps == 0:
                    self.logger.log({'test_loss': total_test_loss / len(self.test_dataloader), 'test_metric': total_test_metric / len(self.test_dataloader)})
            else:
                if not isinstance(self.logger, BasicLogger):
                    self.logger.log({'test_loss': total_test_loss / len(self.test_dataloader)})
                elif isinstance(self.logger, BasicLogger) and step % self.config.logging.log_interval_steps == 0:
                    self.logger.log({'test_loss': total_test_loss / len(self.test_dataloader)})

            if self.config.task == 'md22':
                self.logger.log({'test_energy_loss': self.energy_loss, 'test_forces_loss': self.forces_loss})
            
        return total_test_loss / len(self.test_dataloader)
    
    def get_grad_norm(self):
        batch_grad_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                batch_grad_norm += param_norm.item() ** 2
        batch_grad_norm  = batch_grad_norm ** 0.5
        
        return batch_grad_norm
    