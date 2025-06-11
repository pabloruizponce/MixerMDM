import sys
sys.path.append(sys.path[0] + r"/../../")

import gc
import os
import time
import wandb
import torch
import argparse
import itertools
import torch.optim as optim
import lightning.pytorch as pl

from datasets import DataModule
from os.path import join as pjoin
from collections import OrderedDict
from utils.configs import get_config
from utils.utils import print_current_loss
from models.mixermdm import MixerMDM
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelSummary

# Set the environment variables
os.environ['PL_TORCH_DISTRIBUTED_BACKEND'] = 'nccl'
torch.set_float32_matmul_precision('medium')
wandb.require("core")

class LitTrainModel(pl.LightningModule):
    """
    Lightning model for training the model using the trainer (should be easier)
    """
    def __init__(self, model, cfg, model_cfg, only_discriminator=False, device=None):
        super().__init__()
        
        # Configs
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.automatic_optimization = False
        self.only_discriminator = only_discriminator
        self.device_eval = device
        
        # Paths
        self.save_root = pjoin(self.cfg.GENERAL.CHECKPOINT, self.cfg.GENERAL.EXP_NAME)
        self.model_dir = pjoin(self.save_root, 'model')
        self.meta_dir = pjoin(self.save_root, 'meta')
        self.log_dir = pjoin(self.save_root, 'log')
        self.pred_dir = pjoin(self.save_root, 'pred')

        # Create the directories for saving the checkpoints 
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.pred_dir, exist_ok=True)

        # Pytorch model
        self.model = model

        # Save hyperparameters
        self.save_hyperparameters()

    def _configure_optim(self):
        """
        Make some previous configurations to define the different optimizers of the model.
        In this version we have 2 optimizers (one for the generator and one for the discriminator)
        """
        # Get text conditiong parameters        
        text_params = itertools.chain(
            [self.model.positional_embedding],
            self.model.clipTransEncoder.parameters(),
            self.model.clip_ln.parameters(),
        )

        # Optimizer for the Gererator
        optimizer_generator = optim.AdamW(
            itertools.chain(
                text_params,
                self.model.mixing.influence.parameters(),
                self.model.mixing.sequence_pos_encoder.parameters(),
                self.model.mixing.embed_timestep.parameters(),
                self.model.mixing.motion_embed.parameters(),
                self.model.mixing.text_embed.parameters(),
            ),
            lr=float(self.cfg.TRAIN.LR), 
            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY
        )

        # Optimizer for the Discriminator
        optimizer_discriminator = optim.AdamW(
            itertools.chain(
                self.model.discriminator_i.parameters(),
                self.model.discriminator_I.parameters()
            ),
            lr=float(self.cfg.TRAIN.LR), 
            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY
        )
        return [optimizer_generator, optimizer_discriminator]

    def configure_optimizers(self):
        """
        Call the private method to configure the optimizers
        """
        return self._configure_optim()
    
    def on_train_start(self):
        """
        Initialize the variables for the training
        """
        self.rank = 0
        self.world_size = 1
        self.start_time = time.time()
        self.it = self.cfg.TRAIN.LAST_ITER if self.cfg.TRAIN.LAST_ITER else 0
        self.epoch = self.cfg.TRAIN.LAST_EPOCH if self.cfg.TRAIN.LAST_EPOCH else 0
        self.logs = OrderedDict()

    def forward(self, batch_data, mode):
        """
        Forward pass of the model. 
        As this models wraper is used for training only, this method is used to compute the loss
            :param batch_data: Data of the batch
            :param mode: Modo of the training (generator or discriminator)
        """
        batch = OrderedDict({})

        # Load in the batch the condition data
        batch["text_interaction"] = batch_data[1]
        batch["text_individual1"] = batch_data[5]
        batch["text_individual2"] = batch_data[6]

        # Load in the batch the motions data
        batch["motions"] = torch.cat([batch_data[2], batch_data[3]], dim=-1)

        # Load in the batch 
        batch["motion_lens"] = batch_data[4]

        loss, loss_logs, influence_history = self.model.compute_loss(
            batch, 
            mode, 
            i_loss_factor=self.cfg.TRAIN.INDIVIDUAL_LOSS_FACTOR, 
            I_loss_factor=self.cfg.TRAIN.INTERACTION_LOSS_FACTOR,
            l1=self.cfg.TRAIN.LOSS_L1,
        )
        return loss, loss_logs, influence_history

    
    def training_step(self, batch, batch_idx):
        """
        Training step of the model
            :param batch: Data of the batch
            :param batch_idx: Index of the batch
        """
        opt_gen, opt_dis = self.optimizers()

        # Generator step
        loss_gen, loss_logs_gen, influence_history_g = self.forward(batch, "generator")
        loss_gen = loss_gen / self.cfg.TRAIN.GRAD_ACC_STEPS
        self.manual_backward(loss_gen)
                
        if (batch_idx+1) % self.cfg.TRAIN.GRAD_ACC_STEPS == 0:
            self.clip_gradients(opt_gen, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            opt_gen.step()
            opt_gen.zero_grad()
                
        # Discriminator step
        if (batch_idx+1) % self.cfg.TRAIN.DISCRIMINATOR_STEPS == 0:
            loss_dis, loss_logs_dis, influence_history_d = self.forward(batch, "discriminator")
            loss_dis = loss_dis / self.cfg.TRAIN.GRAD_ACC_STEPS
            self.manual_backward(loss_dis)
      
            if (batch_idx+1) % (self.cfg.TRAIN.GRAD_ACC_STEPS * self.cfg.TRAIN.DISCRIMINATOR_STEPS) == 0:
                self.clip_gradients(opt_dis, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
                opt_dis.step()
                opt_dis.zero_grad()
            
            influence_history = {
                "influence_i1": torch.cat(
                    [
                        influence_history_g['influence_i1'][0], 
                        influence_history_d['influence_i1'][0]
                    ], 
                    dim=0
                ).mean(),
                "influence_i2": torch.cat(
                    [
                        influence_history_g['influence_i2'][0], 
                        influence_history_d['influence_i2'][0]
                    ], 
                    dim=0
                ).mean(),
            }

            # Return the losses
            loss = loss_gen + loss_dis
            loss_logs = {**loss_logs_gen, **loss_logs_dis, **influence_history}
            loss_logs = {**loss_logs_gen, **loss_logs_dis, **influence_history}
        else:
            influence_history = {
                "influence_i1": influence_history_g['influence_i1'][0].mean(),
                "influence_i2": influence_history_g['influence_i2'][0].mean(),
            }
            loss = loss_gen
            loss_logs = {**loss_logs_gen, **influence_history}

        return {
            "loss": loss,
            "loss_logs": loss_logs
        }
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        Method to be called at the end of the batch
            :param outputs: Outputs of the training step (losses)
            :param batch: Batch data
            :param batch_idx: Index of the batch
        """
        # If the batch is skipped or the loss logs are empty, return
        if outputs.get('skip_batch') or not outputs.get('loss_logs'):
            return
        
        # Update the logs
        for k, v in outputs['loss_logs'].items():
            if k not in self.logs:
                self.logs[k] = v.item()
            else:
                self.logs[k] += v.item()

        # If the iteration is multiple of the log steps, print the current loss and log it into the wandb
        self.it += 1
        if self.it % self.cfg.TRAIN.LOG_STEPS == 0:

            # Compute the mean loss
            mean_loss = OrderedDict({})
            for tag, value in self.logs.items():
                if "discriminator" in tag:
                    mean_loss[tag] = value / (self.cfg.TRAIN.LOG_STEPS / self.cfg.TRAIN.DISCRIMINATOR_STEPS)
                else:
                    mean_loss[tag] = value / self.cfg.TRAIN.LOG_STEPS

            # Log the loss into the wandb
            wandb.log(mean_loss)

            # Reset the logs
            self.logs = OrderedDict()

            # Print the current loss
            print_current_loss(self.start_time, self.it, mean_loss,
                               self.trainer.current_epoch,
                               inner_iter=batch_idx,
                               lr=self.trainer.optimizers[0].param_groups[0]['lr'])

    def on_train_epoch_end(self):
        """
        Method to be called at the end of the epoch
        """
        # Update the lr schedulers (if any)
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()

    def save(self, file_name):
        """
        Save the model in a file. (I think this is not being used at the time)
        """
        state = {}
        try:
            state['model'] = self.model.module.state_dict()
        except:
            state['model'] = self.model.state_dict()
        torch.save(state, file_name, _use_new_zipfile_serialization=False)
        return

def list_of_ints(arg):
    """
    Method to conver a string of integers separated by commas to a list of integers.
    Used in the argparse to get the device to use
    """
    return list(map(int, arg.split(',')))

def get_trainable_layers(model):
    """
    Get the name of the trainable layers of the model an return them as a list
    """
    trainable_layers = [name for name, param in model.named_parameters() if param.requires_grad]
    return trainable_layers
    
if __name__ == '__main__':

    # Create the parser
    parser = argparse.ArgumentParser(description="Argparse example with optional arguments")

    # Add arguments
    parser.add_argument('--train', type=str, required=True, help='Training Configuration file')
    parser.add_argument('--model' , type=str, required=True, help='Model Configuration file')
    parser.add_argument('--data', type=str, required=True, help='Data Configuration file')
    parser.add_argument('--resume', type=str, required=False, help='Resume training from checkpoint')
    parser.add_argument('--device', type=list_of_ints, required=True, help='Device to run the training')

    # Parse the arguments
    args = parser.parse_args()

   # Show gpu that is going to be used
    print("Device to use:", args.device) 
    device = torch.device('cuda:%d' % args.device[0] if torch.cuda.is_available() else 'cpu')

    # Load the configuration files
    model_cfg = get_config(args.model)
    train_cfg = get_config(args.train)
    interhuman_cfg = get_config(args.data).interhuman

    # Load dataset
    datamodule = DataModule(interhuman_cfg, train_cfg.TRAIN.BATCH_SIZE, train_cfg.TRAIN.NUM_WORKERS)

    # Load model
    model = MixerMDM(model_cfg)
    litmodel = LitTrainModel(model, train_cfg, model_cfg, only_discriminator=train_cfg.TRAIN.ONLY_DISCRIMINATOR, device=device)

    # Checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=litmodel.model_dir,
                                                       every_n_epochs=train_cfg.TRAIN.SAVE_EPOCH,
                                                       save_top_k=-1)

    # Assume `lightning_model` is your PyTorch Lightning model
    trainable_layers = get_trainable_layers(litmodel)

    # Logger
    wandb_logger = WandbLogger(project="PinacoladaV4", name=train_cfg.GENERAL.EXP_NAME)

    # Trainer
    trainer = pl.Trainer(
        default_root_dir=litmodel.model_dir,
        devices=args.device, accelerator='gpu',
        max_epochs=train_cfg.TRAIN.EPOCH,
        strategy=DDPStrategy(find_unused_parameters=True),
        precision='16-mixed',
        logger=wandb_logger,
        callbacks=[ModelSummary(3), checkpoint_callback],
    )

    # Train or resume
    if args.resume:
        trainer.fit(model=litmodel, datamodule=datamodule, ckpt_path=args.resume)
    else:
        trainer.fit(model=litmodel, datamodule=datamodule)