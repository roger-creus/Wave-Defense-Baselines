import os
import yaml
import argparse
import numpy as np
from pathlib import Path
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
import wandb

from src.representations.models.VAE import VanillaVAE_PL
from  src.representations.main.vae_experiment import VAEXperiment

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default="src/config/vae.yml")

args = parser.parse_args()

with open("src/config/" + args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

wandb_logger = WandbLogger(
    project='WaveDefense',
    name=config['model_params']['name'],
    tags=['vae']
)

wandb_logger.log_hyperparams(config)

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

model = VanillaVAE_PL(**config['model_params'])
experiment = VAEXperiment(model, config['exp_params'])

runner = Trainer(logger=wandb_logger,
                 accelerator='gpu',
                 enable_checkpointing = True,
                 devices=1,
                 #default_root_dir = os.environ["SLURM_TMPDIR"] + "/tmp/checkpoints/",
                 strategy=DDPPlugin(find_unused_parameters=True),
                 **config['trainer_params'])

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment)