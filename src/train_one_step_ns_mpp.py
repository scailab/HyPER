from pathlib import Path

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torch.optim import Adam
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError
from einops import rearrange

import utils
from data.navier_stokes_dataset import NavierStokesDataset

# imports for multiple physics pretraining model
from models.avit import build_avit
from YParams import YParams

from dataclasses import dataclass, asdict
from typing import Literal
import tyro

# setup logging
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter
formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
log.addHandler(ch)

@dataclass(frozen=True)
class TrainConfig:
    """ Code to finetune MPP on Navier-Stokes dataset """

    # Output directory which will get created if it doesn't exist.
    output_dir: str
    # Path to dataset.
    dataset_path: Path = Path("../data/ns_data_64_1000.h5")
    # Dataset split percentages for train, rl, validation, and test sets.
    dataset_split: tuple[float, float, float, float] = (0.40, 0.40, 0.10, 0.10)
    # Random seed.
    random_seed: int = 42
    # Device to run on.
    device: str = "cuda"
    # Batch size.
    batch_size: int = 8
    # Start timestep.
    ST: int = 8
    # End timestep.
    FT: int = 28
    # Learning rate for Adam optimizer
    lr: float = 0.0001
    # Path to MPP model yaml config file.
    mpp_config: Path = Path("./conf/mpp_avit_s_config.yaml")
    # Path to trained surrogate model checkpoint.
    checkpoint_path: Path = Path("../model_checkpoints/MPP_AViT_S")
    # Number of epochs to train
    epochs: int = 200
    # Use training and RL datasets for training (800 trajectories instead of 400).
    use_large_dataset: bool = True
    # Finetune existing model
    finetune_model: bool = False
    # Job name
    job_name: str = f"train-mpp-s-800-lr-{lr}-ns-64x64-bs-8-epochs-{epochs}-1"

def train(config: TrainConfig):
    log.info(f"Starting MPP finetuning with config {config}")
    # create output directory if it doesn't exist
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=False)
    dev = torch.device(config.device)
    # load dataset
    log.info(f"Loading dataset at {config.dataset_path}")
    dataset = NavierStokesDataset(config.dataset_path, provide_velocity=True)
    train_dataset, rl_dataset, val_dataset, test_dataset = random_split(dataset, config.dataset_split, generator=torch.Generator().manual_seed(42))
    large_dataset = ConcatDataset((train_dataset, rl_dataset))
    # load model config
    mpp_config = YParams(config.mpp_config, "basic_config", False)
    model = build_avit(mpp_config)
    # load model checkpoint for MPP
    if config.finetune_model == False:
        checkpoint = torch.load(config.checkpoint_path, map_location="cuda")
        model.load_state_dict(checkpoint)
    elif config.finetune_model == True:
        log.info(f"Loading model at {config.checkpoint_path} for finetuning")
        model = utils.load_model(config.checkpoint_path, model)
    model = model.to(dev)
    model.train()
    optimizer = Adam(model.parameters(), lr=config.lr)
    # dataset statistics for 64x64:
    smoke_mean = 0.7322910149367649
    smoke_std = 0.43668617344401106
    smoke_min = 4.5986561758581956e-07
    smoke_max = 2.121663808822632
    # setup dataloader
    train_dataloader = DataLoader(train_dataset, config.batch_size, shuffle=True, pin_memory=True)
    large_dataloader = DataLoader(large_dataset, config.batch_size, shuffle=True, pin_memory=True)
    if config.use_large_dataset:
        log.info("Using large dataset")
        dataloader = large_dataloader
    else:
        log.info("Using small dataset")
        dataloader = train_dataloader
    log.info(f"Dataloader len: {len(dataloader)}")
    metrics = MetricCollection([
        MeanSquaredError(), MeanAbsoluteError(), MeanAbsolutePercentageError()
    ])
    train_metrics = metrics.clone(prefix="train_").to(dev)
    total_mse = 0.0
    for epoch in tqdm(range(config.epochs)):
        for i, batch in enumerate(tqdm(dataloader, desc="batch")):
            sim_ids = batch["sim_id"].to(dev)
            smoke = batch["smoke"].unsqueeze(2).to(dev)
            # standardize data
            smoke = (smoke - smoke_mean) / smoke_std
            for t in range(config.ST, config.FT):
                inputs = smoke[:, t-1, ...].unsqueeze(1)
                inputs = rearrange(inputs, "b t c h w -> t b c h w")
                # set field labels and boundary conditions
                field_labels = torch.tensor([[0]]).to(dev)
                bcs = torch.as_tensor([[0, 0]]).to(dev)
                output = model(inputs, field_labels, bcs)
                # set ground truth to next timestep smoke field
                gt = smoke[:, t, :, :, :]
                # compute loss, note that we unstandardize the data here to get back to the scale of the original data
                loss = F.mse_loss((output * smoke_std) + smoke_mean, (gt * smoke_std) + smoke_mean)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_mse += loss
                mets = train_metrics((output * smoke_std) + smoke_mean, (gt * smoke_std) + smoke_mean)
            batch_mets = train_metrics.compute()
        epoch_mets = train_metrics.compute()
        # log training metrics
        log.info(f"Epoch {epoch} training metrics:")
        log.info(f"Train Loss: {total_mse / len(dataloader)}")
        log.info(f"Train MSE: {epoch_mets['train_MeanSquaredError']}")
        log.info(f"Train MAE: {epoch_mets['train_MeanAbsoluteError']}")
        log.info(f"Train MAPE: {epoch_mets['train_MeanAbsolutePercentageError']}")
        # reset metric states after each epoch
        train_metrics.reset()
        total_mse = 0.0
    # save final model checkpoint
    utils.save_model_checkpoint(epoch, config.device, model, optimizer, output_path, config.job_name, suffix="final")

if __name__ == "__main__":
    config = tyro.cli(TrainConfig)
    train(config)

