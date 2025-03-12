from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError

import utils
from data.navier_stokes_dataset import NavierStokesDataset
from models.twod_unet_cond import Unet as UnetCond

from dataclasses import dataclass
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
    """ Code to train multistep UNet on Navier-Stokes dataset """

    # Output directory which will get created if it doesn't exist.
    output_dir: str
    # Path to dataset.
    dataset_path: Path = Path("../data/ns_data_64_1000.h5")
    # Dataset split percentages for train, rl, validation, and test sets.
    dataset_split: tuple[float, float, float, float] = (0.40, 0.40, 0.10, 0.10)
    # Random seed.
    random_seed: int = 42
    # Device to run on.
    device: Literal["cuda", "cpu"] = "cuda"
    # Batch size.
    batch_size: int = 8
    # Start timestep.
    ST: int = 8
    # End timestep.
    FT: int = 28
    # Learning rate for Adam optimizer
    lr: float = 0.0001
    # Number of epochs to train
    epochs: int = 200
    # Use training and RL datasets for training (800 trajectories instead of 400)
    use_large_dataset: bool = True
    # Finetune existing model
    finetune_model_path: Path | None = None
    # Job name
    job_name: str = f"train-multistep-20-unet-800-lr-{lr}-ns-64x64-bs-8-epochs-{epochs}-2"

def train(config: TrainConfig):
    log.info(f"Training multistep UNet model with config {config}")
    # create output directory if it doesn't exist
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=False)
    dev = torch.device(config.device)
    # load dataset
    log.info(f"Loading dataset at path {config.dataset_path}")
    dataset = NavierStokesDataset(config.dataset_path, provide_velocity=True)
    train_dataset, rl_dataset, val_dataset, test_dataset = random_split(dataset, config.dataset_split, generator=torch.Generator().manual_seed(42))
    large_dataset = ConcatDataset((train_dataset, rl_dataset))
    # create unet model
    model = UnetCond(1, 0, 1, 0, 1, 1, 64, "gelu", ch_mults=[1, 2, 2], is_attn=[False, False, False]).to(dev)
    if config.finetune_model_path is not None:
        log.info(f"Loading model at {config.finetune_model_path} for finetuning")
        model = utils.load_model(config.finetune_model_path, model).to(dev)
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
        log.info(f"Using large dataset of size {len(large_dataset)}")
        dataloader = large_dataloader
    else:
        log.info(f"Using small dataset of size {len(train_dataset)}")
        dataloader = train_dataloader
    T = config.FT - config.ST
    metrics = MetricCollection([
        MeanSquaredError(), MeanAbsoluteError(), MeanAbsolutePercentageError()
    ])
    train_metrics = metrics.clone(prefix="train_").to(dev)
    val_metrics = metrics.clone(prefix="val_").to(dev)
    total_mse = 0.0
    for epoch in tqdm(range(config.epochs)):
        for i, batch in enumerate(tqdm(dataloader, desc="batch")):
            sim_ids = batch["sim_id"].to(dev)
            smoke = batch["smoke"].unsqueeze(2).to(dev)
            # standardize data
            smoke = (smoke - smoke_mean) / smoke_std
            prev_pred = smoke[:, config.ST-1, :, :, :].unsqueeze(1)
            # train all 20 timesteps at once
            losses = []
            for t in range(config.ST, config.FT):
                inputs = prev_pred
                gt = smoke[:, t, :, :, :].unsqueeze(1)
                time = torch.full((inputs.shape[0],), t, device=dev)
                output = model(inputs, time)
                prev_pred = output
                # calculate loss on original scale
                losses.append(F.mse_loss((output * smoke_std) + smoke_mean, (gt * smoke_std) + smoke_mean))
                train_metrics((output * smoke_std) + smoke_mean, (gt * smoke_std) + smoke_mean)
            # calculate mean loss and update model parameters
            loss = torch.mean(torch.stack(losses))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_mets = train_metrics.compute()
        # log training metrics
        log.info(f"Epoch {epoch} training metrics:")
        log.info(f"Train Loss: {total_mse / len(dataloader)}")
        log.info(f"Train MSE: {epoch_mets['train_MeanSquaredError']}")
        log.info(f"Train MAE: {epoch_mets['train_MeanAbsoluteError']}")
        log.info(f"Train MAPE: {epoch_mets['train_MeanAbsolutePercentageError']}")
        # reset metric states after each epoch
        train_metrics.reset()
        val_metrics.reset()
    # save final model checkpoint
    utils.save_model_checkpoint(epoch, config.device, model, optimizer, output_path, config.job_name)

if __name__ == "__main__":
    config = tyro.cli(TrainConfig)
    train(config)

