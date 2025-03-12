from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError

from dataclasses import dataclass
import tyro

import utils
from data.navier_stokes_dataset import NavierStokesDataset
from neuralop.models import FNO

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
    """ Code to train/finetune fno on Navier-Stokes dataset """

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
    lr: float = 0.00001
    # Number of epochs to train
    epochs: int = 200
    # Use training and RL datasets for training (800 trajectories instead of 400).
    use_large_dataset: bool = True
    # Finetune existing model.
    finetune_model_path: Path | None = None
    # Job name.
    job_name: str = f"train-fno-lr-{lr}-ds-ns-64x64-bs-8-epochs-{epochs}-1"

def train(config: TrainConfig):
    log.info(f"Training FNO model with config {config}")
    # create output directory if it doesn't exist
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=False)
    dev = torch.device(config.device)
    log.info(f"Loading dataset file {config.dataset_path}")
    dataset = NavierStokesDataset(config.dataset_path)
    train_dataset, rl_dataset, val_dataset, test_dataset = random_split(dataset, config.dataset_split, generator=torch.Generator().manual_seed(42))
    large_dataset = ConcatDataset((train_dataset, rl_dataset))
    # create model with similar number of parameters as unet
    model = FNO(n_modes=(27, 27), hidden_channels=64, in_channels=2, out_channels=1)
    if config.finetune_model_path is not None:
        log.info(f"Loading model for finetuning from file {config.finetune_model_path}")
        model = utils.load_model(config.finetune_model_path, model)
    model = model.to(dev)
    model.train()
    optimizer = Adam(model.parameters(), lr=config.lr)
    # dataset statistics for 64x64:
    smoke_mean = 0.7322910149367649
    smoke_std = 0.43668617344401106
    smoke_min = 4.5986561758581956e-07
    smoke_max = 2.121663808822632
    # create dataloaders
    train_dataloader = DataLoader(train_dataset, config.batch_size, True, pin_memory=True)
    large_dataloader = DataLoader(large_dataset, config.batch_size, True, pin_memory=True)
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
    for epoch in tqdm(range(config.epochs), unit="epoch"):
        for i, batch in enumerate(tqdm(dataloader, unit="batch")):
            sim_ids = batch["sim_id"].to(dev)
            smoke = batch["smoke"].unsqueeze(2).to(dev)
            # standardize data
            smoke = (smoke - smoke_mean) / smoke_std
            for t in range(config.ST, config.FT):
                inputs = smoke[:, t-1, :, :, :]
                # scale time to be between 0 and 1
                time_map = torch.full((config.batch_size, inputs.shape[1], inputs.shape[2], inputs.shape[3]), t/config.FT, dtype=torch.float, device=dev)
                inputs = torch.cat((inputs, time_map), dim=1)
                # set ground truth to next timestep smoke field
                gt = smoke[:, t, :, :, :]
                output = model(inputs)
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
    utils.save_model_checkpoint(epoch, config.device, model, optimizer, config.output_dir, config.job_name)

if __name__ == "__main__":
    config = tyro.cli(TrainConfig)
    train(config)
