import math
from collections import deque
from pathlib import Path
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torch.distributions import Bernoulli

from phi.torch.flow import (  # SoftGeometryMask,; Sphere,; batch,; tensor,
    Box,
    CenteredGrid,
    StaggeredGrid,
    advect,
    diffuse,
    extrapolation,
    fluid,
    jit_compile,
    spatial,
    channel,
)
from phi.math import tensor as phi_tensor

import utils
from data.navier_stokes_dataset import NavierStokesDataset
from models.twod_unet_cond import Unet as UnetCond
from models.resnet import resnet18_cond

from dataclasses import dataclass
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
    """ Code to test trained UNet on Navier-Stokes dataset """

    # Output directory which will get created if it doesn't exist.
    output_dir: str
    # Path to dataset.
    dataset_path: Path = Path("../data/ns_data_64_1000.h5")
    # Dataset split percentages for train, test, validation, and test sets.
    dataset_split: tuple[float, float, float, float] = (0.40, 0.40, 0.10, 0.10)
    # Random seed.
    random_seed: int = 42
    # Device to run on.
    device: str = "cuda"
    # Batch size.
    batch_size: int = 1
    # Start timestep.
    ST: int = 8
    # End timestep.
    FT: int = 28
    # Timestep duration in seconds
    dt: float = 96.0 / 64.0
    # Buoyancy factor for sim
    buoyancy_factor: float = 0.5
    # Diffusion coefficient for sim
    diffusion_coef: float = 0.01
    # Path to trained surrogate model checkpoint.
    checkpoint_path: Path = Path("../model_checkpoints/unet-400-lr-0.0001-bs-8-epochs-200-1_model_checkpoint_199_final.pt")
    # Learning rate
    learning_rate: float = 0.00001
    # Number of epochs to train for
    epochs: int = 30
    # Lambda cost parameter, what percentage to use the simulator for each trajectory
    lam: float = 0.30
    # Number of timesteps for which the rl should select actions for with each call
    S: int = 4
    # Job name
    job_name: str = f"train-rl-resnet-18-unet-400-lam-{lam}-multistep-{S}-reinforce-simple-baseline-ns-64-lr-{learning_rate}-ts-20-train-1"

def train(config: TrainConfig):
    if config.batch_size != 1:
        raise ValueError("This training code does not currently support a batch size larger than 1")
    # create output directory if it doesn't exist
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=False)
    dev = torch.device(config.device)
    # load dataset
    log.info(f"Loading dataset file {config.dataset_path}")
    dataset = NavierStokesDataset(config.dataset_path, provide_velocity=True)
    train_dataset, rl_dataset, val_dataset, test_dataset = random_split(dataset, config.dataset_split, generator=torch.Generator().manual_seed(42))
    log.info(f"RL dataset length: {len(rl_dataset)}")
    # for now the batch size must be 1
    rl_dataloader = DataLoader(rl_dataset, 1, False, pin_memory=True)
    # create rl model
    actor_model = resnet18_cond(num_classes=config.S, param_conditioning=False).to(dev)
    actor_optimizer = Adam(actor_model.parameters(), lr=config.learning_rate)
    # load ml/surrogate model
    surr_model = UnetCond(1, 0, 1, 0, 1, 1, 64, "gelu", ch_mults=[1, 2, 2], is_attn=[False, False, False]).to(dev)
    surr_model = utils.load_model(config.checkpoint_path, surr_model)
    surr_model.to(dev)
    surr_model.eval()
    # dataset statistics for 64x64:
    smoke_mean = 0.7322910149367649
    smoke_std = 0.43668617344401106
    smoke_min = 4.5986561758581956e-07
    smoke_max = 2.121663808822632
    best_advantage = -math.inf
    T = config.FT - config.ST
    for epoch in range(config.epochs):
        epoch_reward, epoch_policy_loss, epoch_advantage, epoch_advantage_final_penalty, final_step_advantage = 0.0, 0.0, 0.0, 0.0, 0.0
        percent_sim, percent_sim_epoch = 0.0, 0.0
        log.info(f"Starting epoch {epoch}")
        for i, batch in enumerate(tqdm(rl_dataloader, desc="RL Training", unit="batch")):
            log.info(f"Training batch {i}")
            sim_ids = batch["sim_id"].to(dev)
            smoke = batch["smoke"].unsqueeze(2).to(dev)
            velocity_x = batch["velocity_x"].unsqueeze(2).to(dev)
            velocity_y = batch["velocity_y"].unsqueeze(2).to(dev)
            local_batch_size = smoke.shape[0]
            # standardize data
            smoke_norm = (smoke - smoke_mean) / smoke_std
            # keep track of number of sim calls and surrogate calls
            num_sim_calls, num_model_calls = 0.0, 0.0
            total_mse, total_orig_mse = 0.0, 0.0
            # init smoke prediction
            last_pred = smoke_norm[:, config.ST-1, :, :, :].unsqueeze(1)
            init_state = smoke_norm[:, config.ST-1, :, :, :].unsqueeze(1)
            # lists to save trajectory probs, rewards
            log_probs, rewards, entropies, mse_terms, = [], [], [], []
            cum_mse, final_mse = 0.0, 0.0
            start_time = config.ST
            total_sim_calls = 0
            for s in range(int(T/config.S)):
                rl_input = torch.cat((last_pred.squeeze(1), init_state.squeeze(1)), dim=1)
                # call decision model with time embedding
                time_tensor = torch.full((rl_input.shape[0],), start_time-1, device=dev)
                rl_logits = actor_model(rl_input, time_tensor)
                rl_probs = F.sigmoid(rl_logits)
                rl_dist = Bernoulli(probs=rl_probs.squeeze())
                # sample rl action
                rl_action = rl_dist.sample()
                log_probs.append(rl_dist.log_prob(rl_action))
                entropies.append(rl_dist.entropy())
                for t in range(start_time, start_time+config.S):
                    if rl_action[t-start_time] == 0:
                        # call model
                        num_model_calls += 1
                        model_inputs = last_pred
                        model_gts = smoke_norm[:, t, :, :, :].unsqueeze(1)
                        while model_inputs.dim() < 5:
                            model_inputs = model_inputs.unsqueeze(0)
                        output = call_model(t, model_inputs, surr_model, dev)
                        model_mse = F.mse_loss(output, model_gts)
                        orig_mse = F.mse_loss((output * smoke_std) + smoke_mean, (model_gts * smoke_std) + smoke_mean).item()
                        last_pred = output.detach()
                        mse = model_mse.item()
                        reward = -mse
                    elif rl_action[t-start_time] == 1:
                        # call sim
                        num_sim_calls += 1
                        sim_output_smoke = torch.zeros_like(smoke[:, 0, ...], device='cpu')
                        sim_output_velocity_x = torch.zeros_like(velocity_x[:, 0, ...], device='cpu')
                        sim_output_velocity_y = torch.zeros_like(velocity_y[:, 0, ...], device='cpu')
                        last_pred_unnorm = (last_pred.detach() * smoke_std + smoke_mean).clone()
                        # convert tensors to phiflow grids
                        smoke_grid = torch_to_phi_centered(last_pred_unnorm.squeeze())
                        velocity_grid = torch_to_phi_staggered(velocity_x[:, t-1].squeeze(), velocity_y[:, t-1].squeeze())
                        # call sim
                        smoke_grid, velocity_grid = call_sim(smoke_grid, velocity_grid, config.dt, config.buoyancy_factor, config.diffusion_coef)
                        total_sim_calls += 1
                        # convert phiflow grids back to tensors
                        sim_output_smoke[:, ...] = phi_centered_to_torch(smoke_grid)
                        sim_output_velocity_x[:, :, :], sim_output_velocity_y[:, :, :] = phi_staggered_to_torch(velocity_grid)
                        # move tensors back to correct device
                        sim_output_smoke = sim_output_smoke.to(dev)
                        # normalize tensor
                        sim_output_smoke = (sim_output_smoke - smoke_mean) / smoke_std
                        sim_mse = F.mse_loss(sim_output_smoke[:, ...], smoke_norm[:, t, ...]).item()
                        orig_mse = F.mse_loss(sim_output_smoke[:, ...] * smoke_std + smoke_mean, smoke[:, t, ...]).item()
                        last_pred = sim_output_smoke.unsqueeze(1).detach()
                        mse = sim_mse
                        reward = -mse
                    cum_mse += mse # keep track of cumulative mse
                    if t == config.FT-1:
                        final_mse = mse
                    mse_terms.append(mse)
                    rewards.append(reward)
                    total_mse += mse
                    total_orig_mse += orig_mse
                # increment start_time to the next S step window
                start_time = start_time + config.S
            percent_sim = num_sim_calls / T / local_batch_size
            percent_sim_epoch += percent_sim
            total_reward = sum(rewards)
            baseline_rewards = []
            log.info("Starting baseline reward calculation")
            with torch.no_grad():
                # init smoke prediction
                last_pred = smoke_norm[:, config.ST-1, :, :, :].unsqueeze(1)
                sim_call_choices = np.random.default_rng().choice(range(config.ST, config.FT), int(num_sim_calls), replace=False)
                for t in range(config.ST, config.FT):
                    if t not in sim_call_choices:
                        # call surr model
                        model_inputs = last_pred
                        model_gts = smoke_norm[:, t, :, :, :].unsqueeze(1)
                        output = call_model(t, model_inputs, surr_model, dev)
                        model_mse = F.mse_loss(output, model_gts)
                        orig_mse = F.mse_loss((output * smoke_std) + smoke_mean, (model_gts * smoke_std) + smoke_mean).item()
                        last_pred = output.detach()
                        mse = model_mse.item()
                        reward = -mse
                    elif t in sim_call_choices:
                        # call sim
                        sim_output_smoke = torch.zeros_like(smoke[:, 0, ...], device='cpu')
                        sim_output_velocity_x = torch.zeros_like(velocity_x[:, 0, ...], device='cpu')
                        sim_output_velocity_y = torch.zeros_like(velocity_y[:, 0, ...], device='cpu')
                        # convert tensors to phiflow grids
                        last_pred_unnorm = (last_pred.detach() * smoke_std + smoke_mean).clone()
                        # last_pred_unnorm = last_pred * smoke_std + smoke_mean
                        smoke_grid = torch_to_phi_centered(last_pred_unnorm.squeeze())
                        velocity_grid = torch_to_phi_staggered(velocity_x[:, t-1].squeeze(), velocity_y[:, t-1].squeeze())
                        # call sim
                        smoke_grid, velocity_grid = call_sim(smoke_grid, velocity_grid, config.dt, config.buoyancy_factor, config.diffusion_coef)
                        # convert phiflow grids back to tensors
                        sim_output_smoke[:, ...] = phi_centered_to_torch(smoke_grid)
                        sim_output_velocity_x[:, :, :], sim_output_velocity_y[:, :, :] = phi_staggered_to_torch(velocity_grid)
                        # move tensors back to correct device
                        sim_output_smoke = sim_output_smoke.to(dev)
                        # normalize tensor
                        sim_output_smoke = (sim_output_smoke - smoke_mean) / smoke_std
                        sim_mse = F.mse_loss(sim_output_smoke[:, ...], smoke_norm[:, t, ...]).item()
                        orig_mse = F.mse_loss(sim_output_smoke[:, ...] * smoke_std + smoke_mean, smoke[:, t, ...]).item()
                        last_pred = sim_output_smoke.unsqueeze(1).detach()
                        mse = sim_mse
                        reward = -mse
                    baseline_rewards.append(reward)
            # subtract from baseline to get advantage
            final_rewards = [r - br for r, br in zip(rewards, baseline_rewards)]
            epoch_advantage += sum(final_rewards)
            final_step_advantage += final_rewards[-1]
            epoch_advantage_final_penalty += sum(final_rewards)
            # penalize rewards for being far away from lambda percent sim usage (cost function penalty)
            final_rewards = [r - np.abs(percent_sim - config.lam) for r in final_rewards]
            returns = deque(maxlen=T)
            for t in range(T)[::-1]:
                return_t = (returns[0] if len(returns) > 0 else 0)
                returns.appendleft(return_t + final_rewards[t])
            returns = torch.tensor(returns)
            # train model all steps at once
            policy_loss = []
            # concatenate log probs together if predicting for separate time windows
            log_probs = torch.cat(log_probs, 0)
            assert(len(log_probs) == len(returns))
            for log_prob, disc_return in zip(log_probs, returns):
                policy_loss.append(-log_prob * disc_return)
            policy_loss = sum(policy_loss)
            # update model
            actor_optimizer.zero_grad()
            policy_loss.backward()
            actor_optimizer.step()
            epoch_policy_loss += policy_loss.item()
            epoch_reward += sum(final_rewards)
        epoch_reward /= len(rl_dataloader)
        percent_sim_epoch /= len(rl_dataloader)
        epoch_policy_loss /= len(rl_dataloader)
        epoch_advantage /= len(rl_dataloader)
        epoch_advantage_final_penalty /= len(rl_dataloader)
        final_step_advantage /= len(rl_dataloader)
        # Log training metrics
        log.info(f"Epoch {epoch} training metrics")
        log.info(f"Reward: {epoch_reward}")
        log.info(f"Percent Sim Usage: {percent_sim_epoch}")
        log.info(f"Advantage: {epoch_advantage}")
        # save model with best advantage
        if epoch_advantage > best_advantage:
            best_advantage = epoch_advantage
            log.info(f"Saving best model with advantage {epoch_advantage}")
            utils.save_model_checkpoint(epoch, config.device, actor_model, actor_optimizer, output_path, config.job_name, suffix="best", overwrite=True)
    # save final model
    utils.save_model_checkpoint(epoch, config.device, actor_model, actor_optimizer, output_path, config.job_name, suffix="final")

def convert_data_call_sim(sim_idx, t, smoke, velocity_x, velocity_y, smoke_mean, smoke_std, dt, buoyancy_factor, diffusion_coef):
    # convert tensors to phiflow grids
    smoke_grid = torch_to_phi_centered(smoke[sim_idx, t-1].squeeze())
    velocity_grid = torch_to_phi_staggered(velocity_x[sim_idx, t-1].squeeze(), velocity_y[sim_idx, t-1].squeeze())
    smoke_grid, velocity_grid = call_sim(smoke_grid, velocity_grid, dt, buoyancy_factor, diffusion_coef)
    # convert phiflow grids back to tensors
    sim_output_smoke = phi_centered_to_torch(smoke_grid)
    # normalize smoke tensor, have to use sim_idx.item() here because sim_idx is tensor from a different device
    sim_output_smoke = (sim_output_smoke - smoke_mean) / smoke_std
    sim_output_velocity_x, sim_output_velocity_y = phi_staggered_to_torch(velocity_grid)
    return sim_output_smoke, sim_output_velocity_x, sim_output_velocity_y

def torch_to_phi_centered(data):
    phi_ten = phi_tensor(data.transpose(1, 0), spatial('x,y')) # have to transpose the data to get x and y in the right dims
    phi_grid = CenteredGrid(phi_ten, extrapolation.BOUNDARY, Box['x,y', 0 : 32.0, 0 : 32.0])
    return phi_grid

def phi_centered_to_torch(data):
    data_np = data.values.numpy('x,y')
    return torch.from_numpy(data_np.transpose(1,0)) # transpose to get into y, x order

def torch_to_phi_staggered(data_x, data_y):
    # expand torch tensors to be 1 larger in each dimension
    data_x = torch.cat((data_x, data_x[:, -1].reshape(data_x.shape[1], -1)), 1)
    data_x = torch.cat((data_x, data_x[-1, :].reshape(-1, data_x.shape[1])), 0)
    data_y = torch.cat((data_y, data_y[:, -1].reshape(data_y.shape[1], -1)), 1)
    data_y = torch.cat((data_y, data_y[-1, :].reshape(-1, data_y.shape[1])), 0)
    # stack tensors for velocity vector field
    stacked = torch.stack((data_x.transpose(1,0), data_y.transpose(1,0)), dim=2)
    stacked_ten = phi_tensor(stacked, spatial('x,y'), channel('vector'))
    # create staggered grid
    phi_grid = StaggeredGrid(stacked_ten, extrapolation.ZERO, Box['x,y', 0 : 32.0, 0 : 32.0])
    return phi_grid

def phi_staggered_to_torch(data):
    field = data.staggered_tensor().numpy('x,y,vector')
    field_x = torch.from_numpy(field[:-1, :-1, 0].transpose(1,0))
    field_y = torch.from_numpy(field[:-1, :-1, 1].transpose(1,0))
    return field_x, field_y

# last_pred should be shaped B x T x C x H x W
def call_model(timestep: int, last_pred, model, dev):
    timesteps = torch.full((last_pred.shape[0],), timestep, device=dev)
    # call model
    output = model(last_pred, timesteps)
    return output

@jit_compile
def call_sim(smoke_grid, velocity_grid, dt, buoyancy_factor, diffusion_coef):
    # convert torch tensors to phi flow tensors
    smoke_grid = advect.semi_lagrangian(smoke_grid, velocity_grid, dt) # default dt is 1.5
    buoyancy_force = (smoke_grid * (0, buoyancy_factor)).at(velocity_grid)  # resamples smoke to velocity sample points
    velocity_grid = advect.semi_lagrangian(velocity_grid, velocity_grid, dt) + dt * buoyancy_force
    velocity_grid = diffuse.explicit(velocity_grid, diffusion_coef, dt)
    velocity_grid, _ = fluid.make_incompressible(velocity_grid)
    return smoke_grid, velocity_grid

if __name__ == "__main__":
    config = tyro.cli(TrainConfig)
    train(config)



