from pathlib import Path
import numpy as np

from tqdm import tqdm
from einops import rearrange

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torch.distributions import Bernoulli

import h5py

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
from neuralop.models import FNO
from models.resnet import resnet18_cond
# imports for multiple physics pretraining model
from models.avit import build_avit
from YParams import YParams
from models.pde_refiner import PDERefiner

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
class TestConfig:
    """ Code to test and evaluate all models on Navier-Stokes dataset """

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
    # Path to unet 400 model checkpoint
    unet_400_checkpoint_path: Path = Path("../model_checkpoints/unet-400-lr-0.0001-bs-8-epochs-200-1_model_checkpoint_199_final.pt")
    # Path to unet 800 model checkpoint
    unet_800_checkpoint_path: Path = Path("../model_checkpoints/unet-800-lr-0.0001-bs-8-epochs-200-1_model_checkpoint_199_final.pt")
    # Path to fno 400 model checkpoint
    fno_400_checkpoint_path: Path = Path("../model_checkpoints/fno-400-lr-0.00001-bs-8-epochs-200-1_model_checkpoint_199.pt")
    # Path to fno model checkpoint
    fno_800_checkpoint_path: Path = Path("../model_checkpoints/fno-800-lr-0.00001-bs-8-epochs-200-1_model_checkpoint_199.pt")
    # Path to multistep unet model checkpoint
    multistep_unet_checkpoint_path: Path = Path("../model_checkpoints/multistep-20-unet-800-lr-0.0001-ns-64-bs-8-epochs-200-1_model_checkpoint_199.pt")
    # Path to mpp yaml config file
    mpp_config: Path = Path("./conf/mpp_avit_s_config.yaml")
    # Path to pretrained mpp model checkpoint
    mpp_zs_checkpoint: Path = Path("../model_checkpoints/MPP_AViT_S.pt")
    # Path to finetuned mpp model checkpoint
    mpp_checkpoint_path: Path = Path("../model_checkpoints/mpp_s_800_shuffle_lr_0.0001_ns_64x64_model_checkpoint_199.pt")
    # Path to pde-refiner 800 model checkpoint
    refiner_checkpoint_path: Path = Path("../model_checkpoints/refiner_unet_800_shuffle_lr_0.0001_ns_64x64_model_checkpoint_199.pt")
    # Path to rl model checkpoint
    rl_checkpoint_path: Path = Path("../model_checkpoints/actor-0.3-reinforce-eta-0.0-lr-1e-05_model_checkpoint_29.pt")
    # Number of times to run random baseline
    K: int = 1
    # Number of timesteps for the rl should select actions at each step
    S: int = 4

def test(config: TestConfig):
    log.info("Starting test of all models")
    log.info(f"Config: {config}")
    # create output directory if it doesn't exist
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=False)
    dev = torch.device(config.device)
    log.info(f"Loading dataset file {config.dataset_path}")
    dataset = NavierStokesDataset(config.dataset_path, provide_velocity=True)
    train_dataset, rl_dataset, val_dataset, test_dataset = random_split(dataset, config.dataset_split, generator=torch.Generator().manual_seed(config.random_seed))
    # combine test and val datasets
    test_val_dataset = ConcatDataset([test_dataset, val_dataset])
    # dataset statistics for 64x64:
    smoke_mean = 0.7322910149367649
    smoke_std = 0.43668617344401106
    smoke_min = 4.5986561758581956e-07
    smoke_max = 2.121663808822632
    # NOTE our batch size needs to be 1 here because the code below is written to test only one trajectory at a time
    test_val_dataloader = DataLoader(test_val_dataset, 1, False, pin_memory=True)
    # load all model checkpoints
    rl_model, unet_400, unet_800, fno_400, fno_800, multistep_unet, mpp_zs, mpp, refiner, surr = load_models(config, dev)
    T = config.FT - config.ST
    # create lists to store all predictions of all models
    all_gts, all_unet_400_preds, all_unet_800_preds, all_fno_400_preds, all_fno_800_preds, all_multistep_unet_preds, all_mpp_zs_preds, all_mpp_preds, all_refiner_preds, all_rl_preds, all_random_preds = [], [], [], [], [], [], [], [], [], [], []
    # create lists to store actions and mses for each method
    all_actions, all_unet_400_mses, all_unet_800_mses, all_fno_400_mses, all_fno_800_mses, all_multistep_unet_mses, all_mpp_zs_mses, all_mpp_mses, all_refiner_mses, all_rl_mses, all_random_mses = [], [], [], [], [], [], [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_val_dataloader)):
            log.info(f"Starting batch/trajectory {i}")
            sim_ids = batch["sim_id"].to(dev)
            smoke = batch["smoke"].unsqueeze(2).to(dev)
            velocity_x = batch["velocity_x"].unsqueeze(2).to(dev)
            velocity_y = batch["velocity_y"].unsqueeze(2).to(dev)
            # standardize data
            smoke_norm = (smoke - smoke_mean) / smoke_std
            # init smoke prediction B x C x H x W
            last_pred = smoke_norm[:, config.ST-1, :, :, :].unsqueeze(1)
            init_state = smoke_norm[:, config.ST-1, :, :, :].unsqueeze(1)
            num_sim_calls, num_model_calls = 0.0, 0.0
            # make lists to store values for this trajectory
            actions, unet_400_mses, unet_800_mses, fno_400_mses, fno_800_mses, multistep_unet_mses, mpp_zs_mses, mpp_mses, refiner_mses, rl_mses, random_mses = [], [], [], [], [], [], [], [], [], [], []
            # init smoke prediction
            last_pred = smoke_norm[:, config.ST-1, :, :, :].unsqueeze(1)
            init_state = smoke_norm[:, config.ST-1, :, :, :].unsqueeze(1)
            last_pred_unet_400 = smoke_norm[:, config.ST-1, :, :, :].unsqueeze(1)
            last_pred_unet_800 = smoke_norm[:, config.ST-1, :, :, :].unsqueeze(1)
            last_pred_fno_400 = smoke_norm[:, config.ST-1, :, :, :]
            last_pred_fno_800 = smoke_norm[:, config.ST-1, :, :, :]
            last_pred_multistep_unet = smoke_norm[:, config.ST-1, :, :, :].unsqueeze(1)
            last_pred_mpp_zs = smoke_norm[:, config.ST-1, :, :, :].unsqueeze(1)
            last_pred_mpp = smoke_norm[:, config.ST-1, :, :, :].unsqueeze(1)
            last_pred_refiner = smoke_norm[:, config.ST-1, :, :, :].unsqueeze(1)
            last_pred_random = smoke_norm[:, config.ST-1, :, :, :].unsqueeze(1)
            # create labels and boundary conditions that mpp expects
            field_labels = torch.tensor([[0]]).to(dev)
            bcs = torch.as_tensor([[0, 0]]).to(dev)
            start_time = config.ST
            # TODO create tensors for plotting
            gts = torch.zeros((T, smoke.shape[3], smoke.shape[4]))
            unet_400_preds = torch.zeros((T, smoke.shape[3], smoke.shape[4]))
            unet_800_preds = torch.zeros((T, smoke.shape[3], smoke.shape[4]))
            fno_400_preds = torch.zeros((T, smoke.shape[3], smoke.shape[4]))
            fno_800_preds = torch.zeros((T, smoke.shape[3], smoke.shape[4]))
            multistep_unet_preds = torch.zeros((T, smoke.shape[3], smoke.shape[4]))
            mpp_zs_preds = torch.zeros((T, smoke.shape[3], smoke.shape[4]))
            mpp_preds = torch.zeros((T, smoke.shape[3], smoke.shape[4]))
            refiner_preds = torch.zeros((T, smoke.shape[3], smoke.shape[4]))
            rl_preds = torch.zeros((T, smoke.shape[3], smoke.shape[4]))
            random_preds = torch.zeros((T, smoke.shape[3], smoke.shape[4]))
            # errs = torch.zeros((T, smoke.shape[3], smoke.shape[4]))
            for s in range(int(T/config.S)):
                while last_pred.dim() < 5:
                    last_pred = last_pred.unsqueeze(0)
                rl_input = torch.cat((last_pred.squeeze(1), init_state.squeeze(1)), dim=1)
                # make time embedding
                time_tensor = torch.full((rl_input.shape[0],), start_time-1, device=dev)
                # call decision model with time embedding
                rl_logits = rl_model(rl_input, time_tensor)
                rl_probs = F.sigmoid(rl_logits)
                rl_dist = Bernoulli(probs=rl_probs.squeeze())
                # sample rl action
                rl_action = rl_dist.sample()
                if config.S == 1:
                    rl_action = [rl_action.item()]
                    actions.append(rl_action)
                else:
                    actions.append(rl_action.tolist())
                for t in range(start_time, start_time + config.S):
                    # call all baseline models
                    unet_time = torch.full((last_pred_unet_400.shape[0],), t, device=dev)
                    # call unet 400 model
                    unet_400_output = unet_400(last_pred_unet_400, unet_time)
                    unet_400_mse = F.mse_loss((unet_400_output * smoke_std) + smoke_mean, smoke[:, t, :, :, :].unsqueeze(1))
                    unet_400_mses.append(unet_400_mse.item())
                    # call unet 800 model
                    unet_800_output = unet_800(last_pred_unet_800, unet_time)
                    unet_800_mse = F.mse_loss((unet_800_output * smoke_std) + smoke_mean, smoke[:, t, :, :, :].unsqueeze(1))
                    unet_800_mses.append(unet_800_mse.item())
                    # call fno 400 model
                    # for now scale static time to be between 0 and 1
                    fno_time = torch.full((1, last_pred_fno_400.shape[1], last_pred_fno_400.shape[2], last_pred_fno_400.shape[3]), t/config.FT, dtype=torch.float, device=dev)
                    fno_400_input = torch.cat((last_pred_fno_400, fno_time), dim=1)
                    fno_400_output = fno_400(fno_400_input)
                    fno_400_mse = F.mse_loss((fno_400_output * smoke_std) + smoke_mean, smoke[:, t, :, :, :])
                    fno_400_mses.append(fno_400_mse.item())
                    # call fno 800 model
                    fno_800_input = torch.cat((last_pred_fno_800, fno_time), dim=1)
                    fno_800_output = fno_800(fno_800_input)
                    fno_800_mse = F.mse_loss((fno_800_output * smoke_std) + smoke_mean, smoke[:, t, :, :, :])
                    fno_800_mses.append(fno_800_mse.item())
                    # call multistep unet model
                    multistep_unet_output = multistep_unet(last_pred_multistep_unet, unet_time)
                    multistep_unet_mse = F.mse_loss((multistep_unet_output.squeeze() * smoke_std) + smoke_mean, smoke[:, t, :, :, :].squeeze())
                    multistep_unet_mses.append(multistep_unet_mse.item())
                    # call mpp zs model
                    mpp_zs_input = rearrange(last_pred_mpp_zs, "b t c h w -> t b c h w")
                    mpp_zs_output = mpp_zs(mpp_zs_input, field_labels, bcs)
                    mpp_zs_mse = F.mse_loss((mpp_zs_output.squeeze() * smoke_std) + smoke_mean, smoke[:, t, :, :, :].squeeze())
                    mpp_zs_mses.append(mpp_zs_mse.item())
                    # call mpp model
                    mpp_input = rearrange(last_pred_mpp, "b t c h w -> t b c h w")
                    mpp_output = mpp(mpp_input, field_labels, bcs)
                    mpp_mse = F.mse_loss((mpp_output.squeeze() * smoke_std) + smoke_mean, smoke[:, t, :, :, :].squeeze())
                    mpp_mses.append(mpp_mse.item())
                    # call pde refiner
                    refiner_output = refiner(last_pred_refiner, unet_time)
                    refiner_mse = F.mse_loss((refiner_output.squeeze() * smoke_std) + smoke_mean, smoke[:, t, :, :, :].squeeze())
                    refiner_mses.append(refiner_mse.item())
                    # run rl/hyper model
                    # take rl action
                    if rl_action[t-start_time] == 0:
                        num_model_calls += 1
                        model_inputs = last_pred
                        while model_inputs.dim() < 5:
                            model_inputs = model_inputs.unsqueeze(0)
                        model_gts = smoke_norm[:, t, :, :, :].unsqueeze(1)
                        timestep = torch.full((model_inputs.shape[0],), t, device=dev)
                        output = surr(model_inputs, unet_time)
                        mse = F.mse_loss(output, model_gts)
                        orig_mse = F.mse_loss((output * smoke_std) + smoke_mean, (model_gts * smoke_std) + smoke_mean)
                        rl_mses.append(orig_mse.item())
                        # set last pred to current pred
                        last_pred = output.detach()
                    elif rl_action[t-start_time] == 1:
                        num_sim_calls += 1
                        # convert tensors to phiflow grids
                        last_pred_unnorm = (last_pred.detach() * smoke_std + smoke_mean).clone()
                        # create sim inputs
                        smoke_grid = torch_to_phi_centered(last_pred_unnorm.squeeze())
                        velocity_grid = torch_to_phi_staggered(velocity_x[:, t-1].squeeze(), velocity_y[:, t-1].squeeze())
                        smoke_grid, velocity_grid = call_sim(smoke_grid, velocity_grid, config.dt, config.buoyancy_factor, config.diffusion_coef)
                        # convert phiflow grids back to tensors
                        sim_output_smoke = phi_centered_to_torch(smoke_grid).to(dev)
                        # normalize smoke sim output
                        sim_output_smoke = (sim_output_smoke - smoke_mean) / smoke_std
                        mse = F.mse_loss(sim_output_smoke, smoke_norm[:, t, :, ...].squeeze())
                        orig_mse = F.mse_loss(sim_output_smoke * smoke_std + smoke_mean, smoke[:, t, :, ...].squeeze())
                        rl_mses.append(orig_mse.item())
                        # set last pred to current sim pred
                        last_pred = sim_output_smoke.detach()
                    rl_output = last_pred
                    # store ground truth and all predictions in original scale
                    gts[t-config.ST] = smoke[:, t, :, :, :].squeeze()
                    unet_400_preds[t-config.ST] = (unet_400_output * smoke_std + smoke_mean).squeeze()
                    unet_800_preds[t-config.ST] = (unet_800_output * smoke_std + smoke_mean).squeeze()
                    fno_400_preds[t-config.ST] = (fno_400_output * smoke_std + smoke_mean).squeeze()
                    fno_800_preds[t-config.ST] = (fno_800_output * smoke_std + smoke_mean).squeeze()
                    multistep_unet_preds[t-config.ST] = (multistep_unet_output * smoke_std + smoke_mean).squeeze()
                    mpp_zs_preds[t-config.ST] = (mpp_zs_output * smoke_std + smoke_mean).squeeze()
                    mpp_preds[t-config.ST] = (mpp_output * smoke_std + smoke_mean).squeeze()
                    refiner_preds[t-config.ST] = (refiner_output * smoke_std + smoke_mean).squeeze()
                    rl_preds[t-config.ST] = (rl_output * smoke_std + smoke_mean).squeeze()
                    # update last preds to current pred
                    last_pred_unet_400 = unet_400_output
                    last_pred_unet_800 = unet_800_output
                    last_pred_fno_400 = fno_400_output
                    last_pred_fno_800 = fno_800_output
                    last_pred_multistep_unet = multistep_unet_output
                    last_pred_mpp_zs = mpp_zs_output.unsqueeze(0)
                    last_pred_mpp = mpp_output.unsqueeze(0)
                    last_pred_refiner = refiner_output
                # increment start time to next S step window
                start_time += config.S
            # run random baseline, this needs to be done after the rl model runs
            sim_call_choices = np.random.default_rng().choice(range(config.ST, config.FT), int(num_sim_calls), replace=False)
            for t in range(config.ST, config.FT):
                if t not in sim_call_choices:
                    # call model
                    model_inputs = last_pred_random
                    model_gts = smoke_norm[:, t, :, :, :].unsqueeze(0)
                    timestep = torch.full((last_pred_random.shape[0],), t, device=dev)
                    # call surrogate
                    output = surr(last_pred_random, timestep)
                    mse = F.mse_loss(output, model_gts)
                    orig_mse = F.mse_loss((output * smoke_std) + smoke_mean, (model_gts * smoke_std) + smoke_mean)
                    random_mses.append(orig_mse.item())
                    # set last pred to current pred
                    last_pred_random = output
                elif t in sim_call_choices:
                    # call sim
                    last_pred_unnorm = last_pred_random.detach() * smoke_std + smoke_mean
                    # convert tensors to phiflow grids
                    smoke_grid = torch_to_phi_centered(last_pred_unnorm.squeeze())
                    velocity_grid = torch_to_phi_staggered(velocity_x[:, t-1].squeeze(), velocity_y[:, t-1].squeeze())
                    smoke_grid, velocity_grid = call_sim(smoke_grid, velocity_grid, config.dt, config.buoyancy_factor, config.diffusion_coef)
                    # convert phiflow grids back to tensors
                    sim_output_smoke = phi_centered_to_torch(smoke_grid).to(dev)
                    # normalize smoke sim output
                    sim_output_smoke = (sim_output_smoke - smoke_mean) / smoke_std
                    # sim_output_velocity_x, sim_output_velocity_y = phi_staggered_to_torch(velocity_grid)
                    mse = F.mse_loss(sim_output_smoke, smoke_norm[:, t, :, ...].squeeze())
                    orig_mse = F.mse_loss(sim_output_smoke * smoke_std + smoke_mean, smoke[:, t, :, ...].squeeze())
                    random_mses.append(orig_mse.item())
                    # set last pred to current sim pred
                    last_pred_random = sim_output_smoke[None, None, None, ...]
                random_preds[t-config.ST] = (last_pred_random * smoke_std + smoke_mean).squeeze()
            # save predictions for each model
            all_gts.append(gts)
            all_unet_400_preds.append(unet_400_preds)
            all_unet_800_preds.append(unet_800_preds)
            all_fno_400_preds.append(fno_400_preds)
            all_fno_800_preds.append(fno_800_preds)
            all_multistep_unet_preds.append(multistep_unet_preds)
            all_mpp_zs_preds.append(mpp_zs_preds)
            all_mpp_preds.append(mpp_preds)
            all_refiner_preds.append(refiner_preds)
            all_rl_preds.append(rl_preds)
            all_random_preds.append(random_preds)
            # save rl actions and mses of each model
            all_actions.append(actions)
            # save mses of each model
            all_unet_400_mses.append(unet_400_mses)
            all_unet_800_mses.append(unet_800_mses)
            all_fno_400_mses.append(fno_400_mses)
            all_fno_800_mses.append(fno_800_mses)
            all_multistep_unet_mses.append(multistep_unet_mses)
            all_mpp_zs_mses.append(mpp_zs_mses)
            all_mpp_mses.append(mpp_mses)
            all_refiner_mses.append(refiner_mses)
            all_rl_mses.append(rl_mses)
            all_random_mses.append(random_mses)
        log.info("Converting and saving metrics to hdf5")
        with h5py.File(output_path / "metrics.h5", "w") as hfile:
            hfile.create_dataset("gts", data=all_gts)
            hfile.create_dataset("unet_400_preds", data=all_unet_400_preds)
            hfile.create_dataset("unet_800_preds", data=all_unet_800_preds)
            hfile.create_dataset("fno_400_preds", data=all_fno_400_preds)
            hfile.create_dataset("fno_800_preds", data=all_fno_800_preds)
            hfile.create_dataset("multistep_unet_preds", data=all_multistep_unet_preds)
            hfile.create_dataset("mpp_zs_preds", data=all_mpp_zs_preds)
            hfile.create_dataset("mpp_preds", data=all_mpp_preds)
            hfile.create_dataset("refiner_preds", data=all_refiner_preds)
            hfile.create_dataset("rl_preds", data=all_rl_preds)
            hfile.create_dataset("random_preds", data=all_random_preds)
            hfile.create_dataset("rl_actions", data=all_actions)
            hfile.create_dataset("rl_mses", data=all_rl_mses)
            hfile.create_dataset("unet_400_mses", data=all_unet_400_mses)
            hfile.create_dataset("unet_800_mses", data=all_unet_800_mses)
            hfile.create_dataset("fno_400_mses", data=all_fno_400_mses)
            hfile.create_dataset("fno_800_mses", data=all_fno_800_mses)
            hfile.create_dataset("multistep_unet_mses", data=all_multistep_unet_mses)
            hfile.create_dataset("mpp_zs_mses", data=all_mpp_zs_mses)
            hfile.create_dataset("mpp_mses", data=all_mpp_mses)
            hfile.create_dataset("refiner_mses", data=all_refiner_mses)
            hfile.create_dataset("random_mses", data=all_random_mses)

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

@jit_compile
def call_sim(smoke_grid, velocity_grid, dt, buoyancy_factor, diffusion_coef):
    # convert torch tensors to phi flow tensors
    smoke_grid = advect.semi_lagrangian(smoke_grid, velocity_grid, dt) # default dt is 1.5
    buoyancy_force = (smoke_grid * (0, buoyancy_factor)).at(velocity_grid)  # resamples smoke to velocity sample points
    velocity_grid = advect.semi_lagrangian(velocity_grid, velocity_grid, dt) + dt * buoyancy_force
    velocity_grid = diffuse.explicit(velocity_grid, diffusion_coef, dt)
    velocity_grid, _ = fluid.make_incompressible(velocity_grid)
    return smoke_grid, velocity_grid

def load_models(config: TestConfig, dev: torch.device):
    # We must load the rl model before loading the surrogates because of a class definition that gets overriden
    log.info(f"Loading rl model at {config.rl_checkpoint_path}")
    rl_model = resnet18_cond(num_classes=config.S, param_conditioning=False).to(dev)
    rl_model = utils.load_model(config.rl_checkpoint_path, rl_model).to(dev)
    rl_model.eval()

    log.info(f"Loading unet 400 model at {config.unet_400_checkpoint_path}")
    unet_400 = UnetCond(1, 0, 1, 0, 1, 1, 64, "gelu", ch_mults=[1, 2, 2], is_attn=[False, False, False]).to(dev)
    unet_400 = utils.load_model(config.unet_400_checkpoint_path, unet_400).to(dev)
    unet_400.eval()

    log.info(f"Loading unet 800 model at {config.unet_800_checkpoint_path}")
    unet_800 = UnetCond(1, 0, 1, 0, 1, 1, 64, "gelu", ch_mults=[1, 2, 2], is_attn=[False, False, False]).to(dev)
    unet_800 = utils.load_model(config.unet_800_checkpoint_path, unet_800).to(dev)
    unet_800.eval()

    log.info(f"Loading fno 400 model at {config.fno_400_checkpoint_path}")
    fno_400 = FNO(n_modes=(27, 27), hidden_channels=64, in_channels=2, out_channels=1)
    fno_400 = utils.load_model(config.fno_400_checkpoint_path, fno_400).to(dev)
    fno_400.eval()

    log.info(f"Loading fno 800 model at {config.fno_800_checkpoint_path}")
    fno_800 = FNO(n_modes=(27, 27), hidden_channels=64, in_channels=2, out_channels=1)
    fno_800 = utils.load_model(config.fno_800_checkpoint_path, fno_800).to(dev)
    fno_800.eval()

    log.info(f"Loading multistep unet model at {config.multistep_unet_checkpoint_path}")
    multistep_unet = UnetCond(1, 0, 1, 0, 1, 1, 64, "gelu", ch_mults=[1, 2, 2], is_attn=[False, False, False]).to(dev)
    multistep_unet = utils.load_model(config.multistep_unet_checkpoint_path, multistep_unet).to(dev)
    multistep_unet.eval()

    log.info(f"Loading mpp zero-shot model at {config.mpp_zs_checkpoint}")
    mpp_zs_config = YParams(config.mpp_config, "basic_config", False)
    mpp_zs = build_avit(mpp_zs_config)
    mpp_zs_state_dict = torch.load(config.mpp_zs_checkpoint, map_location=dev)
    mpp_zs.load_state_dict(mpp_zs_state_dict)
    mpp_zs = mpp_zs.to(dev)
    mpp_zs.eval()

    log.info(f"Loading mpp model at {config.mpp_checkpoint_path}")
    mpp_config = YParams(config.mpp_config, "basic_config", False)
    mpp = build_avit(mpp_config)
    mpp = utils.load_model(config.mpp_checkpoint_path, mpp).to(dev)
    mpp.eval()

    log.info(f"Loading pde-refiner model at {config.refiner_checkpoint_path}")
    refiner_unet = UnetCond(2, 0, 1, 0, 1, 1, 64, "gelu", ch_mults=[1, 2, 2], is_attn=[False, False, False], param_conditioning="scalar").to(dev)
    refiner_checkpoint = torch.load(config.refiner_checkpoint_path, map_location=dev)
    refiner_unet.load_state_dict(refiner_checkpoint["model_state_dict"])
    refiner_unet.to(dev)
    refiner_unet.eval()
    refiner = PDERefiner(refiner_unet)

    # set surrogate to the unet-400
    surr = unet_400

    return rl_model, unet_400, unet_800, fno_400, fno_800, multistep_unet, mpp_zs, mpp, refiner, surr

if __name__ == "__main__":
    config = tyro.cli(TestConfig)
    test(config)

