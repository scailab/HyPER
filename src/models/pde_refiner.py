# This code was based on and modified from https://github.com/pdearena/pdearena/blob/main/pdearena/models/pderefiner.py

from functools import partial
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from diffusers.schedulers import DDPMScheduler
# from pytorch_lightning import LightningModule
# from pytorch_lightning.cli import instantiate_class

# from pdearena import utils
# from pdearena.data.utils import PDEDataConfig
from .ema import ExponentialMovingAverage
# NOTE think about if we need to use the custom loss functions, I think we should just use mse loss since this is the fair comparison to our other baselines
# from pdearena.modules.loss import CustomMSELoss, PearsonCorrelationScore, ScaledLpLoss
# from pdearena.rollout import cond_rollout2d

# from .registry import COND_MODEL_REGISTRY

def bootstrap(x: torch.Tensor, Nboot: int, binsize: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bootstrapping the mean of tensor.

    Args:
        x (torch.Tensor):
        Nboot (int): _description_
        binsize (int): _description_

    Returns:
        (Tuple[torch.Tensor, torch.Tensor]): bootstrapped mean and bootstrapped variance
    """
    boots = []
    x = x.reshape(-1, binsize, *x.shape[1:])
    for i in range(Nboot):
        boots.append(torch.mean(x[torch.randint(len(x), (len(x),))], axis=(0, 1)))
    return torch.tensor(boots).mean(), torch.tensor(boots).std()

# one way this could work:
# encapsulates model
# make a prediction with model for 1 timestep
# add noise according to exponential decay schedule and do k step refinements for model prediction
class PDERefiner(nn.Module):
    def __init__(
            self,
            model: nn.Module,
            num_refinement_steps: int = 3,
            min_noise_std: float = 4e-7,
            ema_decay: float = 0.995,
            predict_difference: bool = False,
        ) -> None:
        super().__init__()
        # Set padding for convolutions globally.
        self._mode = "2D"
        nn.Conv2d = partial(nn.Conv2d, padding_mode="zeros")
        self.model = model
        self.predict_difference = predict_difference
        # For Diffusion models and models in general working on small errors,
        # it is better to evaluate the exponential average of the model weights
        # instead of the current weights. If an appropriate scheduler with
        # cooldown is used, the test results will be not influenced.
        # self.ema = ExponentialMovingAverage(self.model, decay=self.hparams.ema_decay)
        # We use the Diffusion implementation here. Alternatively, one could
        # implement the denoising manually.
        betas = [min_noise_std ** (k / num_refinement_steps) for k in reversed(range(num_refinement_steps + 1))]
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_refinement_steps + 1,
            trained_betas=betas,
            prediction_type="v_prediction",
            clip_sample=False,
        )
        # Multiplies k before passing to frequency embedding.
        self.time_multiplier = 1000 / num_refinement_steps

        # time_resolution = self.pde.trajlen
        # Max number of previous points solver can eat
        # reduced_time_resolution = time_resolution - self.hparams.time_history
        # Number of future points to predict
        # self.max_start_time = (
        #     reduced_time_resolution - self.hparams.time_future * self.hparams.max_num_steps - self.hparams.time_gap
        # )
        # self.max_start_time = max(0, self.max_start_time)

    def forward(self, x, cond):
        y_noised = torch.randn(
            size=(x.shape[0], 1, *x.shape[2:]), dtype=x.dtype, device=x.device
        )
        for k in self.scheduler.timesteps:
            time = torch.zeros(size=(x.shape[0],), dtype=x.dtype, device=x.device) + k
            x_in = torch.cat([x, y_noised], axis=1)
            pred = self.model(x_in, time=time * self.time_multiplier, z=cond)
            y_noised = self.scheduler.step(pred, k, y_noised).prev_sample
        y = y_noised
        if self.predict_difference:
            y = y + x[:, -1:]
        return y



