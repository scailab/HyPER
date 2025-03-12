import xarray as xr
import torch
from torch.utils.data import Dataset

class NavierStokesDataset(Dataset):
    def __init__(self, ds_path, provide_velocity=False):
        self.ds = xr.open_dataset(ds_path)
        self.provide_velocity = provide_velocity
    
    def __len__(self):
        return len(self.ds.sim_id)
    
    def __getitem__(self, i):
        sim_id = self.ds.smoke[i].sim_id.values
        smoke = torch.from_numpy(self.ds.smoke[i].values.astype("float32"))
        # NOTE we do not normalize data here
        if self.provide_velocity == True:
            velocity_x = torch.from_numpy(self.ds.velocity_x[i].values.astype("float32"))
            velocity_y = torch.from_numpy(self.ds.velocity_y[i].values.astype("float32"))
            return {"sim_id": sim_id, "smoke": smoke, "velocity_x": velocity_x, "velocity_y": velocity_y}
        else:
            return {"sim_id": sim_id, "smoke": smoke}

# dataset statistics:
# smoke mean: 0.7292205927987597
# smoke std: 0.4536652350161242
# smoke min: 7.383177091924154e-08
# smoke max: 2.0940911769866943

        

