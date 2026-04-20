import sys
import torch
import yaml
import os
import h5py
import caustics
import math
import tqdm
from torch.nn.functional import avg_pool2d
from caustics.utils import gaussian as caustics_gaussian

from dirim_lensing import Config
from dirim_lensing import LensingModel
from dirim_lensing import RIM
from dirim_lensing import SDE
from dirim_lensing import Sampler
from dirim_lensing import load_datasets
from train_rim import load_unet

def save_h5(idx, params_fit, fit, model_name):
    os.makedirs('./results/comparison_traditional', exist_ok=True)

    with h5py.File(f"./results/comparison_traditional/comparison_traditional_{model_name}.h5", "a") as f: 
        if "idx" not in f:
            # Create expandable datasets
            f.create_dataset("idx", data=[idx], maxshape=(None,))
            f.create_dataset("params_fit", data=[params_fit], 
                            maxshape=(None,) + tuple(params_fit.shape),
                            chunks=True)
            f.create_dataset("fit", data=[fit], 
                            maxshape=(None,) + tuple(fit.shape),
                            chunks=True)

        else:
            # Appending data
            N = f["idx"].shape[0]
            
            f["idx"].resize((N + 1,))
            f["idx"][N] = idx

            f["params_fit"].resize((N + 1,) + f["params_fit"].shape[1:])
            f["params_fit"][N] = params_fit

            f["fit"].resize((N + 1,) + f["fit"].shape[1:])
            f["fit"][N] = fit

class Macromodel_Fitter:
    def __init__(self, res, pixelscale, source_pixelscale, z_s, z_l, psf_sigma,
                 sigma_y, upsample_factor, lim_high_fit, lim_low_prior, lim_high_prior, lim_low_phy, lim_high_phy):
        self.res = res
        self.pixelscale = pixelscale
        self.source_pixelscale = source_pixelscale
        self.z_s = z_s
        self.z_l = z_l
        self.psf_sigma = psf_sigma
        self.sigma_y = sigma_y
        self.upsample_factor = upsample_factor
        self.lim_high_fit = lim_high_fit.to('cuda')
        self.lim_low_prior = lim_low_prior.to('cuda')
        self.lim_high_prior = lim_high_prior.to('cuda')
        self.lim_low_phy = lim_low_phy.to('cuda')
        self.lim_high_phy = lim_high_phy.to('cuda')

    def make_simulator(self):
        # Set up cosmology model (flat Lambda-CDM)
        cosmo = caustics.FlatLambdaCDM(name="cosmo")

        # Define the source as a pixelated image
        source = caustics.Pixelated(name="source", shape=(self.res, self.res), 
                           pixelscale=self.source_pixelscale, x0=0, y0=0)  
        
        # Define the lens
        lens_epl = caustics.EPL(cosmology=cosmo, name="epl")

        # External shear
        shear = caustics.ExternalShear(cosmology=cosmo, name="external shear", x0=lens_epl.x0, y0=lens_epl.y0)

        # Multipole perturbations
        multipole3 = caustics.lenses.multipole.Multipole(
            cosmology=cosmo, name='multipole3', m=3, x0=lens_epl.x0, y0=lens_epl.y0)
        multipole4 = caustics.lenses.multipole.Multipole(
            cosmology=cosmo, name='multipole4', m=4, x0=lens_epl.x0, y0=lens_epl.y0) 
        
        # Creating the caustics lens object
        lens = caustics.SinglePlane(cosmo, name='lens', z_s=self.z_s, z_l=self.z_l, 
                                    lenses=(lens_epl, shear, multipole3, multipole4))

        # Define the gaussian PSF
        if self.psf_sigma > 0:
            psf_image = caustics_gaussian(nx=self.res*self.upsample_factor//2, 
                                          ny=self.res*self.upsample_factor//2, 
                                          pixelscale=self.pixelscale/self.upsample_factor,
                                          sigma=self.psf_sigma)
            
        # Create complete tensing simulator
        simulator = caustics.LensSource(lens, source, pixelscale=self.pixelscale, 
                               pixels_x=self.res, 
                               psf=psf_image if self.psf_sigma > 0 else None,
                               upsample_factor=self.upsample_factor,
                               psf_mode='fft' if self.upsample_factor == 1 else 'conv2d')
        
        return simulator.to('cuda')

    def forward(self, params, simulator, source):
        # construct simulator input
        source = source.reshape(-1, self.res**2).repeat(params.shape[0], 1)
        simulator_input = torch.cat((params.reshape(-1, 12), source), dim=1)

        # create simulation
        fit = torch.vmap(simulator)(simulator_input)

        return fit

    def latin_hypercube(self, n_pts, lim_high):
        n_dims = lim_high.shape[0]
        strata = (torch.rand(1, n_pts, n_dims, device='cuda')
                + torch.arange(n_pts, device='cuda').unsqueeze(1).unsqueeze(0)) / n_pts
        
        perm = torch.argsort(torch.rand(1, n_pts, n_dims, device='cuda'), dim=1)
        samples = strata[torch.arange(1).unsqueeze(1).unsqueeze(2), perm, torch.arange(n_dims).unsqueeze(0).unsqueeze(0)] * lim_high.unsqueeze(0).unsqueeze(0)
        
        return samples[0]

    def neg_log_likelihood(self, obs_true, fit):
        residuals = (obs_true - fit.unsqueeze(0)) / self.sigma_y
        nll = 0.5 * torch.sum(residuals**2, dim=(2, 3))
        return nll

    def normalize(self, params):
        # normalizing parameters from the range [lim_low_prior, lim_high_prior] to the range [0, lim_high_fit]
        if params.dim() == 1:
            lim_low_prior = self.lim_low_prior[0:params.shape[-1]]
            lim_high_prior = self.lim_high_prior[0:params.shape[-1]]
            lim_high_fit = self.lim_high_fit[0:params.shape[-1]]
        if params.dim() == 2:
            lim_low_prior = self.lim_low_prior[0:params.shape[-1]].unsqueeze(0)
            lim_high_prior = self.lim_high_prior[0:params.shape[-1]].unsqueeze(0)
            lim_high_fit = self.lim_high_fit[0:params.shape[-1]].unsqueeze(0)
        elif params.dim() == 3:
            lim_low_prior = self.lim_low_prior[0:params.shape[-1]].unsqueeze(0).unsqueeze(0)
            lim_high_prior = self.lim_high_prior[0:params.shape[-1]].unsqueeze(0).unsqueeze(0)
            lim_high_fit = self.lim_high_fit[0:params.shape[-1]].unsqueeze(0).unsqueeze(0)
        params = (params - lim_low_prior) / (lim_high_prior - lim_low_prior)
        params = params * lim_high_fit
        return params
    
    def denormalize(self, params):
        # denormalizing parameters from the range [0, lim_high_fit] to the range [lim_low_prior, lim_high_prior]
        if params.dim() == 1:
            lim_low_prior = self.lim_low_prior[0:params.shape[-1]]
            lim_high_prior = self.lim_high_prior[0:params.shape[-1]]
            lim_high_fit = self.lim_high_fit[0:params.shape[-1]]
        if params.dim() == 2:
            lim_low_prior = self.lim_low_prior[0:params.shape[-1]].unsqueeze(0)
            lim_high_prior = self.lim_high_prior[0:params.shape[-1]].unsqueeze(0)
            lim_high_fit = self.lim_high_fit[0:params.shape[-1]].unsqueeze(0)
        elif params.dim() == 3:
            lim_low_prior = self.lim_low_prior[0:params.shape[-1]].unsqueeze(0).unsqueeze(0)
            lim_high_prior = self.lim_high_prior[0:params.shape[-1]].unsqueeze(0).unsqueeze(0)
            lim_high_fit = self.lim_high_fit[0:params.shape[-1]].unsqueeze(0).unsqueeze(0)
        params = params / lim_high_fit
        params = params * (lim_high_prior - lim_low_prior) + lim_low_prior
        return params

    def angle_wrapping(self, params):
        # wrapping the angle parameter of the EPL and the multipole perturbations
        if params.dim() == 1:
            params[3] = torch.remainder(params[3], math.pi)
            params[9] = torch.remainder(params[9], 2*math.pi/3)
            params[11] = torch.remainder(params[11], math.pi/2)
        elif params.dim() == 2:
            params[:, 3] = torch.remainder(params[:, 3], math.pi)
            params[:, 9] = torch.remainder(params[:, 9], 2*math.pi/3)
            params[:, 11] = torch.remainder(params[:, 11], math.pi/2)
            
        return params

    def optimize_adam(self, obs_true, source, params, simulator, lr, num_steps):
        # making parameters to optimize leaf tensor
        params = params.detach().requires_grad_(True)

        # using Adam optimizer
        optimizer = torch.optim.Adam([params], lr=lr)

        for _ in range(num_steps):
            optimizer.zero_grad()

            fit = self.forward(self.denormalize(params), simulator, source=source)
            loss = self.neg_log_likelihood(obs_true, fit)
            loss_sum = torch.sum(loss)
            loss_sum.backward()
            optimizer.step()

            with torch.no_grad():
                # clamping parameters to their physical range
                params.clamp_(min=self.normalize(self.lim_low_phy)[0:params.shape[-1]], max=self.normalize(self.lim_high_phy)[0:params.shape[-1]])        

        # keeping only the best fit parameters for each kappa map in the batch
        best_idx = torch.argmin(torch.nan_to_num(loss, nan=float('inf')), dim=1)

        return params[best_idx]

    def fit_oneshot(self, obs_true, source, lr, num_walkers, num_steps):
        # making caustics simulator object
        simulator = self.make_simulator()

        # initializing parameters
        params = self.latin_hypercube(num_walkers, self.lim_high_fit)

        # optimizing with Adam
        params = self.optimize_adam(obs_true, source, params, simulator, lr, num_steps)
        params = self.denormalize(params).cpu()

        return self.angle_wrapping(params)

def main(config):
    # Load datasets
    if config.dataset.name == 'SKIRT_TNG':
        (_, _, test_dataset, _, _, _
         ) = load_datasets(save_path = config.skirt_tng_dataset.save_path, 
                           batch_size = 1,
                           augment=False)
        test_observations = torch.load(os.path.join(config.skirt_tng_dataset.save_path, 'observations_test.pt'))

    # Extracting model name
    model_name = sys.argv[1].split('config_')[1].replace('.yaml', '')

    # Loading trained Unet model
    print("Loading trained Unet model...")
    net = load_unet(config, model_name, config.sampling.model_epoch)
    net.to('cuda').eval()
    print("Model loaded successfully.")

    # Creating the lensing model
    print("Initializing lensing model...")
    if config.dataset.name == 'SKIRT_TNG':
        lensingmodel = LensingModel(res = config.dataset.res, 
                                    source_pixelscale = config.skirt_tng_dataset.source_pixelscale, 
                                    pixelscale = config.skirt_tng_dataset.pixelscale, 
                                    z_s = config.skirt_tng_dataset.z_s, 
                                    z_l = config.skirt_tng_dataset.z_l, 
                                    psf_sigma = config.skirt_tng_dataset.psf_sigma, 
                                    sigma_y = config.skirt_tng_dataset.sigma_y,
                                    upsample_factor = config.skirt_tng_dataset.upsample_factor,
                                    device = 'cuda')
    print("Lensing model initialized successfully.")

    # Initializing the SDE
    sde = SDE(kind = config.sde.kind, 
              epsilon = config.sde.epsilon, 
              sigma_min = config.sde.sigma_min, 
              sigma_max = config.sde.sigma_max, 
              beta_min = config.sde.beta_min, 
              beta_max = config.sde.beta_max) 
    
    # Initializing the RIM
    rim = RIM(net = net, 
            lensingmodel = lensingmodel, 
            sde = sde, 
            space_kappa = config.sde.space_kappa, 
            grad_lik = config.rim.grad_lik, 
            loss = config.loss, 
            num_iterations = config.rim.num_iterations, 
            use_log_t = config.rim.use_log_t,
            device = 'cuda')

    # Get test examples from the dataset
    for idx in tqdm.tqdm(config.tests.sample_idxs, desc=f"Fitting macromodels to observations"):
        s0, k0 = test_dataset[idx]
        # Clean source and kappa map
        s0 = s0.to(rim.device).float()
        k0 = k0.to(rim.device).float()
        # Converting kappa map to RIM units
        k0 = rim.caustics_to_rim(k0)
        # Get lensed image from test set
        y = test_observations[idx].to(rim.device)

        # Initialize the macromodel fitter
        fitter = Macromodel_Fitter(res=config.dataset.res, 
                                pixelscale=config.skirt_tng_dataset.pixelscale, 
                                source_pixelscale=config.skirt_tng_dataset.source_pixelscale, 
                                z_s=config.skirt_tng_dataset.z_s, 
                                z_l=config.skirt_tng_dataset.z_l, 
                                psf_sigma=config.skirt_tng_dataset.psf_sigma,
                                sigma_y=config.skirt_tng_dataset.sigma_y, 
                                upsample_factor=config.skirt_tng_dataset.upsample_factor, 
                                lim_high_fit=torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]), 
                                lim_low_prior=torch.tensor([-0.12,-0.12,0.7,0.,1.,0.75,-0.25,-0.25,0.,0.,0.,0.]), 
                                lim_high_prior=torch.tensor([0.12,0.12,1.0,math.pi,2.,1.25,0.25,0.25,0.05,2*math.pi/3,0.05,math.pi/2]), 
                                lim_low_phy=torch.tensor([-float('inf'), -float('inf'), 0., -float('inf'), 0., 0., -float('inf'), -float('inf'), 0., -float('inf'), 0., -float('inf')]), 
                                lim_high_phy=torch.tensor([float('inf'), float('inf'), 1.0, float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')]))        

        # Fitting the macromodel parameters
        params_fit = fitter.fit_oneshot(obs_true=y, source=s0, lr=0.05, num_walkers=1000, num_steps=1000)
        fit = fitter.forward(params_fit.to('cuda'), fitter.make_simulator(), s0).detach().cpu().numpy().reshape(config.dataset.res, config.dataset.res)
        params_fit = params_fit.detach().cpu().numpy().reshape(-1)

        # Save the data to a h5 file
        save_h5(idx, params_fit, fit, model_name)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_skirt_tng.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    config = Config.parse_obj(data)
    
    main(config)