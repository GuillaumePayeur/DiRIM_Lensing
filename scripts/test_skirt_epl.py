import sys
import torch
import yaml
import os
import h5py
import caustics
import math
import tqdm
from torch.nn.functional import avg_pool2d

from dirim_lensing import Config
from dirim_lensing import LensingModel
from dirim_lensing import RIM
from dirim_lensing import SDE
from dirim_lensing import Sampler
from dirim_lensing import load_datasets
from train_rim import load_unet

def save_h5(idx, s, k, y, y_noiseless, s_samples, k_samples, y_samples, k_params_true, k_params_samples, s_size, s_size_samples, model_name):
    os.makedirs('./results/test_sample_large', exist_ok=True)

    with h5py.File(f"./results/test_sample_large/test_sample_large_{model_name}.h5", "a") as f: 
        if "idx" not in f:
            # Create expandable datasets
            f.create_dataset("idx", data=[idx], maxshape=(None,))
            f.create_dataset("s", data=[s], 
                            maxshape=(None,) + tuple(s.shape),
                            chunks=True)
            f.create_dataset("k", data=[k], 
                            maxshape=(None,) + tuple(k.shape),
                            chunks=True)
            f.create_dataset("y", data=[y], 
                            maxshape=(None,) + tuple(y.shape),
                            chunks=True)
            f.create_dataset("y_noiseless", data=[y_noiseless], 
                            maxshape=(None,) + tuple(y_noiseless.shape),
                            chunks=True)
            f.create_dataset("s_samples", data=[s_samples], 
                            maxshape=(None,) + tuple(s_samples.shape),
                            chunks=True)
            f.create_dataset("k_samples", data=[k_samples], 
                            maxshape=(None,) + tuple(k_samples.shape),
                            chunks=True)
            f.create_dataset("y_samples", data=[y_samples], 
                            maxshape=(None,) + tuple(y_samples.shape),
                            chunks=True)
            f.create_dataset("k_params_true", data=[k_params_true], 
                            maxshape=(None,) + tuple(k_params_true.shape),
                            chunks=True)
            f.create_dataset("k_params_samples", data=[k_params_samples], 
                            maxshape=(None,) + tuple(k_params_samples.shape),
                            chunks=True)
            f.create_dataset("s_size", data=[s_size], 
                            maxshape=(None,) + tuple(s_size.shape),
                            chunks=True)
            f.create_dataset("s_size_samples", data=[s_size_samples],
                            maxshape=(None,) + tuple(s_size_samples.shape), 
                            chunks=True)
            
        else:
            # Appending data
            N = f["idx"].shape[0]
            
            f["idx"].resize((N + 1,))
            f["idx"][N] = idx

            f["s"].resize((N + 1,) + f["s"].shape[1:])
            f["s"][N] = s

            f["k"].resize((N + 1,) + f["k"].shape[1:])
            f["k"][N] = k

            f["y"].resize((N + 1,) + f["y"].shape[1:])
            f["y"][N] = y

            f["y_noiseless"].resize((N + 1,) + f["y_noiseless"].shape[1:])
            f["y_noiseless"][N] = y_noiseless

            f["s_samples"].resize((N + 1,) + f["s_samples"].shape[1:])
            f["s_samples"][N] = s_samples

            f["k_samples"].resize((N + 1,) + f["k_samples"].shape[1:])
            f["k_samples"][N] = k_samples

            f["y_samples"].resize((N + 1,) + f["y_samples"].shape[1:])
            f["y_samples"][N] = y_samples

            f["k_params_true"].resize((N + 1,) + f["k_params_true"].shape[1:])
            f["k_params_true"][N] = k_params_true

            f["k_params_samples"].resize((N + 1,) + f["k_params_samples"].shape[1:])
            f["k_params_samples"][N] = k_params_samples

            f["s_size"].resize((N + 1,) + f["s_size"].shape[1:])
            f["s_size"][N] = s_size

            f["s_size_samples"].resize((N + 1,) + f["s_size_samples"].shape[1:])
            f["s_size_samples"][N] = s_size_samples

class Kappa_Fitter:
    def __init__(self, res, pixelscale, z_s, z_l, lim_high_fit, 
                 lim_low_prior, lim_high_prior, 
                 lim_low_phy, lim_high_phy,
                 upsample_factor_kappa, quad_level_kappa):
        self.res = res
        self.pixelscale = pixelscale
        self.z_s = z_s
        self.z_l = z_l
        self.lim_high_fit = lim_high_fit.to('cuda')
        self.lim_low_prior = lim_low_prior.to('cuda')
        self.lim_high_prior = lim_high_prior.to('cuda')
        self.lim_low_phy = lim_low_phy.to('cuda')
        self.lim_high_phy = lim_high_phy.to('cuda')
        self.upsample_factor_kappa = upsample_factor_kappa
        self.quad_level_kappa = quad_level_kappa

    def make_lens(self, include_subhalo):
        # cosmogical model
        cosmo = caustics.FlatLambdaCDM(name="cosmo")

        # EPL main halo
        lens_epl = caustics.EPL(cosmology=cosmo, name="epl")

        # multipole perturbations
        multipole3 = caustics.lenses.multipole.Multipole(
            cosmology=cosmo, name='multipole3', m=3)
        multipole4 = caustics.lenses.multipole.Multipole(
            cosmology=cosmo, name='multipole4', m=4)
        
        # creating the caustics lens object
        if include_subhalo:
            subhalo = caustics.lenses.nfw.NFW(cosmology=cosmo, name='subhalo')
            lens = caustics.SinglePlane(cosmo, name='lens', z_s=self.z_s, z_l=self.z_l, 
                            lenses=(lens_epl, multipole3, multipole4, subhalo))
        else:
            lens = caustics.SinglePlane(cosmo, name='lens', z_s=self.z_s, z_l=self.z_l, 
                            lenses=(lens_epl, multipole3, multipole4))
        
        return lens.to('cuda')

    def make_meshgrid(self, n):
        if self.quad_level_kappa:
            X, Y = caustics.utils.meshgrid(self.pixelscale/self.upsample_factor_kappa, 
                                           self.res*self.upsample_factor_kappa, 
                                           self.res*self.upsample_factor_kappa,
                                           device='cuda')
            Xs, Ys, weight = caustics.utils.gaussian_quadrature_grid(self.pixelscale/self.upsample_factor_kappa, 
                                                                     X, Y, self.quad_level_kappa)
            Xs = Xs.unsqueeze(0).repeat(n, 1, 1, 1)
            Ys = Ys.unsqueeze(0).repeat(n, 1, 1, 1)

        elif not self.quad_level_kappa:
            X, Y = caustics.utils.meshgrid(self.pixelscale/self.upsample_factor_kappa, 
                                           self.res*self.upsample_factor_kappa, 
                                           self.res*self.upsample_factor_kappa,
                                           device='cuda')
            Xs = X.unsqueeze(0).repeat(n, 1, 1)
            Ys = Y.unsqueeze(0).repeat(n, 1, 1)
            weight = None

        return Xs, Ys, weight

    def make_kappa(self, lens_params, lens, X, Y, weight, include_subhalo):
        # kappa map with a subhalo (14 parameters)
        if include_subhalo:
            lens_params_ = lens_params.reshape(-1, 14)
            # ordering parameters as caustics expects them
            lens_params_ = torch.cat([lens_params_[:, :6], lens_params_[:, :2], lens_params_[:, 6:8], lens_params_[:, :2], lens_params_[:, 8:14]], dim=1)
            lens_params_ = torch.cat([lens_params_[:, :16], torch.pow(10.0, lens_params_[:, 16]).unsqueeze(1), lens_params_[:, 17:]], dim=1)
            # computing kappa map
            if self.quad_level_kappa:
                kappa = (torch.vmap(lens.convergence)(X, Y, lens_params_)).reshape((*lens_params.shape[:-1], X.shape[-3], X.shape[-2], weight.shape[-1]))
                kappa = torch.log(avg_pool2d(caustics.utils.gaussian_quadrature_integrator(kappa, weight), kernel_size=self.upsample_factor_kappa, stride=self.upsample_factor_kappa))
            elif not self.quad_level_kappa:
                kappa = (torch.vmap(lens.convergence)(X, Y, lens_params_)).reshape((*lens_params.shape[:-1], X.shape[-2], X.shape[-1]))
                kappa = torch.log(avg_pool2d(kappa, kernel_size=self.upsample_factor_kappa, stride=self.upsample_factor_kappa))

        # kappa map without a subhalo (10 parameters)
        else:
            lens_params_ = lens_params.reshape(-1, 10)
            # ordering parameters as caustics expects them
            lens_params_ = torch.cat([lens_params_[:, :6], lens_params_[:, :2], lens_params_[:, 6:8], lens_params_[:, :2], lens_params_[:, 8:10]], dim=1)
            # computing kappa map
            if self.quad_level_kappa:
                kappa = (torch.vmap(lens.convergence)(X, Y, lens_params_)).reshape((*lens_params.shape[:-1], X.shape[-3], X.shape[-2], weight.shape[-1]))
                kappa = torch.log(avg_pool2d(caustics.utils.gaussian_quadrature_integrator(kappa, weight), kernel_size=self.upsample_factor_kappa, stride=self.upsample_factor_kappa))
            elif not self.quad_level_kappa:
                kappa = (torch.vmap(lens.convergence)(X, Y, lens_params_)).reshape((*lens_params.shape[:-1], X.shape[-2], X.shape[-1]))
                kappa = torch.log(avg_pool2d(kappa, kernel_size=self.upsample_factor_kappa, stride=self.upsample_factor_kappa))

        return kappa
    
    def latin_hypercube(self, batch_size, n_pts, lim_high):
        n_dims = lim_high.shape[0]
        strata = (torch.rand(batch_size, n_pts, n_dims, device='cuda')
                + torch.arange(n_pts, device='cuda').unsqueeze(1).unsqueeze(0)) / n_pts
        
        perm = torch.argsort(torch.rand(batch_size, n_pts, n_dims, device='cuda'), dim=1)
        samples = strata[torch.arange(batch_size).unsqueeze(1).unsqueeze(2), perm, torch.arange(n_dims).unsqueeze(0).unsqueeze(0)] * lim_high.unsqueeze(0).unsqueeze(0)
        
        return samples
    
    def MSE(self, kappa_true, kappa_fit):
        return torch.mean((kappa_true - kappa_fit)**2, dim=(2, 3))
    
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
        params[:, 3] = torch.remainder(params[:, 3], math.pi)
        params[:, 7] = torch.remainder(params[:, 7], 2*math.pi/3)
        params[:, 9] = torch.remainder(params[:, 9], math.pi/2)

        return params
    
    def optimize_adam(self, kappa_target, params, lens, X, Y, weight, lr, num_steps, include_subhalo):
        # making parameters to optimize leaf tensor
        params = params.detach().requires_grad_(True)

        # using Adam optimizer
        optimizer = torch.optim.Adam([params], lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)

        for _ in range(num_steps):
            optimizer.zero_grad()

            kappa_fit = self.make_kappa(self.denormalize(params), lens, X, Y, weight, include_subhalo=include_subhalo)
            loss = self.MSE(kappa_target, kappa_fit)
            loss_sum = torch.sum(loss)
            loss_sum.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                # clamping parameters to their physical range
                params.clamp_(min=self.normalize(self.lim_low_phy)[0:params.shape[-1]], max=self.normalize(self.lim_high_phy)[0:params.shape[-1]])

        # keeping only the best fit parameters for each kappa map in the batch
        best_idx = torch.argmin(torch.nan_to_num(loss, nan=float('inf')), dim=1)

        return params[torch.arange(params.shape[0]), best_idx].detach()

    def fit_kappa_oneshot(self, kappa_target, batch_size, num_walkers, lr, num_steps, include_subhalo):
        # making caustics lens object
        if include_subhalo:
            lens = self.make_lens(include_subhalo=True)
        else:
            lens = self.make_lens(include_subhalo=False)

        n_params = 14 if include_subhalo else 10
        params_all = torch.empty((kappa_target.shape[0], n_params))

        # fitting kappa maps in batches
        for i in range((kappa_target.shape[0] // batch_size) + bool(kappa_target.shape[0] % batch_size)):
            if i == kappa_target.shape[0] // batch_size:
                kappa_target_batch = kappa_target[i*batch_size:].to('cuda')
                # first fitting EPL and multipole parameters only
                params_batch = self.latin_hypercube((kappa_target.shape[0] % batch_size), num_walkers, self.lim_high_fit[0:n_params])
                X, Y, weight = self.make_meshgrid((kappa_target.shape[0] % batch_size)*num_walkers)
            else:
                kappa_target_batch = kappa_target[i*batch_size:(i+1)*batch_size].to('cuda')
                params_batch = self.latin_hypercube(batch_size, num_walkers, self.lim_high_fit[0:n_params])
                X, Y, weight = self.make_meshgrid(batch_size*num_walkers)

            # optimizing with Adam
            params_batch = self.optimize_adam(kappa_target_batch, params_batch, lens, X, Y, weight, lr, num_steps, include_subhalo=include_subhalo)
            # mapping parameters back to their physical range
            params_batch = self.denormalize(params_batch)

            if i == kappa_target.shape[0] // batch_size:
                params_all[i*batch_size:] = params_batch.cpu()
            else:
                params_all[i*batch_size:(i+1)*batch_size] = params_batch.cpu()
    
        return self.angle_wrapping(params_all)
    
    def fit_kappa_twoshot(self, kappa_target, batch_size, lr, num_walkers_1, num_steps_1, num_walkers_2, num_steps_2):
        # making caustics lens objects
        lens_no_subhalo = self.make_lens(include_subhalo=False)
        lens = self.make_lens(include_subhalo=True)

        params_all = torch.empty((kappa_target.shape[0], 14))

        # fitting kappa maps in batches
        for i in range((kappa_target.shape[0] // batch_size) + bool(kappa_target.shape[0] % batch_size)):
            if i == kappa_target.shape[0] // batch_size:
                kappa_target_batch = kappa_target[i*batch_size:].to('cuda')
                # first fitting EPL and multipole parameters only
                params_batch = self.latin_hypercube((kappa_target.shape[0] % batch_size), num_walkers_1, self.lim_high_fit[0:10])
                X, Y, weight = self.make_meshgrid((kappa_target.shape[0] % batch_size)*num_walkers_1)
            else:
                kappa_target_batch = kappa_target[i*batch_size:(i+1)*batch_size].to('cuda')
                params_batch = self.latin_hypercube(batch_size, num_walkers_1, self.lim_high_fit[0:10])
                X, Y, weight = self.make_meshgrid(batch_size*num_walkers_1)

            # optimizing with Adam
            params_batch = self.optimize_adam(kappa_target_batch, params_batch, lens_no_subhalo, X, Y, weight, lr, num_steps_1, include_subhalo=False)

            # generating initial subhalo parameters on Latin hypercube in the range of the prior and concatenating them to the EPL and multipole parameters
            params_subhalo_batch = self.latin_hypercube(params_batch.shape[0], num_walkers_2, self.lim_high_fit[10:14])  
            params_batch = torch.cat([params_batch.unsqueeze(1).repeat(1, num_walkers_2, 1), params_subhalo_batch], dim=2)

            if i == kappa_target.shape[0] // batch_size:
                X, Y, weight = self.make_meshgrid((kappa_target.shape[0] % batch_size)*num_walkers_2)
            else:
                X, Y, weight = self.make_meshgrid(batch_size*num_walkers_2)

            # optimizing with Adam
            params_batch = self.optimize_adam(kappa_target_batch, params_batch, lens, X, Y, weight, lr, num_steps_2, include_subhalo=True)
            params_batch = self.denormalize(params_batch)  

            if i == kappa_target.shape[0] // batch_size:
                params_all[i*batch_size:] = params_batch.cpu()
            else:
                params_all[i*batch_size:(i+1)*batch_size] = params_batch.cpu()
    
        return self.angle_wrapping(params_all) 
        

def size(s, moment, source_pixelscale):
    # Normalize the image so that the sum of pixel values is 1
    sum = torch.sum(s, dim=(2,3), keepdim=True)   
    s = s/sum
    # Compute the mean x and y coordinates of the source light
    mean_x = torch.sum(torch.sum(s, dim=(2)) * torch.arange(s.shape[3], dtype=torch.float32, device=s.device).unsqueeze(0).unsqueeze(0), dim=(2))
    mean_y = torch.sum(torch.sum(s, dim=(3)) * torch.arange(s.shape[2], dtype=torch.float32, device=s.device).unsqueeze(0).unsqueeze(0), dim=(2))
    # Compute the standard deviation of the source light position
    x = torch.linspace(0, 63, 64)
    X, Y = torch.meshgrid(x, x, indexing='xy')
    X = X.to(s.device)
    Y = Y.to(s.device)
    size2 = torch.sum(s * (torch.abs((X.unsqueeze(0).unsqueeze(0) - mean_x.unsqueeze(2).unsqueeze(3))**moment) + torch.abs((Y.unsqueeze(0).unsqueeze(0) - mean_y.unsqueeze(2).unsqueeze(3))**moment)), dim=(1,2,3))
    size = size2**(1/moment)*source_pixelscale 
    return size      

def main(config):
    # Load datasets
    if config.dataset.name == 'SKIRT_EPL':
        (_, _, test_dataset, _, _, _
         ) = load_datasets(save_path = config.skirt_epl_dataset.save_path, 
                           batch_size = 1,
                           augment=False)

    # Load test set lens parameters
    test_k_params = torch.load(config.skirt_epl_dataset.save_path + '/kappa_params_test.pt')

    # Extracting model name
    model_name = sys.argv[1].split('config_')[1].replace('.yaml', '')

    # Loading trained Unet model
    print("Loading trained Unet model...")
    net = load_unet(config, model_name, config.sampling.model_epoch)
    net.to('cuda').eval()
    print("Model loaded successfully.")

    # Creating the lensing model
    print("Initializing lensing model...")
    if config.dataset.name == 'SKIRT_EPL':
        lensingmodel = LensingModel(res = config.dataset.res, 
                                    source_pixelscale = config.skirt_epl_dataset.source_pixelscale, 
                                    pixelscale = config.skirt_epl_dataset.pixelscale, 
                                    z_s = config.skirt_epl_dataset.z_s, 
                                    z_l = config.skirt_epl_dataset.z_l, 
                                    psf_sigma = config.skirt_epl_dataset.psf_sigma, 
                                    sigma_y = config.skirt_epl_dataset.sigma_y,
                                    upsample_factor = config.skirt_epl_dataset.upsample_factor,
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

    # Initializing the sampler
    sampler = Sampler(rim = rim, 
                      sampler_name = config.sampling.sampler_name, 
                      num_steps = config.sampling.num_steps, 
                      jump_to_0 = config.sampling.jump_to_0, 
                      n_corrector = config.sampling.n_corrector, 
                      snr = config.sampling.snr)
    
    # Initializing the kappa fitter
    fitter = Kappa_Fitter(res = 64, 
        pixelscale = config.skirt_epl_dataset.pixelscale,
        z_s = config.skirt_epl_dataset.z_s,
        z_l = config.skirt_epl_dataset.z_l, 
        lim_high_fit = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]), 
        lim_low_prior = torch.tensor([-config.skirt_epl_dataset.pixelscale, -config.skirt_epl_dataset.pixelscale, 0.7, 0., 1., 0.75, 0., 0., 0., 0., -config.skirt_epl_dataset.pixelscale*12, -config.skirt_epl_dataset.pixelscale*12, 10.0, 50.0]), 
        lim_high_prior = torch.tensor([config.skirt_epl_dataset.pixelscale, config.skirt_epl_dataset.pixelscale, 1.0, math.pi, 2., 1.25, 0.03, 2*math.pi/3, 0.03, math.pi/2, config.skirt_epl_dataset.pixelscale*12, config.skirt_epl_dataset.pixelscale*12, 11.0, 100.0]), 
        lim_low_phy = torch.tensor([-float('inf'), -float('inf'), 0., -float('inf'), 0.0, 0.0, 0.0, -float('inf'), 0.0, -float('inf'), -float('inf'), -float('inf'), 0.0, 0.0]), 
        lim_high_phy = torch.tensor([float('inf'), float('inf'), 1.0, float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')]),
        upsample_factor_kappa = config.skirt_epl_dataset.upsample_factor_kappa,
        quad_level_kappa = config.skirt_epl_dataset.quad_level_kappa)

    # Number of samples to generate per test example
    n_samples = config.tests.n_samples

    # Get test examples from the dataset
    for idx in config.tests.sample_idxs:
        s0, k0 = test_dataset[idx]
        k_params_true = test_k_params[idx]
        # Clean source and kappa map
        s0 = s0.to(rim.device).float()
        k0 = k0.to(rim.device).float()
        # Converting kappa map to RIM units
        k0 = rim.caustics_to_rim(k0)
        # Generating lensed image
        _, _, _, y = rim.generate_batch(s0=s0, k0=k0)
        # Generating noiseless lensed image
        y_noiseless = lensingmodel.simulate_lensing(s0, rim.rim_to_caustics(k0), noise=False)
        # Computing source size
        s0_size = size(s0.unsqueeze(0), 1, config.skirt_epl_dataset.source_pixelscale)
        # Arrays to hold the samples and other information
        s_samples = torch.empty(n_samples, config.dataset.res, config.dataset.res)
        k_samples = torch.empty(n_samples, config.dataset.res, config.dataset.res)
        y_samples = torch.empty(n_samples, config.dataset.res, config.dataset.res)
        k_params_samples = torch.empty(n_samples, 14)
        s_size_samples = torch.empty(n_samples)
        print("Starting sampling...")
        for i in tqdm.tqdm(range(n_samples // config.tests.batch_size), desc=f"Sampling test index {idx}"):
            # Generating samples
            s_samples_batch, k_samples_batch = sampler.sample_PC(y, num_samples=config.tests.batch_size)
            s_samples_batch = s_samples_batch.view(config.tests.batch_size, 1, config.dataset.res, config.dataset.res)
            k_samples_batch = k_samples_batch.view(config.tests.batch_size, 1, config.dataset.res, config.dataset.res)
            y_samples_batch = lensingmodel.simulate_lensing(s_samples_batch, rim.rim_to_caustics(k_samples_batch), noise=False)
            # Fitting lens parameters
            k_params_samples_batch = fitter.fit_kappa_twoshot(k_samples_batch, batch_size=config.tests.batch_size, lr=0.05, num_walkers_1=500, num_steps_1=500, num_walkers_2=500, num_steps_2=500)
            # Computing source size
            s_size_samples_batch = size(s_samples_batch, 1, config.skirt_epl_dataset.source_pixelscale)
            # Filling in arrays
            s_samples[i*config.tests.batch_size:(i+1)*config.tests.batch_size] = s_samples_batch[:,0,:,:]
            k_samples[i*config.tests.batch_size:(i+1)*config.tests.batch_size] = k_samples_batch[:,0,:,:]
            y_samples[i*config.tests.batch_size:(i+1)*config.tests.batch_size] = y_samples_batch[:,0,:,:]
            k_params_samples[i*config.tests.batch_size:(i+1)*config.tests.batch_size] = k_params_samples_batch
            s_size_samples[i*config.tests.batch_size:(i+1)*config.tests.batch_size] = s_size_samples_batch
        if n_samples % config.tests.batch_size:
            # Generating samples
            s_samples_batch, k_samples_batch = sampler.sample_PC(y, num_samples=n_samples % config.tests.batch_size)
            s_samples_batch = s_samples_batch.view(n_samples % config.tests.batch_size, 1, config.dataset.res, config.dataset.res)
            k_samples_batch = k_samples_batch.view(n_samples % config.tests.batch_size, 1, config.dataset.res, config.dataset.res)
            y_samples_batch = lensingmodel.simulate_lensing(s_samples_batch, rim.rim_to_caustics(k_samples_batch), noise=False)
            # Fitting lens parameters
            k_params_samples_batch = fitter.fit_kappa_twoshot(k_samples_batch, batch_size=config.tests.batch_size, lr=0.05, num_walkers_1=500, num_steps_1=500, num_walkers_2=500, num_steps_2=500)
            # Computing source size
            s_size_samples_batch = size(s_samples_batch, 1, config.skirt_epl_dataset.source_pixelscale)   
            # Filling in arrays
            s_samples[n_samples // config.tests.batch_size * config.tests.batch_size:] = s_samples_batch[:,0,:,:]
            k_samples[n_samples // config.tests.batch_size * config.tests.batch_size:] = k_samples_batch[:,0,:,:]
            y_samples[n_samples // config.tests.batch_size * config.tests.batch_size:] = y_samples_batch[:,0,:,:]
            k_params_samples[n_samples // config.tests.batch_size * config.tests.batch_size:] = k_params_samples_batch
            s_size_samples[n_samples // config.tests.batch_size * config.tests.batch_size:] = s_size_samples_batch
        # Reshaping and sending to cpu
        s0 = s0.view(config.dataset.res, config.dataset.res).cpu()
        k0 = k0.view(config.dataset.res, config.dataset.res).cpu()
        y = y.view(config.dataset.res, config.dataset.res).cpu()
        y_noiseless = y_noiseless.view(config.dataset.res, config.dataset.res).cpu()
        s_samples = s_samples.cpu()
        k_samples = k_samples.cpu()
        y_samples = y_samples.cpu()
        k_params_samples = k_params_samples.cpu()
        s0_size = s0_size.cpu()
        s_size_samples = s_size_samples.cpu()

        # Save the data to a h5 file
        save_h5(idx, s0, k0, y, y_noiseless, s_samples, k_samples, y_samples, k_params_true, k_params_samples, s0_size, s_size_samples, model_name)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_skirt_epl.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    config = Config.parse_obj(data)
    
    main(config)