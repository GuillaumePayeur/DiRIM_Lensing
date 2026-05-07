import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py

from dirim_lensing import Config
from dirim_lensing import LensingModel
from dirim_lensing import RIM
from dirim_lensing import SDE
from dirim_lensing import Sampler
from dirim_lensing import load_datasets
from train_rim import load_unet

def save_h5(idx, t, s0, k0, y, s0_hat_series, k0_hat_series, grad_s_series, grad_k_series, 
                   delta_s0_series, delta_k0_series, yres_series, model_name):
    os.makedirs('./results/demo_rim_iterations', exist_ok=True)

    with h5py.File(f'./results/demo_rim_iterations/demo_rim_iterations_{model_name}.h5', 'a') as f:
        if 'idx' not in f:
            # Create expandable datasets
            f.create_dataset("idx", data=[idx], maxshape=(None,))
            f.create_dataset("t", data=[t], 
                            maxshape=(None,) + tuple(t.shape),
                            chunks=True)
            f.create_dataset("s0", data=[s0], 
                            maxshape=(None,) + tuple(s0.shape),
                            chunks=True)
            f.create_dataset("k0", data=[k0], 
                            maxshape=(None,) + tuple(k0.shape),
                            chunks=True)
            f.create_dataset("y", data=[y],
                            maxshape=(None,) + tuple(y.shape),
                            chunks=True)
            f.create_dataset("s0_hat_series", data=[s0_hat_series],
                            maxshape=(None,) + tuple(s0_hat_series.shape),
                            chunks=True)
            f.create_dataset("k0_hat_series", data=[k0_hat_series],
                            maxshape=(None,) + tuple(k0_hat_series.shape),
                            chunks=True)
            f.create_dataset("grad_s_series", data=[grad_s_series],
                            maxshape=(None,) + tuple(grad_s_series.shape),
                            chunks=True)
            f.create_dataset("grad_k_series", data=[grad_k_series],
                            maxshape=(None,) + tuple(grad_k_series.shape),
                            chunks=True)
            f.create_dataset("delta_s0_series", data=[delta_s0_series],
                            maxshape=(None,) + tuple(delta_s0_series.shape),
                            chunks=True)
            f.create_dataset("delta_k0_series", data=[delta_k0_series],
                            maxshape=(None,) + tuple(delta_k0_series.shape),
                            chunks=True)
            f.create_dataset("yres_series", data=[yres_series],
                            maxshape=(None,) + tuple(yres_series.shape),
                            chunks=True)
        else:
            # Appending data
            N = f["idx"].shape[0]

            f["idx"].resize((N + 1,))
            f["idx"][N] = idx

            f["t"].resize((N + 1,) + f["t"].shape[1:])
            f["t"][N] = t

            f["s0"].resize((N + 1,) + f["s0"].shape[1:])
            f["s0"][N] = s0

            f["k0"].resize((N + 1,) + f["k0"].shape[1:])
            f["k0"][N] = k0

            f["y"].resize((N + 1,) + f["y"].shape[1:])
            f["y"][N] = y

            f["s0_hat_series"].resize((N + 1,) + f["s0_hat_series"].shape[1:])
            f["s0_hat_series"][N] = s0_hat_series

            f["k0_hat_series"].resize((N + 1,) + f["k0_hat_series"].shape[1:])
            f["k0_hat_series"][N] = k0_hat_series

            f["grad_s_series"].resize((N + 1,) + f["grad_s_series"].shape[1:])
            f["grad_s_series"][N] = grad_s_series

            f["grad_k_series"].resize((N + 1,) + f["grad_k_series"].shape[1:])
            f["grad_k_series"][N] = grad_k_series

            f["delta_s0_series"].resize((N + 1,) + f["delta_s0_series"].shape[1:])
            f["delta_s0_series"][N] = delta_s0_series

            f["delta_k0_series"].resize((N + 1,) + f["delta_k0_series"].shape[1:])
            f["delta_k0_series"][N] = delta_k0_series

            f["yres_series"].resize((N + 1,) + f["yres_series"].shape[1:])
            f["yres_series"][N] = yres_series

def solve_reverse_to_t(sampler, y, t):
        sde = sampler.rim.sde
        epsilon = sde.epsilon
        
        # time discretization from 1 to t (reverse time)
        times = torch.linspace(1.0, epsilon, sampler.num_steps + 1).to(sampler.rim.device)

        # copying y along new dimension for num_samples
        batch_size = 1
        num_samples = 1
        y = y.repeat(num_samples, 1, 1, 1, 1).reshape(num_samples*batch_size, 1, sampler.rim.net.res, sampler.rim.net.res)

        # arrays to store st and kt at the specified times
        st = torch.zeros(t.shape[0], 1, sampler.rim.net.res, sampler.rim.net.res).to(sampler.rim.device)
        kt = torch.zeros(t.shape[0], 1, sampler.rim.net.res, sampler.rim.net.res).to(sampler.rim.device)

        # samples at t=1
        if sde.kind == 'VE':
            sigma_max = sde.sigma_max
            s_samples = torch.randn(num_samples*batch_size, 1, sampler.rim.net.res, sampler.rim.net.res).to(sampler.rim.device) * sigma_max
            k_samples = torch.randn(num_samples*batch_size, 1, sampler.rim.net.res, sampler.rim.net.res).to(sampler.rim.device) * sigma_max
        elif sde.kind == 'VP_linear' or sde.kind == 'VP_exp':
            s_samples = torch.randn(num_samples*batch_size, 1, sampler.rim.net.res, sampler.rim.net.res).to(sampler.rim.device)
            k_samples = torch.randn(num_samples*batch_size, 1, sampler.rim.net.res, sampler.rim.net.res).to(sampler.rim.device)
        else:
            raise ValueError(f"Unsupported SDE kind {sde.kind}")
        
        n = 0 # number of specified time points in t that we have stored so far
        # Predictor-Corrector integration (going backwards in time from t=1 to t=epsilon)
        for i in range(sampler.num_steps):
            # Compute current time and time at next step
            current_time = times[i]
            next_time = times[i+1]

            # If current time has passed the next specified time in t, store the samples at that time
            if current_time <= t[n]:
                st[n] = s_samples[0].detach().cpu()
                kt[n] = k_samples[0].detach().cpu()
                n += 1
                if n == t.shape[0]: # if we have stored samples at all specified times, we can stop storing
                    break
                    
            # Predictor step
            if sampler.sampler_name == 'EM':
                s_samples, k_samples = sampler.EM_step(s_samples, k_samples, y, current_time, next_time, sde)
            elif sampler.sampler_name == 'Heun':
                s_samples, k_samples = sampler.Heun_step(s_samples, k_samples, y, current_time, next_time, sde)
            elif sampler.sampler_name == 'Euler':
                s_samples, k_samples = sampler.Euler_step(s_samples, k_samples, y, current_time, next_time, sde)
            elif sampler.sampler_name == 'RK4':
                s_samples, k_samples = sampler.RK4_step(s_samples, k_samples, y, current_time, next_time, sde)

            # Langevin corrector step(s)
            if sampler.n_corrector > 0:
                for _ in range(sampler.n_corrector):
                    score_s, score_k = sampler.rim.scores(next_time*torch.ones(s_samples.shape[0], device=sampler.rim.device), s_samples, k_samples, y)
                    z_s = torch.randn_like(s_samples, device=s_samples.device)
                    z_k = torch.randn_like(k_samples, device=k_samples.device)

                    # Langevin step size
                    noise_norm_s = torch.norm(z_s)
                    noise_norm_k = torch.norm(z_k)
                    score_norm_s = torch.norm(score_s)
                    score_norm_k = torch.norm(score_k)
                    eps_s = 2 * (sampler.snr * noise_norm_s / score_norm_s)**2
                    eps_k = 2 * (sampler.snr * noise_norm_k / score_norm_k)**2

                    # Drift
                    s_samples += eps_s * score_s
                    k_samples += eps_k * score_k

                    # Diffusion
                    s_samples += torch.sqrt(2 * eps_s) * z_s
                    k_samples += torch.sqrt(2 * eps_k) * z_k  

        return st, kt

def main(config):
    # Load datasets
    if config.dataset.name == 'SKIRT_EPL':
        (_, _, test_dataset, _, _, _
         ) = load_datasets(save_path = config.skirt_epl_dataset.save_path, 
                           batch_size = 1,
                           augment=False)
        test_observations = torch.load(os.path.join(config.skirt_epl_dataset.save_path, 'observations_test.pt'))
    elif config.dataset.name == 'SKIRT_TNG':
        (_, _, test_dataset, _, _, _
         ) = load_datasets(save_path = config.skirt_tng_dataset.save_path, 
                           batch_size = 1,
                           augment = False)
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
    elif config.dataset.name == 'SKIRT_TNG':
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
    
    # Initializing the sampler
    sampler = Sampler(rim = rim, 
                      sampler_name = config.sampling.sampler_name, 
                      num_steps = config.sampling.num_steps, 
                      jump_to_0 = config.sampling.jump_to_0, 
                      n_corrector = config.sampling.n_corrector, 
                      snr = config.sampling.snr)
    
    # Get test examples from the loader
    for idx in config.tests.sample_idxs:
        s0, k0 = test_dataset[idx]
        # Clean source and kappa map
        s0 = s0.to(rim.device).float().unsqueeze(0).repeat(3, 1, 1, 1)
        k0 = k0.to(rim.device).float().unsqueeze(0).repeat(3, 1, 1, 1) 
        # Converting kappa map to RIM units
        k0 = rim.caustics_to_rim(k0)
        # Get lensed image from test set
        y = test_observations[idx].to(rim.device)
        # Defining diffusion time t
        t = torch.tensor([1.0,0.6,0.2], device=rim.device)
        # Generating noisy source and noisy kappa map by solving the reverse SDE from t=1 to t
        print(f"Solving reverse SDE to get noisy source and kappa at specified times for sample index {idx}...")
        st, kt = solve_reverse_to_t(sampler, y, t)
        print(f"Running RIM forward pass for sample index {idx}...")
        # Forward pass through RIM
        y = y.unsqueeze(0).repeat(t.shape[0], 1, 1, 1)
        s0_hat_series, k0_hat_series, grad_s_series, grad_k_series, yres_series = rim.forward_eval(t, st, kt, y)

        # Reshaping and sending to numpy
        t = t.cpu().numpy()
        s0 = s0.reshape(t.shape[0], config.dataset.res, config.dataset.res).cpu().numpy()
        k0 = k0.reshape(t.shape[0], config.dataset.res, config.dataset.res).cpu().numpy()
        y = y.reshape(t.shape[0], config.dataset.res, config.dataset.res).cpu().numpy()
        n = len(s0_hat_series)
        s0_hat_series = np.array([s0_hat_series[i].reshape(t.shape[0], config.dataset.res, config.dataset.res).cpu().numpy() for i in range(n)])
        delta_s0_series = np.array([s0 - s0_hat_series[i] for i in range(n)])
        grad_s_series = np.array([grad_s_series[i].reshape(t.shape[0], config.dataset.res, config.dataset.res).cpu().numpy() for i in range(n)])
        k0_hat_series = np.array([k0_hat_series[i].reshape(t.shape[0], config.dataset.res, config.dataset.res).cpu().numpy() for i in range(n)])
        delta_k0_series = np.array([k0 - k0_hat_series[i] for i in range(n)])
        grad_k_series = np.array([grad_k_series[i].reshape(t.shape[0], config.dataset.res, config.dataset.res).cpu().numpy() for i in range(n)])
        yres_series = np.array([(yres_series[i]).reshape(t.shape[0], config.dataset.res, config.dataset.res).cpu().numpy() for i in range(n)])

        # Save the data to a h5 file
        save_h5(idx, t, s0, k0, y, s0_hat_series, k0_hat_series, grad_s_series, grad_k_series, 
                   delta_s0_series, delta_k0_series, yres_series, model_name) 

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_denoise.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    config = Config.parse_obj(data)
    
    main(config)