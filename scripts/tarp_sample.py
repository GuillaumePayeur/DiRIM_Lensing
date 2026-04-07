import sys
import yaml
import numpy as np
import os
import torch
import tqdm

from dirim_lensing import Config
from dirim_lensing import LensingModel
from dirim_lensing import RIM
from dirim_lensing import SDE
from dirim_lensing import Sampler
from dirim_lensing import load_datasets
from train_rim import load_unet

def save_samples(s_samples_all, k_samples_all, s_theta, k_theta, s_references, k_references, 
                 model_name, config):
    # Converting to numpy arrays and reshaping
    s_samples_all = s_samples_all.numpy().reshape(config.tarp.n_sims, config.tarp.n_samples, 1, config.dataset.res, config.dataset.res)
    k_samples_all = k_samples_all.numpy().reshape(config.tarp.n_sims, config.tarp.n_samples, 1, config.dataset.res, config.dataset.res)
    s_theta = s_theta.numpy().reshape(config.tarp.n_sims, 1, config.dataset.res, config.dataset.res)
    k_theta = k_theta.numpy().reshape(config.tarp.n_sims, 1, config.dataset.res, config.dataset.res)
    s_references = s_references.numpy().reshape(config.tarp.n_sims, 1, config.dataset.res, config.dataset.res)
    k_references = k_references.numpy().reshape(config.tarp.n_sims, 1, config.dataset.res, config.dataset.res)
    # Saving the samples
    os.makedirs('./results/tarp_samples', exist_ok=True)
    np.save(f'./results/tarp_samples/s_samples_{model_name}.npy', s_samples_all)
    np.save(f'./results/tarp_samples/k_samples_{model_name}.npy', k_samples_all)
    np.save(f'./results/tarp_samples/s_theta_{model_name}.npy', s_theta)
    np.save(f'./results/tarp_samples/k_theta_{model_name}.npy', k_theta)
    np.save(f'./results/tarp_samples/s_references_{model_name}.npy', s_references)
    np.save(f'./results/tarp_samples/k_references_{model_name}.npy', k_references)
    print("Samples saved successfully.")

def main(config):
    # Load datasets
    if config.dataset.name == 'SKIRT_EPL':
        (_, _, _, _, _, test_loader
         ) = load_datasets(save_path = config.skirt_epl_dataset.save_path, 
                           batch_size = 1,
                           augment = False)
    elif config.dataset.name == 'SKIRT_TNG':
        (_, _, _, _, _, test_loader
         ) = load_datasets(save_path = config.skirt_tng_dataset.save_path, 
                           batch_size = 1,
                           augment = False)
        
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
    
    # Sampling loop
    print("Starting sampling...")
    test_iter = iter(test_loader)

    def get_next_test_sample():
        return next(test_iter)

    # Arrays to hold samples
    s_samples_all = torch.empty(config.tarp.n_samples, config.tarp.n_sims, 1, config.dataset.res, config.dataset.res)
    k_samples_all = torch.empty(config.tarp.n_samples, config.tarp.n_sims, 1, config.dataset.res, config.dataset.res)
    # Arrays to hold true values (theta in tarp)
    s_theta = torch.empty(config.tarp.n_sims, 1, config.dataset.res, config.dataset.res)
    k_theta = torch.empty(config.tarp.n_sims, 1, config.dataset.res, config.dataset.res)
    for i in tqdm.tqdm(range(config.tarp.n_sims), desc="Generating samples"):
        # Get test examples from the loader
        s0, k0 = get_next_test_sample()
        s_theta[i] = s0
        k_theta[i] = k0
        # Clean source and kappa map
        s0 = s0.to(rim.device).float()
        k0 = k0.to(rim.device).float()
        # Converting kappa map to RIM units
        k0 = rim.caustics_to_rim(k0)
        # Generating lensed image
        _, _, _, y = rim.generate_batch(s0=s0, k0=k0)
        # Sampling from the model
        s_samples, k_samples = sampler.sample_PC(y, num_samples=config.tarp.n_samples)
        k_samples = rim.rim_to_caustics(k_samples)
        s_samples_all[:,i] = s_samples[:,0].cpu()
        k_samples_all[:,i] = k_samples[:,0].cpu()
    # Arrays to hold tarp reference values
    s_references = torch.empty(config.tarp.n_sims, 1, config.dataset.res, config.dataset.res)
    k_references = torch.empty(config.tarp.n_sims, 1, config.dataset.res, config.dataset.res)
    for i in range(config.tarp.n_sims):
        # Get test examples from the loader
        s0, k0 = get_next_test_sample()
        s_references[i] = s0
        k_references[i] = k0

    print("Sampling completed. Saving results...")
    save_samples(s_samples_all, k_samples_all, s_theta, k_theta, s_references, k_references, 
                 model_name, config) 

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tarp_sample.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    config = Config.parse_obj(data)
    
    main(config)