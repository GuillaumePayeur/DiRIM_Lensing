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

def save_chi2(chi2_values, model_name, config):
    # Converting to numpy arrays and reshaping
    chi2_values = chi2_values.numpy().reshape(-1)
    # Saving the samples
    os.makedirs('./results/chi2', exist_ok=True)
    np.save(f'./results/chi2/chi2_{model_name}.npy', chi2_values)
    print("Chi2 values saved successfully.")

def main(config):
    # Load datasets
    if config.dataset.name == 'SKIRT_EPL':
        (_, _, _, _, _, test_loader
         ) = load_datasets(save_path = config.skirt_epl_dataset.save_path, 
                           batch_size = 128,
                           augment = False)
        test_observations = torch.load(os.path.join(config.skirt_epl_dataset.save_path, 'observations_test.pt'))
    elif config.dataset.name == 'SKIRT_TNG':
        (_, _, _, _, _, test_loader
         ) = load_datasets(save_path = config.skirt_tng_dataset.save_path, 
                           batch_size = 128,
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
    
    # Sampling loop
    print("Starting sampling...")
    
    # Array to hold chi2 values
    chi2_values = torch.empty(len(test_loader.dataset))
    idx = 0

    for s0, k0 in tqdm.tqdm(test_loader, desc="Generating samples"):
        # Clean source and kappa map
        s0 = s0.to(rim.device).float()
        k0 = k0.to(rim.device).float()
        # Converting kappa map to RIM units
        k0 = rim.caustics_to_rim(k0)
        # Get lensed image from test set
        batch_size = s0.shape[0]
        y = test_observations[idx:idx + batch_size].to(rim.device)
        # Sampling from the model
        s_samples, k_samples = sampler.sample_PC(y, num_samples=1)
        # Generating lensed image and lensed image reconstructions
        y_hat = lensingmodel.simulate_lensing(s_samples.reshape(-1, 1, config.dataset.res, config.dataset.res),
                                              rim.rim_to_caustics(k_samples).reshape(-1, 1, config.dataset.res, config.dataset.res), 
                                              noise=False).reshape(-1, 1, config.dataset.res, config.dataset.res)
        # Computing chi2 values
        sigma_y = lensingmodel.sigma_y
        chi2 = torch.sum((y - y_hat)**2, dim=[1, 2, 3]) / sigma_y**2
        # Storing chi2 values
        chi2_values[idx:idx + batch_size] = chi2.cpu()
        idx += batch_size


    print("Sampling completed. Saving results...")
    save_chi2(chi2_values, model_name, config) 

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tarp_sample.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    config = Config.parse_obj(data)
    
    main(config)