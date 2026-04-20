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

def main(config):
    # Load datasets
    if config.dataset.name == 'SKIRT_EPL':
        (_, _, _, _, _, test_loader
         ) = load_datasets(save_path = config.skirt_epl_dataset.save_path, 
                           batch_size = 128,
                           augment = False)
    elif config.dataset.name == 'SKIRT_TNG':
        (_, _, _, _, _, test_loader
         ) = load_datasets(save_path = config.skirt_tng_dataset.save_path, 
                           batch_size = 128,
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
    print("Creating test set observations...")
    
    # Tensor to hold observations
    y_all = torch.empty(len(test_loader.dataset), 1, config.dataset.res, config.dataset.res)
    idx = 0

    for s0, k0 in tqdm.tqdm(test_loader, desc="Generating observations"):
        # Clean source and kappa map
        s0 = s0.to(rim.device).float()
        k0 = k0.to(rim.device).float()
        # Converting kappa map to RIM units
        k0 = rim.caustics_to_rim(k0)
        # Generating lensed image
        _, _, _, y = rim.generate_batch(s0=s0, k0=k0)
        # Storing the observation
        y_all[idx:idx + y.shape[0]] = y
        idx += y.shape[0]

    # Save observations to file
    if config.dataset.name == 'SKIRT_EPL':
        save_path = config.skirt_epl_dataset.save_path
    elif config.dataset.name == 'SKIRT_TNG':
        save_path = config.skirt_tng_dataset.save_path
    os.makedirs(save_path, exist_ok=True)
    torch.save(y_all, os.path.join(save_path, 'observations_test.pt'))
    print(f"Observations saved to {save_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tarp_sample.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    config = Config.parse_obj(data)
    
    main(config)