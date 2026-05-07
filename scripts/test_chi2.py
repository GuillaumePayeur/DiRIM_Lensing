import sys
import yaml
import numpy as np
import os
import torch
import tqdm
import h5py

from dirim_lensing import Config
from dirim_lensing import LensingModel
from dirim_lensing import RIM
from dirim_lensing import SDE
from dirim_lensing import Sampler
from dirim_lensing import load_datasets
from train_rim import load_unet

def save_h5(chi2, s0, k0, s_samples, k_samples, y, y_hat, model_name, config):
    os.makedirs('./results/chi2', exist_ok=True)

    with h5py.File(f"./results/chi2/chi2_{model_name}.h5", "a") as f:
        # If datasets don't exist yet, create them with expandable dimensions
        if "chi2" not in f:
            f.create_dataset("chi2", data=chi2.numpy(), maxshape=(None,))
            f.create_dataset("s0", data=s0.numpy(), maxshape=(None, *s0.shape[1:]))
            f.create_dataset("k0", data=k0.numpy(), maxshape=(None, *k0.shape[1:]))
            f.create_dataset("s_samples", data=s_samples.numpy(), maxshape=(None, *s_samples.shape[1:]))
            f.create_dataset("k_samples", data=k_samples.numpy(), maxshape=(None, *k_samples.shape[1:]))
            f.create_dataset("y", data=y.numpy(), maxshape=(None, *y.shape[1:]))
            f.create_dataset("y_hat", data=y_hat.numpy(), maxshape=(None, *y_hat.shape[1:]))
        else:
            # Resize and append the new batch
            num_new = chi2.shape[0]
            for key, data in zip(["chi2", "s0", "k0", "s_samples", "k_samples", "y", "y_hat"],
                                 [chi2, s0, k0, s_samples, k_samples, y, y_hat]):
                dataset = f[key]
                current_size = dataset.shape[0]
                dataset.resize(current_size + num_new, axis=0)
                dataset[current_size:] = data.numpy()

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
        delta_y = (y - y_hat) / sigma_y
        chi2 = torch.sum(delta_y**2, dim=[1, 2, 3])
        # Reshaping and sending to CPU
        s0 = s0.view(batch_size, config.dataset.res, config.dataset.res).cpu()
        k0 = k0.view(batch_size, config.dataset.res, config.dataset.res).cpu()
        s_samples = s_samples.view(batch_size, config.dataset.res, config.dataset.res).cpu()
        k_samples = k_samples.view(batch_size, config.dataset.res, config.dataset.res).cpu()
        y = y.view(batch_size, config.dataset.res, config.dataset.res).cpu()
        y_hat = y_hat.view(batch_size, config.dataset.res, config.dataset.res).cpu()
        chi2 = chi2.cpu()
        # Storing in h5 file
        save_h5(chi2, s0, k0, s_samples, k_samples, y, y_hat, model_name, config)
        
        idx += batch_size

    print("Sampling complete.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tarp_sample.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    config = Config.parse_obj(data)
    
    main(config)