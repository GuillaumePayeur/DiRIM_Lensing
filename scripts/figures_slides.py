import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from dirim_lensing import Config
from dirim_lensing import LensingModel
from dirim_lensing import RIM
from dirim_lensing import SDE
from dirim_lensing import load_datasets
from train_rim import load_unet

def save_image(image, cmap, save_path):
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

def plot_slides(s0, k0, y, s0_hat_series, k0_hat_series, grad_s_series, grad_k_series, 
                yres_series, config, sigma_y):
    # Extract images to plot and send everything to numpy
    s0 = s0.reshape(config.dataset.res, config.dataset.res).cpu().numpy()
    k0 = k0.reshape(config.dataset.res, config.dataset.res).cpu().numpy()
    y_hat = (y - yres_series[-1]*sigma_y).reshape(config.dataset.res, config.dataset.res).cpu().numpy()
    y = y.reshape(config.dataset.res, config.dataset.res).cpu().numpy()
    s0_hat = s0_hat_series[-1].reshape(config.dataset.res, config.dataset.res).cpu().numpy()
    k0_hat = k0_hat_series[-1].reshape(config.dataset.res, config.dataset.res).cpu().numpy()
    st = s0_hat_series[0].reshape(config.dataset.res, config.dataset.res).cpu().numpy()
    kt = k0_hat_series[0].reshape(config.dataset.res, config.dataset.res).cpu().numpy()
    grad_s0 = grad_s_series[0].reshape(config.dataset.res, config.dataset.res).cpu().numpy()
    grad_k0 = grad_k_series[0].reshape(config.dataset.res, config.dataset.res).cpu().numpy()
    yres = yres_series[0].reshape(config.dataset.res, config.dataset.res).cpu().numpy()

    os.makedirs('./results/figures_slides', exist_ok=True)
    save_image(s0, cmap='bone', save_path=f'./results/figures_slides/s0.png')
    save_image(k0, cmap='hot', save_path=f'./results/figures_slides/k0.png')
    save_image(y_hat, cmap='bone', save_path=f'./results/figures_slides/y_hat.png')
    save_image(y, cmap='bone', save_path=f'./results/figures_slides/y.png')
    save_image(s0_hat, cmap='bone', save_path=f'./results/figures_slides/s0_hat.png')
    save_image(k0_hat, cmap='hot', save_path=f'./results/figures_slides/k0_hat.png')
    save_image(st, cmap='bone', save_path=f'./results/figures_slides/st.png')
    save_image(kt, cmap='hot', save_path=f'./results/figures_slides/kt.png')
    save_image(grad_s0, cmap='PuOr_r', save_path=f'./results/figures_slides/grad_s0.png')
    save_image(grad_k0, cmap='PuOr_r', save_path=f'./results/figures_slides/grad_k0.png')
    save_image(yres, cmap='seismic', save_path=f'./results/figures_slides/yres.png')

def main(config):
    # Load datasets
    if config.dataset.name == 'SKIRT_EPL':
        (_, _, _, _, _, test_loader
         ) = load_datasets(save_path = config.skirt_epl_dataset.save_path, 
                           batch_size = 1,
                           augment = config.skirt_epl_dataset.augment,
                           shuffle_test=True)
    elif config.dataset.name == 'SKIRT_TNG':
        (_, _, _, _, _, test_loader
         ) = load_datasets(save_path = config.skirt_tng_dataset.save_path, 
                           batch_size = 1,
                           augment = config.skirt_tng_dataset.augment,
                           shuffle_test=True)

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
        sigma_y = config.skirt_epl_dataset.sigma_y
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
        sigma_y = config.skirt_tng_dataset.sigma_y
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
    
    # Get test examples from the loader
    s0, k0 = next(iter(test_loader))
    # Clean source and kappa map
    s0 = s0.to(rim.device).float()
    k0 = k0.to(rim.device).float()
    # Converting kappa map to RIM units
    k0 = rim.caustics_to_rim(k0)
    # Defining diffusion time t
    t = torch.tensor([0.4], device=rim.device)
    # Generating noisy source and kappa map, and lensed image
    t, st, kt, y = rim.generate_batch(t=t, s0=s0, k0=k0)   
    # Forward pass through RIM
    s0_hat_series, k0_hat_series, grad_s_series, grad_k_series, yres_series = rim.forward_eval(t, st, kt, y)

    # Plotting results
    plot_slides(s0, k0, y, s0_hat_series, k0_hat_series, grad_s_series, grad_k_series, 
                   yres_series, config, sigma_y) 

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_denoise.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    config = Config.parse_obj(data)
    
    main(config)