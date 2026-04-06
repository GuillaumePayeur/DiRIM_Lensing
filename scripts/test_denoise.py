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

def plot_denoising(t, s0, k0, y, s0_hat_series, k0_hat_series, grad_s_series, grad_k_series, 
                   yres_series, model_name, config, sigma_y):
    # Extract images to plot and send everything to numpy
    t = t.cpu().numpy()
    s0 = s0.reshape(6, config.dataset.res, config.dataset.res).cpu().numpy()
    k0 = k0.reshape(6, config.dataset.res, config.dataset.res).cpu().numpy()
    y = y.reshape(6, config.dataset.res, config.dataset.res).cpu().numpy()
    n = len(s0_hat_series)
    s0_hat_series = [s0_hat_series[i].reshape(6, config.dataset.res, config.dataset.res).cpu().numpy() for i in range(n)]
    delta_s0_series = [s0_hat_series[i] - s0 for i in range(n)]
    grad_s_series = [grad_s_series[i].reshape(6, config.dataset.res, config.dataset.res).cpu().numpy() for i in range(n)]
    k0_hat_series = [k0_hat_series[i].reshape(6, config.dataset.res, config.dataset.res).cpu().numpy() for i in range(n)]
    delta_k0_series = [k0_hat_series[i] - k0 for i in range(n)]
    grad_k_series = [grad_k_series[i].reshape(6, config.dataset.res, config.dataset.res).cpu().numpy() for i in range(n)]
    yres_series = [(yres_series[i]).reshape(6, config.dataset.res, config.dataset.res).cpu().numpy() for i in range(n)]
    y_hat_series = [y - yres_series[i]*sigma_y for i in range(n)]

    # Make the plots
    plt.style.use('dark_background')
    for i in range(6):
        fig, axes = plt.subplots(8, n+2, figsize=((n+1)*2+0.25, 16.5), width_ratios=[2]*(n+1)+[0.25])
        for j in range(8):
            for k in range(n+1):
                axes[j, k].axis('off')

        # True source
        im_s0 = axes[0, 0].imshow(s0[i], cmap='bone')
        axes[0, 0].set_title(r'True $s_0$', fontsize=12)

        # Source reconstructions
        for j in range(1, n+1):
            axes[0, j].imshow(s0_hat_series[j-1][i], cmap='bone',
                              vmin=s0[i].min(), vmax=s0[i].max())
            axes[0, j].set_title(fr'$\hat{{s}}_0^{{({j-1})}}$', fontsize=12)

        # Source colorbar
        cbar_s0 = fig.colorbar(im_s0, cax=axes[0, -1])
        cbar_s0.ax.tick_params(labelsize=10)

        # Source residuals
        for j in range(1, n+1):
            im_delta_s0 = axes[1, j].imshow(delta_s0_series[j-1][i], cmap='seismic',
                              vmin=-np.abs(delta_s0_series[-1][i]).max(), vmax=np.abs(delta_s0_series[-1][i]).max())
            axes[1, j].set_title(fr'$\delta s_0^{{({j-1})}}$', fontsize=12)

        # Source residuals colorbar
        cbar_delta_s0 = fig.colorbar(im_delta_s0, cax=axes[1, -1])
        cbar_delta_s0.ax.tick_params(labelsize=10) 

        # Source gradients
        for j in range(1, n+1):
            im_grad_s = axes[2, j].imshow(grad_s_series[j-1][i], cmap='PuOr_r',
                                          vmin=-np.abs(grad_s_series[-1][i]).max(), vmax=np.abs(grad_s_series[-1][i]).max())
            axes[2, j].set_title(fr'$\nabla_s \mathcal{{L}}^{{({j-1})}}$', fontsize=12)      

        # Source gradients colorbar
        cbar_grad_s = fig.colorbar(im_grad_s, cax=axes[2, -1])
        cbar_grad_s.ax.tick_params(labelsize=10)

        # True kappa map
        im_k0 = axes[3, 0].imshow(k0[i], cmap='hot')
        axes[3, 0].set_title(r'True $\log \kappa_0$', fontsize=12)

        # Kappa map reconstructions
        for j in range(1, n+1):
            axes[3, j].imshow(k0_hat_series[j-1][i], cmap='hot',
                              vmin=k0[i].min(), vmax=k0[i].max())
            axes[3, j].set_title(fr'$\log \hat{{\kappa}}_0^{{({j-1})}}$', fontsize=12)

        # Kappa map colorbar
        cbar_k0 = fig.colorbar(im_k0, cax=axes[3, -1])
        cbar_k0.ax.tick_params(labelsize=10)

        # Kappa map residuals
        for j in range(1, n+1):
            im_delta_k0 = axes[4, j].imshow(delta_k0_series[j-1][i], cmap='seismic',
                              vmin=-np.abs(delta_k0_series[-1][i]).max(), vmax=np.abs(delta_k0_series[-1][i]).max())
            axes[4, j].set_title(fr'$\delta \log \kappa_0^{{({j-1})}}$', fontsize=12)

        # Kappa map residuals colorbar
        cbar_delta_k0 = fig.colorbar(im_delta_k0, cax=axes[4, -1])
        cbar_delta_k0.ax.tick_params(labelsize=10)

        # Kappa map gradients
        for j in range(1, n+1):
            im_grad_k = axes[5, j].imshow(grad_k_series[j-1][i], cmap='PuOr_r',
                                          vmin=-np.abs(grad_k_series[-1][i]).max(), vmax=np.abs(grad_k_series[-1][i]).max())
            axes[5, j].set_title(fr'$\nabla_\kappa \mathcal{{L}}^{{({j-1})}}$', fontsize=12)

        # Kappa map gradients colorbar
        cbar_grad_k = fig.colorbar(im_grad_k, cax=axes[5, -1])
        cbar_grad_k.ax.tick_params(labelsize=10)    

        # Observation
        im_y = axes[6, 0].imshow(y[i], cmap='bone')
        axes[6, 0].set_title(r'Observed $y$', fontsize=12)

        # Lensed image reconstructions
        for j in range(1, n+1):
            axes[6, j].imshow(y_hat_series[j-1][i], cmap='bone',
                              vmin=y[i].min(), vmax=y[i].max())
            axes[6, j].set_title(fr'$\hat{{y}}^{{({j-1})}}$', fontsize=12)

        # Lensed image colorbar
        cbar_yhat = fig.colorbar(im_y, cax=axes[6, -1])
        cbar_yhat.ax.tick_params(labelsize=10)

        # Lensed image residuals
        for j in range(1, n+1):
            im_yres = axes[7, j].imshow(yres_series[j-1][i]*sigma_y, cmap='seismic',
                              vmin=-np.abs(yres_series[-1][i]*sigma_y).max(), vmax=np.abs(yres_series[-1][i]*sigma_y).max())
            axes[7, j].set_title(fr'$\delta y^{{({j-1})}}$', fontsize=12)
        
        # Lensed image residuals colorbar
        cbar_yres = fig.colorbar(im_yres, cax=axes[7, -1])
        cbar_yres.ax.tick_params(labelsize=10)

        # saving the plot
        os.makedirs('./results/test_denoise', exist_ok=True)
        save_dir = f'./results/test_denoise/{model_name}_t{t[i]:.1f}.pdf'
        plt.tight_layout()
        fig.subplots_adjust(hspace=0.1, wspace=0.1) 
        plt.savefig(save_dir, dpi=150)
        print(f"Plot saved to {save_dir}")

def main(config):
    # Load datasets
    if config.dataset.name == 'SKIRT_EPL':
        (_, _, _, _, _, test_loader
         ) = load_datasets(save_path = config.skirt_epl_dataset.save_path, 
                           batch_size = 6,
                           augment = config.skirt_epl_dataset.augment,
                           shuffle_test=True)
    elif config.dataset.name == 'SKIRT_TNG':
        (_, _, _, _, _, test_loader
         ) = load_datasets(save_path = config.skirt_tng_dataset.save_path, 
                           batch_size = 6,
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
    t = torch.tensor([config.sde.epsilon,0.2,0.4,0.6,0.8,1.0], device=rim.device)
    # Generating noisy source and kappa map, and lensed image
    t, st, kt, y = rim.generate_batch(t=t, s0=s0, k0=k0)   
    # Forward pass through RIM
    s0_hat_series, k0_hat_series, grad_s_series, grad_k_series, yres_series = rim.forward_eval(t, st, kt, y)

    # Plotting results
    plot_denoising(t, s0, k0, y, s0_hat_series, k0_hat_series, grad_s_series, grad_k_series, 
                   yres_series, model_name, config, sigma_y) 

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_denoise.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    config = Config.parse_obj(data)
    
    main(config)