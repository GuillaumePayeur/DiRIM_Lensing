import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
import torch
import warnings
import h5py

from dirim_lensing import Config
from dirim_lensing import LensingModel
from dirim_lensing import RIM
from dirim_lensing import SDE
from dirim_lensing import Sampler
from train_rim import load_unet
from test_sample import plot_sampling

def save_h5(s, k, y, y_noiseless, s_samples, k_samples, y_samples, model_name):
    with h5py.File(f'./results/test_ood_source_kappa/{model_name}.h5', 'w') as f:
        f.create_dataset('s', data=s.cpu().numpy())
        f.create_dataset('k', data=k.cpu().numpy())
        f.create_dataset('y', data=y.cpu().numpy())
        f.create_dataset('y_noiseless', data=y_noiseless.cpu().numpy())
        f.create_dataset('s_samples', data=s_samples.cpu().numpy())
        f.create_dataset('k_samples', data=k_samples.cpu().numpy())
        f.create_dataset('y_samples', data=y_samples.cpu().numpy())

def plot_sampling(s0, k0, y, s_samples, k_samples, y_noiseless, y_hat, 
                  model_name, config, sigma_y, num_images=5):
    # Extract images to plot and send everything to numpy
    s0 = s0.reshape(num_images, config.dataset.res, config.dataset.res).cpu().numpy()
    k0 = k0.reshape(num_images, config.dataset.res, config.dataset.res).cpu().numpy()
    y = y.reshape(num_images, config.dataset.res, config.dataset.res).cpu().numpy()
    s_samples = s_samples.reshape(5, num_images, config.dataset.res, config.dataset.res).cpu().numpy()
    k_samples = k_samples.reshape(5, num_images, config.dataset.res, config.dataset.res).cpu().numpy()
    y_noiseless = y_noiseless.reshape(num_images, config.dataset.res, config.dataset.res).cpu().numpy()
    y_hat = y_hat.reshape(5, num_images, config.dataset.res, config.dataset.res).cpu().numpy()

    # Make the plots
    plt.style.use('dark_background')
    for i in range(num_images):
        fig, axes = plt.subplots(6, 7, figsize=((6)*2+0.25, 12.5), width_ratios=[2]*(6)+[0.25])
        for j in range(6):
            for k in range(6):
                axes[j, k].axis('off')    

        # True source
        im_s0 = axes[0, 0].imshow(s0[i], cmap='bone')
        axes[0, 0].set_title(r'True $s_0$', fontsize=12)

        # Source samples
        for j in range(1, 6):
            axes[0, j].imshow(s_samples[j-1, i], cmap='bone',
                            vmin=s0[i].min(), vmax=s0[i].max())
            axes[0, j].set_title(r'Sampled $s_0$', fontsize=12)

        # Source colorbar
        cbar_s0 = fig.colorbar(im_s0, cax=axes[0, -1])
        cbar_s0.ax.tick_params(labelsize=10)

        # Source residuals
        for j in range(1, 6):
            im_delta_s0 = axes[1, j].imshow(s_samples[j-1, i]-s0[i], cmap='seismic',
                            vmin=-np.abs(s_samples[:, i]-s0[i]).max(), vmax=np.abs(s_samples[:, i]-s0[i]).max())
            axes[1, j].set_title(r'$\delta s_0$', fontsize=12)
        
        # Source residuals colorbar
        cbar_delta_s0 = fig.colorbar(im_delta_s0, cax=axes[1, -1])
        cbar_delta_s0.ax.tick_params(labelsize=10)

        # True kappa
        im_k0 = axes[2, 0].imshow(k0[i], cmap='hot')
        axes[2, 0].set_title(r'True $\log \kappa_0$', fontsize=12)

        # Kappa samples
        for j in range(1, 6):
            axes[2, j].imshow(k_samples[j-1, i], cmap='hot',
                            vmin=k0[i].min(), vmax=k0[i].max())
            axes[2, j].set_title(r'Sampled $\kappa_0$', fontsize=12)

        # Kappa colorbar
        cbar_k0 = fig.colorbar(im_k0, cax=axes[2, -1])
        cbar_k0.ax.tick_params(labelsize=10)

        # Kappa residuals
        for j in range(1, 6):
            im_delta_k0 = axes[3, j].imshow(k_samples[j-1, i]-k0[i], cmap='seismic',
                            vmin=-np.abs(k_samples[:, i]-k0[i]).max(), vmax=np.abs(k_samples[:, i]-k0[i]).max())
            axes[3, j].set_title(r'$\delta \kappa_0$', fontsize=12)
        
        # Kappa residuals colorbar
        cbar_delta_k0 = fig.colorbar(im_delta_k0, cax=axes[3, -1])
        cbar_delta_k0.ax.tick_params(labelsize=10)

        # Observation
        im_y = axes[4, 0].imshow(y[i], cmap='bone')
        axes[4, 0].set_title(r'Observed $y$', fontsize=12) 

        # Lensed image reconstructions
        for j in range(1, 6):
            axes[4, j].imshow(y_hat[j-1, i], cmap='bone',
                            vmin=y[i].min(), vmax=y[i].max())
            axes[4, j].set_title(r'Inferred $y$', fontsize=12)

        # Lensed image colorbar
        cbar_y = fig.colorbar(im_y, cax=axes[4, -1])
        cbar_y.ax.tick_params(labelsize=10)

        # Lensed image noise
        standardized_noise = (y_noiseless[i]-y[i])/sigma_y
        chi2_stat = np.sum(standardized_noise**2)
        dof = standardized_noise.size
        p_value = 1 - stats.chi2.cdf(chi2_stat, dof)
        im_noise = axes[5, 0].imshow(standardized_noise, cmap='seismic',
                                    vmin=-5, vmax=5)
        axes[5, 0].set_title(r'$-\mathcal{N}/\sigma_y$' + f', p={p_value:.3f}', fontsize=12)    

        # Lensed image reconstruction residuals
        for j in range(1, 6):
            standardized_noise = (y_hat[j-1, i]-y[i])/sigma_y
            chi2_stat = np.sum(standardized_noise**2)
            dof = standardized_noise.size
            p_value = 1 - stats.chi2.cdf(chi2_stat, dof)
            axes[5, j].imshow(standardized_noise, cmap='seismic',
                                            vmin=-5, vmax=5)
            axes[5, j].set_title(r'$\delta y/\sigma_y$' + f', p={p_value:.3f}', fontsize=12)
        
        # Lensed image residuals colorbar
        cbar_noise = fig.colorbar(im_noise, cax=axes[5, -1])
        cbar_noise.ax.tick_params(labelsize=10)

        # saving the plot
        os.makedirs('./results/test_ood_source_kappa', exist_ok=True)
        save_dir = f'./results/test_ood_source_kappa/{model_name}_sample{i}.pdf'
        plt.tight_layout()
        fig.subplots_adjust(hspace=0.1, wspace=0.1) 
        plt.savefig(save_dir, dpi=150)
        print(f"Plot saved to {save_dir}")


def main(config):
    # Load ood kappa maps
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        s0 = torch.load('data/ood_sources/ood_sources.pt')
        s0 = s0[torch.randperm(s0.shape[0])]
        k0 = torch.load('data/ood_kappa/ood_kappa.pt')
        k0 = k0[torch.randperm(k0.shape[0])]
        batch_size = s0.shape[0]

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

    # Initializing the sampler
    sampler = Sampler(rim = rim, 
                      sampler_name = config.sampling.sampler_name, 
                      num_steps = config.sampling.num_steps, 
                      jump_to_0 = config.sampling.jump_to_0, 
                      n_corrector = config.sampling.n_corrector, 
                      snr = config.sampling.snr)
    
    # Clean source and kappa map
    s0 = s0.to(rim.device).float()
    k0 = k0.to(rim.device).float()
    # Converting kappa map to RIM units
    k0 = rim.caustics_to_rim(k0)
    # Generating lensed image
    _, _, _, y = rim.generate_batch(s0=s0, k0=k0)
    # Sampling from the model
    print("Starting sampling...")
    s_samples, k_samples = sampler.sample_PC(y, num_samples=5)
    # Generating lensed image and lensed image reconstructions
    y_noiseless = lensingmodel.simulate_lensing(s0, rim.rim_to_caustics(k0), noise=False)
    y_hat = lensingmodel.simulate_lensing(s_samples.reshape(int(5*batch_size),1,config.dataset.res,config.dataset.res), 
                                          rim.rim_to_caustics(k_samples).reshape(int(5*batch_size),1,config.dataset.res,config.dataset.res), 
                                          noise=False).reshape(5,batch_size,1,config.dataset.res,config.dataset.res)

    # Plotting results
    print("Sampling completed. Generating plots...")
    plot_sampling(s0, k0, y, s_samples, k_samples, y_noiseless, y_hat, 
                  model_name, config, sigma_y, num_images=batch_size) 

    # Saving results as h5 file
    save_h5(s0, k0, y, y_noiseless, s_samples, k_samples, y_hat, model_name)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_ood_source_kappa.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    config = Config.parse_obj(data)
    
    main(config)