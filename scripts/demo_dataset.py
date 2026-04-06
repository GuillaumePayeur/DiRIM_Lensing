import sys
import yaml
import torch
import matplotlib.pyplot as plt
import os

from dirim_lensing import Config
from dirim_lensing import LensingModel
from dirim_lensing import RIM
from dirim_lensing import SDE
from dirim_lensing import SongUNet
from dirim_lensing import load_datasets

def demo_plot(s0, st, k0, kt, y, config):
    # Send everything to cpu and numpy
    s0 = s0.reshape(11, config.dataset.res, config.dataset.res).cpu().numpy()
    st = st.reshape(11, config.dataset.res, config.dataset.res).cpu().numpy()
    k0 = k0.reshape(11, config.dataset.res, config.dataset.res).cpu().numpy()
    kt = kt.reshape(11, config.dataset.res, config.dataset.res).cpu().numpy()
    y = y.reshape(11, config.dataset.res, config.dataset.res).cpu().numpy()

    # Make the plots
    plt.style.use('dark_background')
    fig, axes = plt.subplots(5, 11, figsize=(22, 10.5))
    for j in range(5):
        for k in range(11):
            axes[j, k].axis('off')

    for k in range(11):
        # s0
        axes[0, k].imshow(s0[k], cmap='bone')
        axes[0, k].set_title(r'$s_0$', fontsize=12)
        axes[0, k].text(0.96, 0.04, f'max: {s0[k].max():.2f}\nmin: {s0[k].min():.2f}',
                                transform=axes[0, k].transAxes,
                            fontsize=8, verticalalignment='bottom',
                            horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='white',
                                        alpha=0.5))        

        # st
        axes[1, k].imshow(st[k], cmap='bone',
                          vmin=s0[k].min(), vmax=s0[k].max())
        axes[1, k].set_title(fr'$s_t$ $(t={(k/10):.1f})$', fontsize=12)
        axes[1, k].text(0.96, 0.04, f'max: {st[k].max():.2f}\nmin: {st[k].min():.2f}',
                                transform=axes[1, k].transAxes,
                            fontsize=8, verticalalignment='bottom',
                            horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='white',
                                        alpha=0.5))  

        # k0
        axes[2, k].imshow(k0[k], cmap='hot')
        axes[2, k].set_title(r'$k_0$', fontsize=12)
        axes[2, k].text(0.96, 0.04, f'max: {k0[k].max():.2f}\nmin: {k0[k].min():.2f}',
                                transform=axes[2, k].transAxes,
                            fontsize=8, verticalalignment='bottom',
                            horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='white',
                                        alpha=0.5))  

        # kt
        axes[3, k].imshow(kt[k], cmap='hot')
        axes[3, k].set_title(fr'$k_t$ $(t={(k/10):.1f})$', fontsize=12)
        axes[3, k].text(0.96, 0.04, f'max: {kt[k].max():.2f}\nmin: {kt[k].min():.2f}',
                                transform=axes[3, k].transAxes,
                            fontsize=8, verticalalignment='bottom',
                            horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='white',
                                        alpha=0.5))  
        # y
        axes[4, k].imshow(y[k], cmap='bone')
        axes[4, k].set_title(r'$y$', fontsize=12)
        axes[4, k].text(0.96, 0.04, f'max: {y[k].max():.2f}\nmin: {y[k].min():.2f}',
                                transform=axes[4, k].transAxes,
                            fontsize=8, verticalalignment='bottom',
                            horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='white',
                                        alpha=0.5))

    # Saving the figure
    config_name = sys.argv[1].split('config_')[1].replace('.yaml', '')
    if config.dataset.name == 'SKIRT_EPL':
        save_dir = os.path.join(config.skirt_epl_dataset.save_path, f'data_demo_{config_name}.pdf')
    elif config.dataset.name == 'SKIRT_TNG':
        save_dir = os.path.join(config.skirt_tng_dataset.save_path, f'data_demo_{config_name}.pdf')
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.1, wspace=0.1) 
    plt.savefig(save_dir, dpi=150)
    print(f"Plot saved to {save_dir}")

def main(config):

    # Load datasets
    if config.dataset.name == 'SKIRT_EPL':
        (_, _, _, train_loader, _, _
         ) = load_datasets(save_path = config.skirt_epl_dataset.save_path, 
                           batch_size = 11,
                           augment = config.skirt_epl_dataset.augment)
    elif config.dataset.name == 'SKIRT_TNG':
        (_, _, _, train_loader, _, _
         ) = load_datasets(save_path = config.skirt_tng_dataset.save_path, 
                           batch_size = 11,
                           augment = config.skirt_tng_dataset.augment)
        
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
    rim = RIM(net = None, 
            lensingmodel = lensingmodel, 
            sde = sde, 
            space_kappa = config.sde.space_kappa, 
            grad_lik = config.rim.grad_lik, 
            loss = config.loss, 
            num_iterations = config.rim.num_iterations,
            use_log_t = config.rim.use_log_t,
            device = 'cuda')

    # Get test examples from the loader
    s0, k0 = next(iter(train_loader))
    # Clean source and kappa map
    s0 = s0.to(rim.device).float()
    k0 = k0.to(rim.device).float()
    # Converting kappa map to RIM units
    k0 = rim.caustics_to_rim(k0)
    # Defining diffusion time t
    t = torch.tensor([config.sde.epsilon,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], device=rim.device)
    # Generating noisy source and kappa map, and lensed image
    t, st, kt, y = rim.generate_batch(t=t, s0=s0, k0=k0)

    # Making a plot showing s0, st, k0, kt, y
    demo_plot(s0, st, k0, kt, y, config)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python demo_dataset.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    config = Config.parse_obj(data)
    
    main(config)