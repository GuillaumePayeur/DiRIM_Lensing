import sys
import caustics
import yaml
import torch
import numpy as np
import os
import h5py
from torch.nn.functional import avg_pool2d

from dirim_lensing import Config

def main(config):
    print('Creating SKIRT lensing datasets...')

    # Create source images first
    print('Creating source images...')

    # Load raw data file
    filename = "./data/skirt64_grizy_microjy.h5"
    with h5py.File(filename, "r") as f:
        images = torch.tensor(f['images'][:,2])

    # Filter galaxies that are too small or touch the edge
    images_max1 = images/images.amax(dim=(1,2), keepdim=True)
    sums = torch.sum(images_max1,axis=(1,2))

    indices_remove = []
    for i in range(images_max1.shape[0]):
        if sums[i] < 20:
            indices_remove.append(i)
        if (torch.max(images_max1[i][0:3,:]) > 0.02 or 
                torch.max(images_max1[i][-3:,:]) > 0.02 or
                torch.max(images_max1[i][:,0:3]) > 0.02 or 
                torch.max(images_max1[i][:,-3:]) > 0.02):
            indices_remove.append(i)

    indices_remove = torch.tensor(indices_remove)
    indices = torch.tensor([i for i in range(images.shape[0]) if i not in indices_remove])
    sources = images[indices]

    # Normalize images so that their peak intensity is uniformly distributed between 0.9 and 1.0
    max = torch.amax(sources, dim=(1,2), keepdim=True)
    sources = sources * (torch.rand(sources.shape[0], 1, 1)*0.1 + 0.9) / max

    # Validate that split fractions add up to 1
    train_split=config.skirt_epl_dataset.train_split # Training set proportion
    validation_split=config.skirt_epl_dataset.validation_split # Validation set proportion
    test_split=config.skirt_epl_dataset.test_split # Test set proportion
    if abs(train_split + validation_split + test_split - 1.0) > 1e-6:
        raise ValueError("train_split + validation_split + test_split must equal 1.0")
    
    # Calculate split sizes
    total_size = sources.shape[0]
    train_size = int(train_split * total_size)
    val_size = int(validation_split * total_size)
    test_size = total_size - train_size - val_size

    # Split the dataset with fixed seed for reproducibility
    s0_train, s0_val, s0_test = torch.utils.data.random_split(
        sources,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    s0_train = s0_train[:].reshape(train_size, 1, 64, 64).float()
    s0_val = s0_val[:].reshape(val_size, 1, 64, 64).float()
    s0_test = s0_test[:].reshape(test_size, 1, 64, 64).float()

    # Create kappa maps next
    print('Creating kappa maps...')

    # Array to hold kappa maps
    k0 = torch.empty((total_size, 1, 64, 64), dtype=torch.float32)

    # Array to hold kappa map parameters
    k0_params = torch.empty((total_size, 14), dtype=torch.float32)

    # Lens center
    x0 = torch.rand(total_size)*(config.skirt_epl_dataset.x0_max - config.skirt_epl_dataset.x0_min) + config.skirt_epl_dataset.x0_min
    y0 = torch.rand(total_size)*(config.skirt_epl_dataset.y0_max - config.skirt_epl_dataset.y0_min) + config.skirt_epl_dataset.y0_min 
    # EPL parameters
    q = torch.rand(total_size)*(config.skirt_epl_dataset.q_max - config.skirt_epl_dataset.q_min) + config.skirt_epl_dataset.q_min
    phi = torch.rand(total_size)*(config.skirt_epl_dataset.phi_max - config.skirt_epl_dataset.phi_min) + config.skirt_epl_dataset.phi_min
    Rein = torch.rand(total_size)*(config.skirt_epl_dataset.Rein_max - config.skirt_epl_dataset.Rein_min) + config.skirt_epl_dataset.Rein_min
    tau = torch.rand(total_size)*(config.skirt_epl_dataset.tau_max - config.skirt_epl_dataset.tau_min) + config.skirt_epl_dataset.tau_min
    # m=3 multipole parameters
    am3 = torch.rand(total_size)*(config.skirt_epl_dataset.am3_max - config.skirt_epl_dataset.am3_min) + config.skirt_epl_dataset.am3_min
    thetam3 = torch.rand(total_size)*(config.skirt_epl_dataset.thetam3_max - config.skirt_epl_dataset.thetam3_min) + config.skirt_epl_dataset.thetam3_min
    # m=4 multipole parameters
    am4 = torch.rand(total_size)*(config.skirt_epl_dataset.am4_max - config.skirt_epl_dataset.am4_min) + config.skirt_epl_dataset.am4_min
    thetam4 = torch.rand(total_size)*(config.skirt_epl_dataset.thetam4_max - config.skirt_epl_dataset.thetam4_min) + config.skirt_epl_dataset.thetam4_min

    # Subhalo parameters
    logM_min = config.skirt_epl_dataset.logM_min
    logM_max = config.skirt_epl_dataset.logM_max
    c_min = config.skirt_epl_dataset.c_min
    c_max = config.skirt_epl_dataset.c_max

    # Filling in array of lens parameters
    k0_params[:, 0] = x0
    k0_params[:, 1] = y0
    k0_params[:, 2] = q
    k0_params[:, 3] = phi
    k0_params[:, 4] = Rein
    k0_params[:, 5] = tau
    k0_params[:, 6] = am3
    k0_params[:, 7] = thetam3
    k0_params[:, 8] = am4
    k0_params[:, 9] = thetam4

    # creating the kappa maps
    for i in range(total_size):
        # cosmology
        cosmo = caustics.FlatLambdaCDM(name="cosmo")    
        # EPL main halo
        lens_epl = caustics.EPL(cosmology=cosmo, name="epl", x0=x0[i], y0=y0[i], q=q[i], 
                                phi=phi[i], Rein=Rein[i], t=tau[i])
        # m=3 multipole
        multipole3 = caustics.lenses.multipole.Multipole(cosmology=cosmo, name='multipole3', 
                                                        x0=x0[i], y0=y0[i], a_m=am3[i], phi_m=thetam3[i], m=3)
        # m=4 multipole
        multipole4 = caustics.lenses.multipole.Multipole(cosmology=cosmo, name='multipole4', 
                                                        x0=x0[i], y0=y0[i], a_m=am4[i], phi_m=thetam4[i], m=4)       
        # subhalos
        subhalos = []
        # number of subhalos
        N = np.random.randint(2)
        if N > 0:
            for j in range(N):
                M_sub = torch.pow(10.0, (torch.rand(1)*(logM_max - logM_min)) + (logM_min))
                r_sub = torch.rand(1)*config.skirt_epl_dataset.pixelscale*12 + config.skirt_epl_dataset.pixelscale*8
                theta_sub = torch.rand(1)*np.pi*2
                x_sub = r_sub * torch.cos(theta_sub) + x0[i]
                y_sub = r_sub * torch.sin(theta_sub) + y0[i]
                c = torch.rand(1)*(c_max - c_min) + c_min
                subhalos.append(caustics.lenses.nfw.NFW(cosmology=cosmo, name=f'subhalo{j}',
                                            x0=x_sub.item(), y0=y_sub.item(), mass=M_sub.item(), c=c.item()))

                # filling in array of lens parameters
                k0_params[i, 10] = x_sub
                k0_params[i, 11] = y_sub
                k0_params[i, 12] = torch.log10(M_sub)
                k0_params[i, 13] = c
        else:
            # filling in array of lens parameters
            k0_params[i, 10] = float('nan')
            k0_params[i, 11] = float('nan')
            k0_params[i, 12] = float('nan')
            k0_params[i, 13] = float('nan')

        # lens model with all components
        z_s = config.skirt_epl_dataset.z_s
        z_l = config.skirt_epl_dataset.z_l
        lens = caustics.SinglePlane(cosmo, name='lens', z_s=z_s, z_l=z_l, 
                        lenses=(lens_epl, multipole3, multipole4, *subhalos))
        
        # rendering kappa map onto grid
        quad_level_kappa = config.skirt_epl_dataset.quad_level_kappa
        upsample_factor_kappa = config.skirt_epl_dataset.upsample_factor_kappa
        pixelscale = config.skirt_epl_dataset.pixelscale
        res = config.dataset.res
        if quad_level_kappa:
            X, Y = caustics.utils.meshgrid(pixelscale/upsample_factor_kappa, res*upsample_factor_kappa, res*upsample_factor_kappa)
            Xs, Ys, weight = caustics.utils.gaussian_quadrature_grid(pixelscale/upsample_factor_kappa, X, Y, quad_level_kappa)
            k0[i] = avg_pool2d((caustics.utils.gaussian_quadrature_integrator(lens.convergence(Xs, Ys), weight)).unsqueeze(0), kernel_size=upsample_factor_kappa, stride=upsample_factor_kappa)

        if not quad_level_kappa:
            X, Y = caustics.utils.meshgrid(pixelscale/upsample_factor_kappa, res*upsample_factor_kappa, res*upsample_factor_kappa)
            k0[i] = avg_pool2d((lens.convergence(X, Y)).unsqueeze(0), kernel_size=upsample_factor_kappa, stride=upsample_factor_kappa)

    # split kappa maps
    k0_train = k0[0:train_size].clone()
    k0_val = k0[train_size:train_size+val_size].clone()
    k0_test = k0[train_size+val_size:total_size].clone()

    # split kappa map params
    k0_params_train = k0_params[0:train_size].clone()
    k0_params_val = k0_params[train_size:train_size+val_size].clone()
    k0_params_test = k0_params[train_size+val_size:total_size].clone()

    # Save datasets to file
    save_path = config.skirt_epl_dataset.save_path
    os.makedirs(save_path, exist_ok=True)
    torch.save(s0_train, os.path.join(save_path, 'source_train.pt'))
    torch.save(s0_val, os.path.join(save_path, 'source_val.pt'))
    torch.save(s0_test, os.path.join(save_path, 'source_test.pt'))
    torch.save(k0_train, os.path.join(save_path, 'kappa_train.pt'))
    torch.save(k0_val, os.path.join(save_path, 'kappa_val.pt'))
    torch.save(k0_test, os.path.join(save_path, 'kappa_test.pt'))
    torch.save(k0_params_train, os.path.join(save_path, 'kappa_params_train.pt'))
    torch.save(k0_params_val, os.path.join(save_path, 'kappa_params_val.pt'))
    torch.save(k0_params_test, os.path.join(save_path, 'kappa_params_test.pt'))
    print(f"Datasets saved to {save_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python create_dataset_skirt_epl.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    config = Config.parse_obj(data)
    
    main(config)