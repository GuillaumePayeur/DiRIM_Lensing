import sys
import torch
import yaml
import os
import h5py
import caustics
import math
import tqdm
from torch.nn.functional import avg_pool2d

from dirim_lensing import Config
from dirim_lensing import LensingModel
from dirim_lensing import RIM
from dirim_lensing import SDE
from dirim_lensing import Sampler
from dirim_lensing import load_datasets
from train_rim import load_unet

def save_h5(idx, s, k, y, y_noiseless, s_samples, k_samples, y_samples, model_name):
    os.makedirs('./results/test_sample_large', exist_ok=True)

    with h5py.File(f"./results/test_sample_large/test_sample_large_{model_name}.h5", "a") as f: 
        if "idx" not in f:
            # Create expandable datasets
            f.create_dataset("idx", data=[idx], maxshape=(None,))
            f.create_dataset("s", data=[s], 
                            maxshape=(None,) + tuple(s.shape),
                            chunks=True)
            f.create_dataset("k", data=[k], 
                            maxshape=(None,) + tuple(k.shape),
                            chunks=True)
            f.create_dataset("y", data=[y], 
                            maxshape=(None,) + tuple(y.shape),
                            chunks=True)
            f.create_dataset("y_noiseless", data=[y_noiseless], 
                            maxshape=(None,) + tuple(y_noiseless.shape),
                            chunks=True)
            f.create_dataset("s_samples", data=[s_samples], 
                            maxshape=(None,) + tuple(s_samples.shape),
                            chunks=True)
            f.create_dataset("k_samples", data=[k_samples], 
                            maxshape=(None,) + tuple(k_samples.shape),
                            chunks=True)
            f.create_dataset("y_samples", data=[y_samples], 
                            maxshape=(None,) + tuple(y_samples.shape),
                            chunks=True)
            
        else:
            # Appending data
            N = f["idx"].shape[0]
            
            f["idx"].resize((N + 1,))
            f["idx"][N] = idx

            f["s"].resize((N + 1,) + f["s"].shape[1:])
            f["s"][N] = s

            f["k"].resize((N + 1,) + f["k"].shape[1:])
            f["k"][N] = k

            f["y"].resize((N + 1,) + f["y"].shape[1:])
            f["y"][N] = y

            f["y_noiseless"].resize((N + 1,) + f["y_noiseless"].shape[1:])
            f["y_noiseless"][N] = y_noiseless

            f["s_samples"].resize((N + 1,) + f["s_samples"].shape[1:])
            f["s_samples"][N] = s_samples

            f["k_samples"].resize((N + 1,) + f["k_samples"].shape[1:])
            f["k_samples"][N] = k_samples

            f["y_samples"].resize((N + 1,) + f["y_samples"].shape[1:])
            f["y_samples"][N] = y_samples

def main(config):
    # Load datasets
    if config.dataset.name == 'SKIRT_TNG':
        (_, _, test_dataset, _, _, _
         ) = load_datasets(save_path = config.skirt_tng_dataset.save_path, 
                           batch_size = 1,
                           augment=False)
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
    if config.dataset.name == 'SKIRT_TNG':
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
    
    # Number of samples to generate per test example
    n_samples = config.tests.n_samples

    # Get test examples from the dataset
    for idx in config.tests.sample_idxs:
        s0, k0 = test_dataset[idx]
        # Clean source and kappa map
        s0 = s0.to(rim.device).float()
        k0 = k0.to(rim.device).float()
        # Converting kappa map to RIM units
        k0 = rim.caustics_to_rim(k0)
        # Get lensed image from test set
        y = test_observations[idx].to(rim.device)
        # Generating noiseless lensed image
        y_noiseless = lensingmodel.simulate_lensing(s0, rim.rim_to_caustics(k0), noise=False)
        # Arrays to hold the samples and other information
        s_samples = torch.empty(n_samples, config.dataset.res, config.dataset.res)
        k_samples = torch.empty(n_samples, config.dataset.res, config.dataset.res)
        y_samples = torch.empty(n_samples, config.dataset.res, config.dataset.res)
        print("Starting sampling...")
        for i in tqdm.tqdm(range(n_samples // config.tests.batch_size), desc=f"Sampling test index {idx}"):
            # Generating samples
            s_samples_batch, k_samples_batch = sampler.sample_PC(y, num_samples=config.tests.batch_size)
            s_samples_batch = s_samples_batch.view(config.tests.batch_size, 1, config.dataset.res, config.dataset.res)
            k_samples_batch = k_samples_batch.view(config.tests.batch_size, 1, config.dataset.res, config.dataset.res)
            y_samples_batch = lensingmodel.simulate_lensing(s_samples_batch, rim.rim_to_caustics(k_samples_batch), noise=False)
            # Filling in arrays
            s_samples[i*config.tests.batch_size:(i+1)*config.tests.batch_size] = s_samples_batch[:,0,:,:]
            k_samples[i*config.tests.batch_size:(i+1)*config.tests.batch_size] = k_samples_batch[:,0,:,:]
            y_samples[i*config.tests.batch_size:(i+1)*config.tests.batch_size] = y_samples_batch[:,0,:,:]
        if n_samples % config.tests.batch_size:
            # Generating samples
            s_samples_batch, k_samples_batch = sampler.sample_PC(y, num_samples=n_samples % config.tests.batch_size)
            s_samples_batch = s_samples_batch.view(n_samples % config.tests.batch_size, 1, config.dataset.res, config.dataset.res)
            k_samples_batch = k_samples_batch.view(n_samples % config.tests.batch_size, 1, config.dataset.res, config.dataset.res)
            y_samples_batch = lensingmodel.simulate_lensing(s_samples_batch, rim.rim_to_caustics(k_samples_batch), noise=False)
            # Filling in arrays
            s_samples[n_samples // config.tests.batch_size * config.tests.batch_size:] = s_samples_batch[:,0,:,:]
            k_samples[n_samples // config.tests.batch_size * config.tests.batch_size:] = k_samples_batch[:,0,:,:]
            y_samples[n_samples // config.tests.batch_size * config.tests.batch_size:] = y_samples_batch[:,0,:,:]
        # Reshaping and sending to cpu
        s0 = s0.view(config.dataset.res, config.dataset.res).cpu()
        k0 = k0.view(config.dataset.res, config.dataset.res).cpu()
        y = y.view(config.dataset.res, config.dataset.res).cpu()
        y_noiseless = y_noiseless.view(config.dataset.res, config.dataset.res).cpu()
        s_samples = s_samples.cpu()
        k_samples = k_samples.cpu()
        y_samples = y_samples.cpu()

        # Save the data to a h5 file
        save_h5(idx, s0, k0, y, y_noiseless, s_samples, k_samples, y_samples, model_name)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_skirt_tng.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    config = Config.parse_obj(data)
    
    main(config)