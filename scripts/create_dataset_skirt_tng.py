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
    print('Creating SKIRT source and TNG halo lensing datasets...')

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
    train_split=config.skirt_tng_dataset.train_split # Training set proportion
    validation_split=config.skirt_tng_dataset.validation_split # Validation set proportion
    test_split=config.skirt_tng_dataset.test_split # Test set proportion
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

    # Load raw data file
    filename = "./data/hkappa128hst_TNG100_rau_trainset.h5"
    with h5py.File(filename, "r") as f:
        k0 = images = torch.tensor(f['kappa'][0:total_size])

    # Resize to 64x64
    k0 = avg_pool2d(k0, kernel_size=2, stride=2)

    # Split kappa maps contiguously to keep augmentations of the same kappa map 
    # together, then shuffle within each split only.
    k0_train = k0[:train_size]
    k0_val = k0[train_size:train_size + val_size]
    k0_test = k0[train_size + val_size:train_size + val_size + test_size]

    generator = torch.Generator().manual_seed(42)
    k0_train = k0_train[torch.randperm(train_size, generator=generator)]
    k0_val = k0_val[torch.randperm(val_size, generator=generator)]
    k0_test = k0_test[torch.randperm(test_size, generator=generator)]

    k0_train = k0_train.reshape(train_size, 1, 64, 64).float()
    k0_val = k0_val.reshape(val_size, 1, 64, 64).float()
    k0_test = k0_test.reshape(test_size, 1, 64, 64).float()

    # Save datasets to file
    save_path = config.skirt_tng_dataset.save_path
    os.makedirs(save_path, exist_ok=True)
    torch.save(s0_train, os.path.join(save_path, 'source_train.pt'))
    torch.save(s0_val, os.path.join(save_path, 'source_val.pt'))
    torch.save(s0_test, os.path.join(save_path, 'source_test.pt'))
    torch.save(k0_train, os.path.join(save_path, 'kappa_train.pt'))
    torch.save(k0_val, os.path.join(save_path, 'kappa_val.pt'))
    torch.save(k0_test, os.path.join(save_path, 'kappa_test.pt'))
    print(f"Datasets saved to {save_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python create_dataset_skirt_tng.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    config = Config.parse_obj(data)
    
    main(config)