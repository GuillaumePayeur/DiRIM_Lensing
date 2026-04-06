import torch
import os
import warnings
from torch.utils.data import Dataset, DataLoader

class LensingDataset(Dataset):
    def __init__(self, source_data, kappa_data, augment):
        if len(source_data) != len(kappa_data):
            raise ValueError(
                f"source_data and kappa_data must have the same length, got {len(source_data)} and {len(kappa_data)}"
            )
        self.source_data = source_data
        self.kappa_data = kappa_data
        self.augment = augment

    def __len__(self):
        return len(self.source_data)
    
    def __getitem__(self, idx):
        s0 = self.source_data[idx]
        k0 = self.kappa_data[idx]

        if self.augment:
            s0 = self.augment_D4(s0)
            k0 = self.augment_D4(k0)

        return s0, k0
    
    def augment_D4(self, x):
        '''
        Image augmentation: random flips and rotations (dihedral group D4).
        '''
        # Random rotation (0, 90, 180, or 270 degrees)
        rotation_choice = torch.randint(0, 4, (1,)).item()
        if rotation_choice == 1:  # 90 degrees
            x = torch.rot90(x, k=1, dims=(-2, -1)).contiguous()
        elif rotation_choice == 2:  # 180 degrees
            x = torch.rot90(x, k=2, dims=(-2, -1)).contiguous()
        elif rotation_choice == 3:  # 270 degrees
            x = torch.rot90(x, k=3, dims=(-2, -1)).contiguous()

        # Random horizontal mirroring (flip along x-axis)
        if torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=(-1,)).contiguous()

        return x

def load_datasets(save_path, batch_size, augment, num_workers=0, shuffle_test=False):
    print(f"Loading datasets from {save_path}...")

    # verify that the dataset files exist
    required_files = ['source_train.pt', 'source_val.pt', 'source_test.pt',
                      'kappa_train.pt', 'kappa_val.pt', 'kappa_test.pt']
    for file in required_files:
        file_path = os.path.join(save_path, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"""Required dataset file not found:
                                    {file_path}""") 

    # Load data tensors
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        source_train = torch.load(os.path.join(save_path, 'source_train.pt'))
        source_val = torch.load(os.path.join(save_path, 'source_val.pt'))
        source_test = torch.load(os.path.join(save_path, 'source_test.pt'))
        kappa_train = torch.load(os.path.join(save_path, 'kappa_train.pt'))
        kappa_val = torch.load(os.path.join(save_path, 'kappa_val.pt'))
        kappa_test = torch.load(os.path.join(save_path, 'kappa_test.pt'))        

    # Create datasets
    train_dataset = LensingDataset(source_train, kappa_train, augment=augment)
    val_dataset = LensingDataset(source_val, kappa_val, augment=False)
    test_dataset = LensingDataset(source_test, kappa_test, augment=False)

    # Create data loaders
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=shuffle_test,
                             num_workers=num_workers)

    print(f"""Loaded {len(train_dataset)} training, {len(val_dataset)} validation, and {len(test_dataset)} test samples""")
    
    return (train_dataset, val_dataset, test_dataset, 
            train_loader, val_loader, test_loader)
