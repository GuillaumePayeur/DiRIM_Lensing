import sys
import yaml
import torch
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

from dirim_lensing import Config
from dirim_lensing import LensingModel
from dirim_lensing import RIM
from dirim_lensing import SDE
from dirim_lensing import SongUNet
from dirim_lensing import load_datasets

def create_unet(config):
    net = SongUNet(img_resolution = config.dataset.res,
                    in_channels = 8 if config.rim.use_residuals else 7,
                    out_channels = 2,
                    label_dim = 0,
                    augment_dim = 0,
                    model_channels = config.rim.model_channels,
                    channel_mult = config.rim.channel_mult,
                    channel_mult_emb = 4,
                    num_blocks = config.rim.num_blocks,
                    attn_resolutions = config.rim.attn_resolutions,
                    dropout = config.training.dropout,
                    label_dropout = 0,
                    embedding_type = config.rim.embedding_type,
                    channel_mult_noise = config.rim.channel_mult_noise,
                    encoder_type = config.rim.encoder_type,
                    decoder_type = config.rim.decoder_type,
                    resample_filter = config.rim.resample_filter,
                    use_residuals = config.rim.use_residuals,
                    mem_type = config.rim.memory.type,
                    device = 'cuda')
    net.to('cuda').train()
    
    return net

def load_unet(config, model_name, epoch):
    # Load the model
    model_path = os.path.join('./results/model_weights', f'{model_name}_epoch{epoch}')
    net = create_unet(config)
    net.load_state_dict(torch.load(f"{model_path}.pt",
                                    map_location='cuda',
                                    weights_only=True))

    return net

def save_model(net, epoch, model_name):
    # Create save path
    save_path = os.path.join('./results/model_weights', f'{model_name}_epoch{epoch+1}')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save model state dict as PyTorch file
    torch.save(net.state_dict(), f"{save_path}.pt")

def load_loss_history(model_name, start_epoch):
    # Paths to loss history files
    train_history_path = os.path.join(f"./results/loss_histories", f"{model_name}_train_loss_history.txt")
    val_history_path = os.path.join(f"./results/loss_histories", f"{model_name}_val_loss_history.txt")
    
    # Loading files
    train_loss_history = np.loadtxt(train_history_path)[0:start_epoch].tolist()
    val_loss_history = np.loadtxt(val_history_path)[0:start_epoch].tolist()

    return train_loss_history, val_loss_history

def plot_loss_curves(train_loss_history, val_loss_history, model_name):
    # Extract number of epochs
    epochs = np.arange(1, len(train_loss_history) + 1)

    # Create the plot
    plt.style.use('dark_background')
    plt.figure(figsize=(6, 4))

    # Plot loss on log scale
    plt.semilogy(epochs, train_loss_history, color='aqua', linewidth=2, label='Training Loss')
    plt.semilogy(epochs, val_loss_history, color='orange', linewidth=2, label='Validation Loss')

    # Add formatting
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.1)
    plt.tight_layout()

    # Save the loss histories and the plot
    os.makedirs('./results/loss_histories', exist_ok=True)
    train_history_path = os.path.join(f"./results/loss_histories", f"{model_name}_train_loss_history.txt")
    val_history_path = os.path.join(f"./results/loss_histories", f"{model_name}_val_loss_history.txt")
    np.savetxt(train_history_path, np.array(train_loss_history))
    np.savetxt(val_history_path, np.array(val_loss_history))
    os.makedirs('./results/loss_curves', exist_ok=True)
    plot_path = os.path.join(f"./results/loss_curves", f"{model_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

def main(config):
    # Load datasets
    if config.dataset.name == 'SKIRT_EPL':
        (train_dataset, _, _, train_loader, val_loader, _
         ) = load_datasets(save_path = config.skirt_epl_dataset.save_path, 
                           batch_size = config.training.batch_size,
                           augment = config.skirt_epl_dataset.augment)
    elif config.dataset.name == 'SKIRT_TNG':
        (train_dataset, _, _, train_loader, val_loader, _
         ) = load_datasets(save_path = config.skirt_tng_dataset.save_path, 
                           batch_size = config.training.batch_size,
                           augment = config.skirt_tng_dataset.augment)

    # Extracting model name
    model_name = sys.argv[1].split('config_')[1].replace('.yaml', '')

    # Initializing or loading Unet model
    if config.training.resume_train:
        print("Loading trained Unet model...")
        net = load_unet(config, model_name, config.training.start_epoch)
        net.to('cuda').train()
        train_loss_history, val_loss_history = load_loss_history(model_name, config.training.start_epoch)
        print("Model loaded successfully.")
    else:
        print("Initializing new Unet model...")
        net = create_unet(config)
        train_loss_history, val_loss_history = [], []
        print(f"Model initialized succesfully. The model has {sum(p.numel() for p in net.parameters())} parameters")

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
    
    # Setting up optimizer, learning rate scheduler and EMA
    optimizer = torch.optim.Adam(net.parameters(), lr=config.training.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.training.lr_decay)
    if config.training.ema_decay:
        ema = ExponentialMovingAverage(net.parameters(), decay=config.training.ema_decay)

    # Initializing patience
    patience = config.training.patience
    best_val_loss = 1e10
    epochs_without_improvement = 0

    # Main training loop
    start_epoch = config.training.start_epoch if config.training.resume_train else 0
    num_epochs = config.training.num_epochs
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(start_epoch, start_epoch + num_epochs):
        train_epoch_loss = 0.0
        train_num_batches = 0
        val_epoch_loss = 0.0
        val_num_batches = 0

        # Training loop
        rim.net.train()

        # Restoring original weights after evaluation if EMA is used
        if config.training.ema_decay and epoch > start_epoch:
            ema.restore()

        print(f'Starting epoch {epoch+1}/{start_epoch+num_epochs}. Epochs without improvement: {epochs_without_improvement}/{patience}')
        # Shuffling source and kappa training data independently
        train_dataset.source_data = train_dataset.source_data[torch.randperm(len(train_dataset.source_data))]
        train_dataset.kappa_data = train_dataset.kappa_data[torch.randperm(len(train_dataset.kappa_data))]
        for s0, k0 in tqdm(train_loader, desc=f'Epoch {epoch+1}/{start_epoch+num_epochs}'):
            # Clean source and kappa map
            s0 = s0.to(rim.device).float()
            k0 = k0.to(rim.device).float()
            # Converting kappa map to RIM units
            k0 = rim.caustics_to_rim(k0)
            # Generating noisy source and kappa map, and lensed image
            t, st, kt, y = rim.generate_batch(s0, k0)
            # Forward pass through RIM
            s0_hat_series, k0_hat_series = rim.forward(t, st, kt, y)
            # Computing loss
            loss = rim.loss_fn(t, s0, s0_hat_series, k0, k0_hat_series)
            # Backward pass
            optimizer.zero_grad()
            loss.backward(inputs=list(rim.net.parameters()))
            # Gradient clipping
            if config.training.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(rim.net.parameters(), config.training.gradient_clipping)
            # Updating weights
            optimizer.step()
            # Updating EMA
            if config.training.ema_decay:
                ema.update()
            # Updating epoch loss and batch count
            train_epoch_loss += loss.item()
            train_num_batches += 1

        # Average loss over batches
        train_avg_epoch_loss = train_epoch_loss / train_num_batches
        train_loss_history.append(train_avg_epoch_loss)

        # Validation loop
        rim.net.eval()

        # Switching to EMA weights for evaluation if EMA is used
        if config.training.ema_decay:
            ema.store()
            ema.copy_to()

        for s0, k0 in tqdm(val_loader, desc=f'Epoch {epoch+1}/{start_epoch+num_epochs}'):
            # Clean source and kappa map
            s0 = s0.to(rim.device).float()
            k0 = k0.to(rim.device).float()
            # Converting kappa map to RIM units
            k0 = rim.caustics_to_rim(k0)
            # Generating noisy source and kappa map, and lensed image
            t, st, kt, y = rim.generate_batch(s0, k0)
            # Forward pass through RIM
            s0_hat_series, k0_hat_series, _, _, _ = rim.forward_eval(t, st, kt, y)
            # Computing loss
            loss = rim.loss_fn(t, s0, s0_hat_series, k0, k0_hat_series)
            # Updating epoch loss and batch count
            val_epoch_loss += loss.item()
            val_num_batches += 1

        # Average loss over batches
        val_avg_epoch_loss = val_epoch_loss / val_num_batches
        val_loss_history.append(val_avg_epoch_loss)

        # Plot loss curves
        plot_loss_curves(train_loss_history, val_loss_history, model_name)

        # Early stopping check
        if val_avg_epoch_loss < best_val_loss:
            best_val_loss = val_avg_epoch_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"No improvement in validation loss for {patience} epochs, early stopping triggered at epoch {epoch + 1}.")
            print("Saving trained Unet model...")        
            # Saving the model
            save_model(rim.net, epoch, model_name)
            print("Model saved successfully.")
            break

        # Update learning rate
        scheduler.step()

        # Saving the model every 5 epochs and at the last epoch
        if (epoch + 1) % 5 == 0 or epoch == start_epoch + num_epochs - 1:
            print("Saving trained Unet model...")        
            save_model(rim.net, epoch, model_name)
            print("Model saved successfully.")

    print("Training completed!")   

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_rim.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    config = Config.parse_obj(data)
    
    main(config)
    
