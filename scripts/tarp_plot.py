import tarp
import sys
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os

from dirim_lensing import Config

def tarp_plot(ecp, alpha, model_name):
    plt.style.use('dark_background')
    _, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # Calculate mean and standard deviation across bootstrap dimension
    ecp_mean = np.mean(ecp, axis=0)
    ecp_2low = np.percentile(ecp, 2.275, axis=0)
    ecp_2high = np.percentile(ecp, 97.725, axis=0)
    ecp_1low = np.percentile(ecp, 15.865, axis=0)
    ecp_1high = np.percentile(ecp, 84.135, axis=0)

    # Plot the mean
    ax.plot(alpha, ecp_mean, color='gold', label='Mean ECP')
    
    # Plot 1σ and 2σ confidence regions
    ax.fill_between(alpha, ecp_1low, ecp_1high, 
                    alpha=0.25, color='gold', label=r'$1\sigma$ region')
    ax.fill_between(alpha, ecp_2low, ecp_2high, 
                    alpha=0.15, color='gold', label=r'$2\sigma$ region')
    
    # Plot the diagonal reference line
    ax.plot([0, 1], [0, 1], color='white', linestyle='--', 
            label='Perfect calibration')
    
    ax.set_xlabel('Credibility Level')
    ax.set_ylabel('Expected Coverage')
    ax.set_title('TARP Test')
    ax.legend()
    plt.tight_layout()

    # Save plot
    os.makedirs('./results/tarp_plots', exist_ok=True)
    plt.savefig(f'./results/tarp_plots/tarp_test_{model_name}.png')

def main(config):
    # Extracting model name
    model_name = sys.argv[1].split('config_')[1].replace('.yaml', '')    

    # Load samples, theta, and references
    s_samples = np.load(f'./results/tarp_samples/s_samples_{model_name}.npy')
    k_samples = np.load(f'./results/tarp_samples/k_samples_{model_name}.npy')
    s_theta = np.load(f'./results/tarp_samples/s_theta_{model_name}.npy')
    k_theta = np.load(f'./results/tarp_samples/k_theta_{model_name}.npy')
    s_references = np.load(f'./results/tarp_samples/s_references_{model_name}.npy')
    k_references = np.load(f'./results/tarp_samples/k_references_{model_name}.npy')

    # Normalize source and kappa with respect to one another
    std_s = np.std(s_samples)
    std_k = np.std(k_samples)
    s_samples = s_samples / std_s
    k_samples = k_samples / std_k
    s_theta = s_theta / std_s
    k_theta = k_theta / std_k
    s_references = s_references / std_s
    k_references = k_references / std_k

    # Reshape arrays for TARP
    n_sims = config.tarp.n_sims
    n_samples = config.tarp.n_samples
    res = config.dataset.res
    samples = np.concatenate((
        s_samples.reshape(n_samples, n_sims, res*res),
        k_samples.reshape(n_samples, n_sims, res*res)), axis=2)
    theta = np.concatenate((
        s_theta.reshape(n_sims, res*res),
        k_theta.reshape(n_sims, res*res)), axis=1)
    references = np.concatenate((
        s_references.reshape(n_sims, res*res),
        k_references.reshape(n_sims, res*res)), axis=1)
    
    # Run TARP
    num_alpha_bins = config.tarp.num_alpha_bins
    norm = config.tarp.norm
    ecp, alpha = tarp.get_tarp_coverage(samples, theta, references, 
                                        num_alpha_bins=num_alpha_bins, 
                                        bootstrap=True, num_bootstrap=200, 
                                        norm=norm)

    # Make TARP plot
    tarp_plot(ecp, alpha, model_name)

    # Saving ECP and alpha
    os.makedirs('./results/tarp_ecp_alpha', exist_ok=True)
    np.save(f'./results/tarp_ecp_alpha/ecp_{model_name}.npy', ecp)
    np.save(f'./results/tarp_ecp_alpha/alpha_{model_name}.npy', alpha)

    # Run TARP on source only
    samples = s_samples.reshape(n_samples, n_sims, res*res)
    theta = s_theta.reshape(n_sims, res*res)
    references = s_references.reshape(n_sims, res*res)
    ecp_s, alpha_s = tarp.get_tarp_coverage(samples, theta, references, 
                                            num_alpha_bins=num_alpha_bins, 
                                            bootstrap=True, num_bootstrap=200, 
                                            norm=norm)
    
    # Make TARP plot for source only
    tarp_plot(ecp_s, alpha_s, model_name+'_source_only')

    # Saving ECP and alpha for source only
    np.save(f'./results/tarp_ecp_alpha/ecp_{model_name}_source_only.npy', ecp_s)
    np.save(f'./results/tarp_ecp_alpha/alpha_{model_name}_source_only.npy', alpha_s)

    # Run TARP on kappa only
    samples = k_samples.reshape(n_samples, n_sims, res*res)
    theta = k_theta.reshape(n_sims, res*res)
    references = k_references.reshape(n_sims, res*res)
    ecp_k, alpha_k = tarp.get_tarp_coverage(samples, theta, references, 
                                            num_alpha_bins=num_alpha_bins, 
                                            bootstrap=True, num_bootstrap=200, 
                                            norm=norm)
    # Make TARP plot for kappa only
    tarp_plot(ecp_k, alpha_k, model_name+'_kappa_only')

    # Saving ECP and alpha for kappa only
    np.save(f'./results/tarp_ecp_alpha/ecp_{model_name}_kappa_only.npy', ecp_k)
    np.save(f'./results/tarp_ecp_alpha/alpha_{model_name}_kappa_only.npy', alpha_k)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tarp_plot.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    config = Config.parse_obj(data)
    
    main(config)