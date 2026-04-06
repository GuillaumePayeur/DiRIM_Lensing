import torch
from caustics import Pixelated, PixelatedConvergence, LensSource, FlatLambdaCDM
from caustics.utils import gaussian as caustics_gaussian

class LensingModel:
    def __init__(self, res, source_pixelscale, pixelscale, z_s, z_l, psf_sigma, sigma_y, upsample_factor, device):
        self.res = res
        self.source_pixelscale = source_pixelscale
        self.pixelscale = pixelscale
        self.z_s = z_s
        self.z_l = z_l
        self.psf_sigma = psf_sigma
        self.sigma_y = sigma_y
        self.upsample_factor = upsample_factor
        self.device = device

        self.simulator = self.create_simulator()

    def create_simulator(self):
        '''
        Create a Caustics lensing simulator generating lensed images from a 
        pixelated source and convergence map.
        '''
        # Set up cosmology model (flat Lambda-CDM)
        cosmo = FlatLambdaCDM(name="cosmo")

        # Define the source as a pixelated image
        source = Pixelated(name="source", shape=(self.res, self.res), 
                           pixelscale=self.source_pixelscale, x0=0, y0=0)               

        # Define the lens as a pixelated image
        lens = PixelatedConvergence(cosmology=cosmo, name="lens", 
                                    pixelscale=self.pixelscale, 
                                    shape=(self.res, self.res),
                                    z_l=self.z_l, z_s=self.z_s)

        # Define the gaussian PSF
        if self.psf_sigma > 0:
            psf_image = caustics_gaussian(nx=self.res*self.upsample_factor//2, 
                                          ny=self.res*self.upsample_factor//2, 
                                          pixelscale=self.pixelscale/self.upsample_factor,
                                          sigma=self.psf_sigma)

        # Create the complete lensing simulator
        simulator = LensSource(lens, source, pixelscale=self.pixelscale, 
                               pixels_x=self.res, 
                               psf=psf_image if self.psf_sigma > 0 else None,
                               upsample_factor=self.upsample_factor,
                               psf_mode='fft' if self.upsample_factor == 1 else 'conv2d')
        
        return simulator.to(self.device)

    def simulate_lensing(self, s, k, noise=False):
        '''
        Create a lensed image given a source s and convergence map k.
        '''
        # Construct simulator input
        simulator_input = torch.cat((k.reshape(-1, self.res**2), s.reshape(-1, self.res**2)), dim=1)

        # Create simulation
        y = torch.vmap(self.simulator)(simulator_input)

        # Add Gaussian noise if specified
        if noise:
            y = y + self.sigma_y * torch.randn_like(y)
        
        # Return the lensed image
        if s.dim() == 3:
            return y
        elif s.dim() == 4:
            return y.unsqueeze(1)
        else:
            raise ValueError(f"Expected s to have 3 or 4 dimensions, got {s.dim()}")
        
    def neg_log_likelihood(self, y, y_hat):
        '''
        Negative log-likelihood of lensed image prediction yhat given 
        observation y.
        '''
        residuals = (y - y_hat) / self.sigma_y
        nll = 0.5 * torch.sum(residuals**2)
        return residuals, nll