import torch
import math

class SDE:
    def __init__(self, kind, epsilon, sigma_min=None, sigma_max=None, beta_min=None, beta_max=None):
        self.kind = kind
        self.epsilon = epsilon
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t):
        if self.kind == 'VP_linear':
            return self.beta_min + t * (self.beta_max - self.beta_min)
        elif self.kind == 'VP_exp':
            return self.beta_min * (self.beta_max / self.beta_min) ** t
        else:
            return None

    def alpha(self, t):
        if self.kind == 'VP_linear':
            return torch.exp(-0.5 * (self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2))
        elif self.kind == 'VP_exp':
            return torch.exp(-0.5 * (self.beta(t) - self.beta_min) / math.log(self.beta_max / self.beta_min))
        else:
            return None

    def sigma(self, t):
        if self.kind == 'VE':
            return self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        elif self.kind in {'VP_linear', 'VP_exp'}:
            return torch.sqrt(1 - self.alpha(t)**2)
        else:
            raise ValueError(f"Unsupported SDE kind {self.kind}")

    def forward_process(self, x0, t):
        """
        Generate xt from x0 given t.
        """
        if self.kind == 'VE':
            sigma = self.sigma(t)
            noise = torch.randn_like(x0, device=x0.device) * sigma[:,None,None,None]
            return x0 + noise
        elif self.kind in {'VP_linear', 'VP_exp'}:
            alpha = self.alpha(t)
            sigma = self.sigma(t)
            noise = torch.randn_like(x0, device=x0.device) * sigma[:,None,None,None]
            return alpha[:,None,None,None] * x0 + noise

    def get_score(self, xt, x0, t):
        """
        Compute score given xt, t and estimate of x0.
        """
        if self.kind == 'VE':
            sigma_t = self.sigma(t)
            score = (x0 - xt) / (sigma_t[:,None,None,None]**2)
            return score
        elif self.kind in {'VP_linear', 'VP_exp'}:
            alpha = self.alpha(t)
            sigma = self.sigma(t)
            score = (x0 * alpha[:,None,None,None] - xt) / sigma[:,None,None,None]**2
            return score
