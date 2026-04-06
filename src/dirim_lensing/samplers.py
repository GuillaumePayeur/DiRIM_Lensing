import torch

class Sampler:
    def __init__(self, rim, sampler_name, num_steps, jump_to_0, n_corrector=0, snr=None):
        self.rim = rim
        self.sampler_name = sampler_name
        self.num_steps = num_steps
        self.jump_to_0 = jump_to_0
        self.n_corrector = n_corrector
        self.snr = snr

    def sample_PC(self, y, num_samples):
        '''
        Predictor-corrector sampler.
        '''
        sde = self.rim.sde
        epsilon = sde.epsilon
        
        # time discretization from 1 to t (reverse time)
        times = torch.linspace(1.0, epsilon, self.num_steps + 1).to(self.rim.device)

        # copying y along new dimension for num_samples
        batch_size = y.shape[0]
        y = y.repeat(num_samples, 1, 1, 1, 1).reshape(num_samples*batch_size, 1, self.rim.net.res, self.rim.net.res)

        # samples at t=1
        if sde.kind == 'VE':
            sigma_max = sde.sigma_max
            s_samples = torch.randn(num_samples*batch_size, 1, self.rim.net.res, self.rim.net.res).to(self.rim.device) * sigma_max
            k_samples = torch.randn(num_samples*batch_size, 1, self.rim.net.res, self.rim.net.res).to(self.rim.device) * sigma_max
        elif sde.kind == 'VP_linear' or sde.kind == 'VP_exp':
            s_samples = torch.randn(num_samples*batch_size, 1, self.rim.net.res, self.rim.net.res).to(self.rim.device)
            k_samples = torch.randn(num_samples*batch_size, 1, self.rim.net.res, self.rim.net.res).to(self.rim.device)
        else:
            raise ValueError(f"Unsupported SDE kind {sde.kind}")
        # Predictor-Corrector integration (going backwards in time from t=1 to t=epsilon)
        for i in range(self.num_steps):
            # Compute current time and time at next step
            current_time = times[i]
            next_time = times[i+1]

            # Predictor step
            if self.sampler_name == 'EM':
                s_samples, k_samples = self.EM_step(s_samples, k_samples, y, current_time, next_time, sde)
            elif self.sampler_name == 'Heun':
                s_samples, k_samples = self.Heun_step(s_samples, k_samples, y, current_time, next_time, sde)
            elif self.sampler_name == 'Euler':
                s_samples, k_samples = self.Euler_step(s_samples, k_samples, y, current_time, next_time, sde)
            elif self.sampler_name == 'RK4':
                s_samples, k_samples = self.RK4_step(s_samples, k_samples, y, current_time, next_time, sde)

            # Langevin corrector step(s)
            if self.n_corrector > 0:
                for _ in range(self.n_corrector):
                    score_s, score_k = self.rim.scores(next_time*torch.ones(s_samples.shape[0], device=self.rim.device), s_samples, k_samples, y)
                    z_s = torch.randn_like(s_samples, device=s_samples.device)
                    z_k = torch.randn_like(k_samples, device=k_samples.device)

                    # Langevin step size
                    noise_norm_s = torch.norm(z_s)
                    noise_norm_k = torch.norm(z_k)
                    score_norm_s = torch.norm(score_s)
                    score_norm_k = torch.norm(score_k)
                    eps_s = 2 * (self.snr * noise_norm_s / score_norm_s)**2
                    eps_k = 2 * (self.snr * noise_norm_k / score_norm_k)**2

                    # Drift
                    s_samples += eps_s * score_s
                    k_samples += eps_k * score_k

                    # Diffusion
                    s_samples += torch.sqrt(2 * eps_s) * z_s
                    k_samples += torch.sqrt(2 * eps_k) * z_k                


        # Optionally, jump to t=0 at the last step using denoising model
        if self.jump_to_0:
            t = epsilon
            s_samples, k_samples = self.rim.denoise(t*torch.ones(s_samples.shape[0], device=self.rim.device), s_samples, k_samples, y)

        # reshaping samples back to (num_samples, batch_size, 1, res, res)
        s_samples = s_samples.view(num_samples, -1, 1, self.rim.net.res, self.rim.net.res)
        k_samples = k_samples.view(num_samples, -1, 1, self.rim.net.res, self.rim.net.res)

        return s_samples, k_samples

    def EM_step(self, s_samples, k_samples, y, current_time, next_time, sde):
            '''
            One step of Euler-Maruyama method.
            '''
            score_s, score_k = self.rim.scores(current_time*torch.ones(s_samples.shape[0], device=self.rim.device), s_samples, k_samples, y)
            z_s = torch.randn_like(s_samples, device=s_samples.device)
            z_k = torch.randn_like(k_samples, device=k_samples.device)

            if sde.kind == 'VE':
                current_sigma = sde.sigma(current_time)
                next_sigma = sde.sigma(next_time)
                # Drift
                s_samples += (current_sigma**2 - next_sigma**2)*score_s
                k_samples += (current_sigma**2 - next_sigma**2)*score_k
                # Diffusion
                s_samples += torch.sqrt(current_sigma**2 - next_sigma**2) * z_s
                k_samples += torch.sqrt(current_sigma**2 - next_sigma**2) * z_k
            elif sde.kind == 'VP_linear' or sde.kind == 'VP_exp':
                current_beta = sde.beta(current_time)
                # Drift
                delta_t = current_time - next_time
                s_samples += (current_beta * delta_t) * (0.5 * s_samples + score_s)
                k_samples += (current_beta * delta_t) * (0.5 * k_samples + score_k)
                # Diffusion
                s_samples += torch.sqrt(current_beta * delta_t) * z_s
                k_samples += torch.sqrt(current_beta * delta_t) * z_k 
            
            return s_samples, k_samples
    
    def Heun_step(self, s_samples, k_samples, y, current_time, next_time, sde):
            '''
            One step of Heun's method.
            '''
            score_s, score_k = self.rim.scores(current_time*torch.ones(s_samples.shape[0], device=self.rim.device), s_samples, k_samples, y)
            z_s = torch.randn_like(s_samples, device=s_samples.device)
            z_k = torch.randn_like(k_samples, device=k_samples.device)

            if sde.kind == 'VE':
                current_sigma = sde.sigma(current_time)
                next_sigma = sde.sigma(next_time)
                # Predictor step
                s_samples_pred = s_samples + (current_sigma**2 - next_sigma**2)*score_s + torch.sqrt(current_sigma**2 - next_sigma**2) * z_s
                k_samples_pred = k_samples + (current_sigma**2 - next_sigma**2)*score_k + torch.sqrt(current_sigma**2 - next_sigma**2) * z_k
                # Corrector step
                score_s_pred, score_k_pred = self.rim.scores(next_time*torch.ones(s_samples.shape[0], device=self.rim.device), s_samples_pred, k_samples_pred, y)
                s_samples += 0.5 * (current_sigma**2 - next_sigma**2) * (score_s + score_s_pred) + torch.sqrt(current_sigma**2 - next_sigma**2) * z_s
                k_samples += 0.5 * (current_sigma**2 - next_sigma**2) * (score_k + score_k_pred) + torch.sqrt(current_sigma**2 - next_sigma**2) * z_k
            elif sde.kind == 'VP_linear' or sde.kind == 'VP_exp':
                current_beta = sde.beta(current_time)
                next_beta = sde.beta(next_time)
                # Predictor step
                delta_t = current_time - next_time
                s_samples_pred = s_samples + (current_beta * delta_t) * (0.5 * s_samples + score_s) + torch.sqrt(current_beta * delta_t) * z_s
                k_samples_pred = k_samples + (current_beta * delta_t) * (0.5 * k_samples + score_k) + torch.sqrt(current_beta * delta_t) * z_k
                # Corrector step
                score_s_pred, score_k_pred = self.rim.scores(next_time*torch.ones(s_samples.shape[0], device=self.rim.device), s_samples_pred, k_samples_pred, y)
                s_samples += 0.5 * (current_beta * delta_t) * (0.5 * (s_samples + s_samples_pred) + score_s + score_s_pred) + 0.5 * (torch.sqrt(current_beta * delta_t) + torch.sqrt(next_beta * delta_t)) * z_s
                k_samples += 0.5 * (current_beta * delta_t) * (0.5 * (k_samples + k_samples_pred) + score_k + score_k_pred) + 0.5 * (torch.sqrt(current_beta * delta_t) + torch.sqrt(next_beta * delta_t)) * z_k

            return s_samples, k_samples

    def Euler_step(self, s_samples, k_samples, y, current_time, next_time, sde):
            '''
            One step of Euler method.
            '''
            score_s, score_k = self.rim.scores(current_time*torch.ones(s_samples.shape[0], device=self.rim.device), s_samples, k_samples, y)

            if sde.kind == 'VE':
                current_sigma = sde.sigma(current_time)
                next_sigma = sde.sigma(next_time)
                # Drift
                s_samples += (0.5) * (current_sigma**2 - next_sigma**2)*score_s
                k_samples += (0.5) * (current_sigma**2 - next_sigma**2)*score_k
            elif sde.kind == 'VP_linear' or sde.kind == 'VP_exp':
                current_beta = sde.beta(current_time)
                # Drift
                delta_t = current_time - next_time
                s_samples += (0.5) * (current_beta * delta_t) * (s_samples + score_s)
                k_samples += (0.5) * (current_beta * delta_t) * (k_samples + score_k)
            
            return s_samples, k_samples
    
    def RK4_step(self, s_samples, k_samples, y, current_time, next_time, sde):
            '''
            One step of Runge-Kutta 4th order method.
            '''
            score_s, score_k = self.rim.scores(current_time*torch.ones(s_samples.shape[0], device=self.rim.device), s_samples, k_samples, y)

            if sde.kind == 'VE':
                current_sigma = sde.sigma(current_time)
                next_sigma = sde.sigma(next_time)
                mid_time = 0.5 * (current_time + next_time)

                k1_s = 0.5 * (current_sigma**2 - next_sigma**2) * score_s
                k1_k = 0.5 * (current_sigma**2 - next_sigma**2) * score_k
                score_s_k2, score_k_k2 = self.rim.scores(mid_time*torch.ones(s_samples.shape[0], device=self.rim.device), s_samples + 0.5 * k1_s, k_samples + 0.5 * k1_k, y)
                k2_s = 0.5 * (current_sigma**2 - next_sigma**2) * score_s_k2
                k2_k = 0.5 * (current_sigma**2 - next_sigma**2) * score_k_k2
                score_s_k3, score_k_k3 = self.rim.scores(mid_time*torch.ones(s_samples.shape[0], device=self.rim.device), s_samples + 0.5 * k2_s, k_samples + 0.5 * k2_k, y)
                k3_s = 0.5 * (current_sigma**2 - next_sigma**2) * score_s_k3
                k3_k = 0.5 * (current_sigma**2 - next_sigma**2) * score_k_k3
                score_s_k4, score_k_k4 = self.rim.scores(next_time*torch.ones(s_samples.shape[0], device=self.rim.device), s_samples + k3_s, k_samples + k3_k, y)
                k4_s = 0.5 * (current_sigma**2 - next_sigma**2) * score_s_k4
                k4_k = 0.5 * (current_sigma**2 - next_sigma**2) * score_k_k4

                s_samples += (1/6) * (k1_s + 2*k2_s + 2*k3_s + k4_s)
                k_samples += (1/6) * (k1_k + 2*k2_k + 2*k3_k + k4_k)
            elif sde.kind == 'VP_linear' or sde.kind == 'VP_exp':
                current_beta = sde.beta(current_time)
                next_beta = sde.beta(next_time)
                mid_time = 0.5 * (current_time + next_time)
                mid_beta = sde.beta(mid_time)
                delta_t = current_time - next_time

                k1_s = (0.5) * (current_beta * delta_t) * (s_samples + score_s)
                k1_k = (0.5) * (current_beta * delta_t) * (k_samples + score_k)
                score_s_k2, score_k_k2 = self.rim.scores(mid_time*torch.ones(s_samples.shape[0], device=self.rim.device), s_samples + 0.5 * k1_s, k_samples + 0.5 * k1_k, y)
                k2_s = (0.5) * (mid_beta * delta_t) * ((s_samples + 0.5 * k1_s) + score_s_k2)
                k2_k = (0.5) * (mid_beta * delta_t) * ((k_samples + 0.5 * k1_k) + score_k_k2)
                score_s_k3, score_k_k3 = self.rim.scores(mid_time*torch.ones(s_samples.shape[0], device=self.rim.device), s_samples + 0.5 * k2_s, k_samples + 0.5 * k2_k, y)
                k3_s = (0.5) * (mid_beta * delta_t) * ((s_samples + 0.5 * k2_s) + score_s_k3)
                k3_k = (0.5) * (mid_beta * delta_t) * ((k_samples + 0.5 * k2_k) + score_k_k3)
                score_s_k4, score_k_k4 = self.rim.scores(next_time*torch.ones(s_samples.shape[0], device=self.rim.device), s_samples + k3_s, k_samples + k3_k, y)
                k4_s = (0.5) * (next_beta * delta_t) * ((s_samples + k3_s) + score_s_k4)
                k4_k = (0.5) * (next_beta * delta_t) * ((k_samples + k3_k) + score_k_k4)

                s_samples += (1/6) * (k1_s + 2*k2_s + 2*k3_s + k4_s)
                k_samples += (1/6) * (k1_k + 2*k2_k + 2*k3_k + k4_k)
            
            return s_samples, k_samples
