import torch

class RIM():
    def __init__(self, net, lensingmodel, sde, space_kappa, grad_lik, loss, num_iterations, use_log_t, device):
        self.net = net
        self.lensingmodel = lensingmodel
        self.sde = sde
        self.space_kappa = space_kappa
        self.grad_lik = grad_lik
        self.loss = loss
        self.num_iterations = num_iterations
        self.use_log_t = use_log_t
        self.device = device

        if self.space_kappa == 'linear':
            self.rim_to_caustics = torch.nn.Identity()
            self.caustics_to_rim = torch.nn.Identity()
        elif self.space_kappa == 'log':
            self.rim_to_caustics = torch.exp
            self.caustics_to_rim = torch.log
        else:
            raise ValueError('Invalid space_kappa value. Must be "linear" or "log".')

        if self.loss.iteration_weights == 'uniform':
            self.iteration_weights = torch.ones(self.num_iterations, device=self.device)/self.num_iterations
        else:
            self.iteration_weights = torch.tensor(self.loss.iteration_weights, device=self.device)

    def sample_t(self, batch_size):
        '''
        Sample t uniformly from [epsilon, 1].
        '''
        epsilon = self.sde.epsilon
        t = torch.rand(batch_size, device=self.device) * (1.0 - epsilon) + epsilon

        return t
    
    def generate_batch(self, s0, k0, t=None):
        '''
        Generate a training batch (t, st, kt, y) from (s0, k0) in which t is 
        sampled uniformly from [epsilon, 1], or from (s0, k0, t) in which t is 
        specified.
        '''
        # sample t if not provided
        if t is None:
            batch_size = s0.shape[0]
            t = self.sample_t(batch_size)

        # add diffusion noise to source and kappa map
        st = self.sde.forward_process(s0, t)
        kt = self.sde.forward_process(k0, t)

        # simulate lensed observation
        y = self.lensingmodel.simulate_lensing(s=s0, k=self.rim_to_caustics(k0), noise=True)

        return t, st, kt, y

    def likelihood_gradients(self, y, s0, k0):
        '''
        Compute gradients of negative log-likelihood wrt s0 and k0 and 
        residuals between observation and lensed image prediction.
        '''
        # copy s0_hat, k0_hat, y so that likelihood gradients are computed on separate graph
        s0 = s0.clone()
        k0 = k0.clone()

        # require gradients for s0_hat and k0_hat
        s0.requires_grad_(True)
        k0.requires_grad_(True)

        # Explicitly enable grad in case caller is under torch.no_grad().
        with torch.enable_grad():
            # mapping kappa from RIM to caustics units
            k0 = self.rim_to_caustics(k0)

            # lensed image prediction
            y_hat = self.lensingmodel.simulate_lensing(s=s0, k=k0, noise=False)

            # negative log-likelihood
            yres, nll = self.lensingmodel.neg_log_likelihood(y, y_hat)

            # compute residuals and gradients
            grad_s0, grad_k0 = torch.autograd.grad(nll, 
                                               (s0, k0),
                                               create_graph=False)
        
        # detach residuals from compute graph
        yres = yres.detach()
        yres.requires_grad_(False)

        return yres, grad_s0, grad_k0
    
    def tanh_grad_update(self, grad_s, grad_k):
        '''
        Update likelihood gradients with tanh function.
        '''
        grad_s = torch.tanh(grad_s/self.grad_lik.grad_norm_source)
        grad_k = torch.tanh(grad_k/self.grad_lik.grad_norm_kappa)

        return grad_s, grad_k
    
    def arcsinh_grad_update(self, grad_s, grad_k):
        '''
        Update likelihood gradients with arcsinh function.
        '''
        grad_s = torch.asinh(grad_s/self.grad_lik.grad_norm_source)
        grad_k = torch.asinh(grad_k/self.grad_lik.grad_norm_kappa)

        return grad_s, grad_k
    
    def Adam_grad_update(self, grad_s, grad_k, time_step):
        '''
        Update likelihood gradients with Adam optimizer.
        '''
        # calculate first and second moments
        m_s = self.grad_lik.beta1 * self.m_s + (1 - self.grad_lik.beta1) * grad_s
        m_k = self.grad_lik.beta1 * self.m_k + (1 - self.grad_lik.beta1) * grad_k
        v_s = self.grad_lik.beta2 * self.v_s + (1 - self.grad_lik.beta2) * (grad_s**2)
        v_k = self.grad_lik.beta2 * self.v_k + (1 - self.grad_lik.beta2) * (grad_k**2)
        
        # if time step = 0, perform Adam update without storing moments
        if time_step == 0:
            # bias correction
            m_hat_s = m_s / (1 - self.grad_lik.beta1)
            m_hat_k = m_k / (1 - self.grad_lik.beta1)
            v_hat_s = v_s / (1 - self.grad_lik.beta2)
            v_hat_k = v_k / (1 - self.grad_lik.beta2)

        if time_step > 0:
            # store moments
            self.m_s = m_s
            self.m_k = m_k
            self.v_s = v_s
            self.v_k = v_k
        
            # bias correction
            m_hat_s = m_s / (1 - self.grad_lik.beta1**(time_step))
            m_hat_k = m_k / (1 - self.grad_lik.beta1**(time_step))
            v_hat_s = v_s / (1 - self.grad_lik.beta2**(time_step))
            v_hat_k = v_k / (1 - self.grad_lik.beta2**(time_step))

        # Adam update
        grad_s = m_hat_s / (torch.sqrt(v_hat_s + self.grad_lik.adam_epsilon) + self.grad_lik.adam_epsilon)
        grad_k = m_hat_k / (torch.sqrt(v_hat_k + self.grad_lik.adam_epsilon) + self.grad_lik.adam_epsilon)

        return grad_s, grad_k
    
    def initialize_states(self, t, st, kt, y):
        '''
        Initialize RIM states (s0_hat, k0_hat, likelihood gradients, residuals 
        and hidden states) at time t given st, kt and y.
        '''
        # initial estimates of s0 and k0 are st and kt
        s0_hat = st.clone()
        k0_hat = kt.clone()

        # initial lensed image residuals and likelihood gradients
        yres, grad_s, grad_k = self.likelihood_gradients(y, s0_hat, k0_hat)

        # initial memory hidden states
        h = self.net.init_hidden(batch_size=st.shape[0])

        # initial Adam moments
        if self.grad_lik.type == 'Adam':
            self.m_s = torch.zeros_like(grad_s, device=self.device)
            self.m_k = torch.zeros_like(grad_k, device=self.device)
            self.v_s = torch.zeros_like(grad_s, device=self.device)
            self.v_k = torch.zeros_like(grad_k, device=self.device)

        return t, s0_hat, k0_hat, st, kt, y, grad_s, grad_k, yres, h

    def time_step(self, t, s0_hat, k0_hat, st, kt, y, grad_s, grad_k, yres, h):
        '''
        RIM time step.
        '''
        if self.use_log_t:
            delta_s, delta_k, h = self.net(torch.log(t), s0_hat, k0_hat, st, kt, y, grad_s, grad_k, yres, h)
        else:
            delta_s, delta_k, h = self.net(t, s0_hat, k0_hat, st, kt, y, grad_s, grad_k, yres, h)
        s0_hat = s0_hat + delta_s
        k0_hat = k0_hat + delta_k

        return s0_hat, k0_hat, h
    
    def forward(self, t, st, kt, y):
        '''
        RIM full forward pass.
        '''
        # lists to hold source and kappa map estimates at each iteration
        s0_hat_series = []
        k0_hat_series = []

        # intialize RIM states
        t, s0_hat, k0_hat, st, kt, y, grad_s, grad_k, yres, h = self.initialize_states(t, st, kt, y)

        # update likelihood gradients
        if self.grad_lik.type == 'tanh':
            grad_s, grad_k = self.tanh_grad_update(grad_s, grad_k)
        elif self.grad_lik.type == 'arcsinh':
            grad_s, grad_k = self.arcsinh_grad_update(grad_s, grad_k)
        elif self.grad_lik.type == 'Adam':
            grad_s, grad_k = self.Adam_grad_update(grad_s, grad_k, time_step=0)

        # store estimates, gradients and residuals
        s0_hat_series.append(s0_hat)
        k0_hat_series.append(k0_hat)

        for i in range(self.num_iterations):
            # RIM time step
            s0_hat, k0_hat, h = self.time_step(t, s0_hat, k0_hat, st, kt, y, grad_s, grad_k, yres, h)

            # compute lensed image residuals and likelihood gradients
            yres, grad_s, grad_k = self.likelihood_gradients(y, s0_hat, k0_hat)

            # update likelihood gradients
            if self.grad_lik.type == 'tanh':
                grad_s, grad_k = self.tanh_grad_update(grad_s, grad_k)
            elif self.grad_lik.type == 'arcsinh':
                grad_s, grad_k = self.arcsinh_grad_update(grad_s, grad_k)
            elif self.grad_lik.type == 'Adam':
                grad_s, grad_k = self.Adam_grad_update(grad_s, grad_k, time_step=i+1)

            # store estimates, gradients and residuals
            s0_hat_series.append(s0_hat)
            k0_hat_series.append(k0_hat)

        return s0_hat_series, k0_hat_series

    def forward_eval(self, t, st, kt, y):
        '''
        RIM full forward pass with no gradient tracking.
        '''
        with torch.no_grad():
            # lists to hold source and kappa map estimates, likelihood gradients and residuals at each iteration
            s0_hat_series = []
            k0_hat_series = []
            grad_s_series = []
            grad_k_series = []
            yres_series = []

            # intialize RIM states
            t, s0_hat, k0_hat, st, kt, y, grad_s, grad_k, yres, h = self.initialize_states(t, st, kt, y)

            # update likelihood gradients
            if self.grad_lik.type == 'tanh':
                grad_s, grad_k = self.tanh_grad_update(grad_s, grad_k)
            elif self.grad_lik.type == 'arcsinh':
                grad_s, grad_k = self.arcsinh_grad_update(grad_s, grad_k)
            elif self.grad_lik.type == 'Adam':
                grad_s, grad_k = self.Adam_grad_update(grad_s, grad_k, time_step=0)

            # store estimates, gradients and residuals
            s0_hat_series.append(s0_hat)
            k0_hat_series.append(k0_hat)
            grad_s_series.append(grad_s)
            grad_k_series.append(grad_k)
            yres_series.append(yres)

            for i in range(self.num_iterations):
                # RIM time step
                s0_hat, k0_hat, h = self.time_step(t, s0_hat, k0_hat, st, kt, y, grad_s, grad_k, yres, h)

                # compute lensed image residuals and likelihood gradients
                yres, grad_s, grad_k = self.likelihood_gradients(y, s0_hat, k0_hat)

                # update likelihood gradients
                if self.grad_lik.type == 'tanh':
                    grad_s, grad_k = self.tanh_grad_update(grad_s, grad_k)
                elif self.grad_lik.type == 'arcsinh':
                    grad_s, grad_k = self.arcsinh_grad_update(grad_s, grad_k)
                elif self.grad_lik.type == 'Adam':
                    grad_s, grad_k = self.Adam_grad_update(grad_s, grad_k, time_step=i+1)

                # store estimates, gradients and residuals
                s0_hat_series.append(s0_hat)
                k0_hat_series.append(k0_hat)
                grad_s_series.append(grad_s)
                grad_k_series.append(grad_k)
                yres_series.append(yres)

            return s0_hat_series, k0_hat_series, grad_s_series, grad_k_series, yres_series

    def forward_eval_final(self, t, st, kt, y):
        '''
        RIM forward pass with no gradient tracking that returns only final estimates.
        '''
        with torch.no_grad():
            # intialize RIM states
            t, s0_hat, k0_hat, st, kt, y, grad_s, grad_k, yres, h = self.initialize_states(t, st, kt, y)

            # update likelihood gradients
            if self.grad_lik.type == 'tanh':
                grad_s, grad_k = self.tanh_grad_update(grad_s, grad_k)
            elif self.grad_lik.type == 'arcsinh':
                grad_s, grad_k = self.arcsinh_grad_update(grad_s, grad_k)
            elif self.grad_lik.type == 'Adam':
                grad_s, grad_k = self.Adam_grad_update(grad_s, grad_k, time_step=0)

            for i in range(self.num_iterations):
                # RIM time step
                s0_hat, k0_hat, h = self.time_step(t, s0_hat, k0_hat, st, kt, y, grad_s, grad_k, yres, h)

                # compute lensed image residuals and likelihood gradients
                yres, grad_s, grad_k = self.likelihood_gradients(y, s0_hat, k0_hat)

                # update likelihood gradients
                if self.grad_lik.type == 'tanh':
                    grad_s, grad_k = self.tanh_grad_update(grad_s, grad_k)
                elif self.grad_lik.type == 'arcsinh':
                    grad_s, grad_k = self.arcsinh_grad_update(grad_s, grad_k)
                elif self.grad_lik.type == 'Adam':
                    grad_s, grad_k = self.Adam_grad_update(grad_s, grad_k, time_step=i+1)

            return s0_hat, k0_hat
        
    def denoise(self, t, st, kt, y):
        '''
        Estimating s0 and k0 from st, kt, y at time t using RIM.
        '''
        s0_hat, k0_hat = self.forward_eval_final(t, st, kt, y)
        return s0_hat, k0_hat
    
    def scores(self, t, st, kt, y):
        '''
        Compute scores for source and kappa map at time t.
        '''
        s0_hat, k0_hat = self.denoise(t, st, kt, y)
        score_s = self.sde.get_score(st, s0_hat, t)
        score_k = self.sde.get_score(kt, k0_hat, t)
        return score_s, score_k
    
    def loss_weights_sde_time(self, t):
        '''
        Compute loss weights based on SDE time t.
        '''
        if self.sde.kind == 'VE':
            sigma_t_cutoff = self.loss.time_weights_cutoff
            sigma_t = self.sde.sigma(t)
            loss_weights = sigma_t_cutoff**2 / sigma_t.clamp_max(sigma_t_cutoff)**2
        elif self.sde.kind == 'VP_linear' or self.sde.kind == 'VP_exp':
            sigma_t_cutoff = self.loss.time_weights_cutoff
            sigma_t = self.sde.sigma(t)
            alpha_t = self.sde.alpha(t)
            sigma_t_over_alpha = sigma_t / alpha_t
            loss_weights = sigma_t_cutoff**2 / sigma_t_over_alpha.clamp_max(sigma_t_cutoff)**2

        return loss_weights

    def loss_fn(self, t, s0, s0_hat_series, k0, k0_hat_series):
        '''
        Compute MSE loss between true and estimated source and kappa maps.
        '''
        loss_s = 0.0
        loss_k = 0.0

        # removing first element of series which corresponds to initial estimates st and kt
        s0_hat_series = s0_hat_series[1:]
        k0_hat_series = k0_hat_series[1:]

        if self.loss.kappa_weights == 'sqrt_kappa':
            sqrt_kappa = torch.sqrt(self.rim_to_caustics(k0))
            weights = sqrt_kappa * (k0.shape[-1]**2) / torch.sum(sqrt_kappa, dim=(1,2,3), keepdim=True)

        # compute weighted loss across RIM iterations
        n_steps = len(s0_hat_series)
        for i in range(n_steps):
            loss_s += torch.mean((s0 - s0_hat_series[i])**2, dim=(1,2,3)) * self.iteration_weights[i]
            if self.loss.kappa_weights == 'uniform':
                loss_k += torch.mean((k0 - k0_hat_series[i])**2, dim=(1,2,3)) * self.iteration_weights[i]
            elif self.loss.kappa_weights == 'sqrt_kappa':
                loss_k += torch.mean((k0 - k0_hat_series[i])**2 * weights, dim=(1,2,3)) * self.iteration_weights[i]

        loss = torch.mean((loss_s + loss_k) * self.loss_weights_sde_time(t))

        return loss
    

    
