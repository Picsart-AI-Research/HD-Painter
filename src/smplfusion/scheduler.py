import torch

def linear(n_timestep = 1000, start = 1e-4, end = 2e-2):
    return Schedule(torch.linspace(start ** 0.5, end ** 0.5, n_timestep, dtype = torch.float64) ** 2)

class Schedule:
    def __init__(self, betas):
        self.betas = betas
        self._alphas = 1 - betas 
        self.alphas = torch.cumprod(self._alphas, 0)
        self.one_minus_alphas = 1 - self.alphas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_one_minus_alphas = torch.sqrt(1 - self.alphas)
        self.sqrt_noise_signal_ratio = self.sqrt_one_minus_alphas / self.sqrt_alphas
        self.noise_signal_ratio = (1 - self.alphas) / self.alphas
    
    def range(self, dt):
        return range(len(self.betas)-1, 0, -dt)

    def sigma(self, t, dt):
        return torch.sqrt((1 - self._alphas[t - dt]) / (1 - self._alphas[t]) * (1 - self._alphas[t] / self._alphas[t - dt]))
