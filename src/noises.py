import math
import numpy as np
import scipy as sp
import scipy.special
import torch
from torch.distributions import Normal, Uniform, Laplace, Gamma


class Noise(object):

    def __init__(self, sigma, device):
        self.sigma = sigma
        self.device = device

    def sample(self, shape):
        raise NotImplementedError

    def certify(self, prob_lower_bound):
        raise NotImplementedError


class Clean(Noise):

    def __init__(self, device, **kwargs):
        super().__init__(None, device)

    def sample(self, shape):
        return torch.zeros(shape, device=self.device)

    def certify(self, prob_lower_bound):
        return torch.zeros_like(prob_lower_bound)


class GaussianNoise(Noise):

    def __init__(self, sigma, device, **kwargs):
        super().__init__(sigma, device)
        self.norm_dist = Normal(
            loc=torch.tensor(0., device=device),
            scale=torch.tensor(sigma, device=device))

    def sample(self, shape):
        return self.norm_dist.sample(shape)

    def certify(self, prob_lower_bound):
        return self.norm_dist.icdf(prob_lower_bound)

 
class LaplaceNoise(Noise):

    def __init__(self, sigma, device, **kwargs):
        super().__init__(sigma, device)
        self.lambd = sigma * 2 ** (-0.5)
        self.laplace_dist = Laplace(
            loc=torch.tensor(0., device=device), 
            scale=torch.tensor(self.lambd, device=device))

    def sample(self, shape):
        return self.laplace_dist.sample(shape)
        
    def certify(self, prob_lower_bound):
        a = 0.5 * self.lambd * \
            torch.log(prob_lower_bound / (1 - prob_lower_bound))
        b = -self.lambd * (torch.log(2 * (1 - prob_lower_bound)))
        return torch.max(a, b)


class ExpInfNoise(Noise):

    def __init__(self, sigma, device, dim=3*32*32, k=1.0, **kwargs):
        super().__init__(sigma, device)
        self.dim = dim
        self.k = k
        self.lambd = sigma / (np.exp(sp.special.loggamma((dim + 1) / k) - \
                                     sp.special.loggamma(dim / k))) * 3 ** 0.5
        self.gamma_dist = Gamma(
            concentration=torch.tensor(dim / k, device=device),
            rate=torch.tensor((1 / self.lambd) ** k, device=device))
        self.gamma_factor = np.exp(sp.special.loggamma((dim + k) / k) - \
                                   sp.special.loggamma((dim + k - 1) / k))

    def sample(self, shape):
        radius = (self.gamma_dist.sample((shape[0],))) ** (1 / self.k)
        x = (2 * torch.rand(shape, device=self.device) - 1)
        x = x.reshape((shape[0], -1)) * radius.unsqueeze(1)
        sel_dims = torch.randint(x.shape[1], size=(x.shape[0],))
        idxs = torch.arange(0, x.shape[0], dtype=torch.long)
        noises = torch.round(torch.rand(x.shape[0], device=self.device) - 0.5)
        x[idxs, sel_dims] = noises * radius
        return x.reshape(shape)

    def certify(self, prob_lower_bound, d=3*32*32):
        return 2 * self.lambd * self.gamma_factor * (prob_lower_bound - 0.5)


class UniformNoise(Noise):

    def __init__(self, sigma, device, **kwargs):
        super().__init__(sigma, device)
        self.lambd = sigma * 3 ** 0.5

    def sample(self, shape):
        return (torch.rand(shape, device=self.device) - 0.5) * 2 * self.lambd

    def certify(self, prob_lower_bound):
        return self.lambd * (prob_lower_bound - 0.5)

