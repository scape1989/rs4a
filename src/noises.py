import math
import numpy as np
import torch
from torch.distributions import Normal, Exponential, Uniform, Laplace


class Noise(object):

    def __init__(self, sigma, device):
        self.sigma = sigma
        self.device = device

    def sample(self, shape):
        raise NotImplementedError

    def certify(self, prob_lower_bound):
        raise NotImplementedError


class Clean(Noise):

    def sample(self, shape):
        return torch.zeros(shape, device=self.device)

    def certify(self, prob_lower_bound):
        return torch.zeros_like(prob_lower_bound)


class GaussianNoise(Noise):

    def __init__(self, sigma, device):
        super().__init__(sigma, device)
        self.norm_dist = Normal(
            loc=torch.tensor(0., device=device),
            scale=torch.tensor(sigma, device=device))

    def sample(self, shape):
        return self.norm_dist.sample(shape)

    def certify(self, prob_lower_bound):
        return self.norm_dist.icdf(prob_lower_bound)

 
class LaplaceNoise(Noise):

    def __init__(self, sigma, device):
        super().__init__(sigma, device)
        self.laplace_dist = Laplace(
            loc=torch.tensor(0., device=device), 
            scale=torch.tensor(sigma, device=device))

    def sample(self, shape):
        return self.laplace_dist.sample(shape)
        
    def certify(self, prob_lower_bound):
        a = 0.5 * self.sigma* torch.log(prob_lower_bound / (1-prob_lower_bound))
        b = - self.sigma * (torch.log(2 * (1 - prob_lower_bound)))
        return torch.max(a, b)


class ExpInfNoise(Noise):
    """
    Noise p(x) \propto \exp(-(||x||_\infty/\sigma)^k)
    """
    def __init__(self, sigma, device):
        super().__init__(sigma, device)
        self.expon_dist = Exponential(rate=1/torch.tensor(sigma, device=device))
        self.sigma = (3*32*32 / self.sigma) ** (-1.)# / args.power)

    def sample(self, shape):
        radius = self.expon_dist.sample((shape[0],))
        x = (2 * torch.rand(shape, device=self.device) - 1)
        x = x.reshape((shape[0], -1)) * radius.unsqueeze(1)
        sel_dims = torch.randint(x.shape[1], size=(x.shape[0],))
        idxs = torch.arange(0, x.shape[0], dtype=torch.long)
        noises = torch.round(torch.rand(x.shape[0], device=self.device) - 0.5)
        x[idxs, sel_dims] = noises * radius
        return x.reshape(shape)

    def certify(self, prob_lower_bound, d=3*32*32):
        a = self.sigma * 2 * d * (prob_lower_bound - 0.5)
        b = self.sigma * (d-1+math.log(0.5*d) - torch.log(1-prob_lower_bound))
        mask = prob_lower_bound < 1 - 1 / (2 * d)
        return mask * a + ~mask * b

class UniformNoise(Noise):

    def sample(self, shape):
        return (torch.rand(shape, device=self.device) - 0.5) * 2 * self.sigma

    def certify(self, prob_lower_bound):
        return self.sigma * (prob_lower_bound - 0.5)

