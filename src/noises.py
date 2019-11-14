import math
import numpy as np
import scipy as sp
import scipy.stats
import torch
from torch.distributions import Normal, Uniform, Laplace, Gamma, Dirichlet


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

class UniformNoise(Noise):

    def __init__(self, sigma, device, **kwargs):
        super().__init__(sigma, device)
       # self.lambd = sigma * 3 ** 0.5
        self.lambd = 2 * sigma

    def sample(self, shape):
        return (torch.rand(shape, device=self.device) - 0.5) * 2 * self.lambd

    def certify(self, prob_lower_bound):
        return self.lambd * (prob_lower_bound - 0.5)

class GaussianNoise(Noise):

    def __init__(self, sigma, device, **kwargs):
        super().__init__(sigma, device)
        self.lambd = sigma * math.sqrt(math.pi / 2)
        self.norm_dist = Normal(loc=torch.tensor(0., device=device),
                                 scale=torch.tensor(self.lambd, device=device))
                                 #scale=torch.tensor(sigma, device=device))

    def sample(self, shape):
        return self.norm_dist.sample(shape)

    def certify(self, prob_lower_bound):
        return self.norm_dist.icdf(prob_lower_bound) 

 
class LaplaceNoise(Noise):

    def __init__(self, sigma, device, **kwargs):
        super().__init__(sigma, device)
#        self.lambd = sigma * 2 ** (-0.5)
        self.lambd = sigma
        self.laplace_dist = Laplace(loc=torch.tensor(0.0, device=device), 
                                    scale=torch.tensor(self.lambd, device=device))

    def sample(self, shape):
        return self.laplace_dist.sample(shape)
        
    def certify(self, prob_lower_bound):
        a = 0.5 * self.lambd *  torch.log(prob_lower_bound / (1 - prob_lower_bound))
        b = -self.lambd * (torch.log(2 * (1 - prob_lower_bound)))
        return torch.max(a, b)


class ExpInfNoise(Noise):

    def __init__(self, sigma, device, dim=3*32*32, k=1.0, **kwargs):
        super().__init__(sigma, device)
        self.dim = dim
        self.k = k
#        self.lambd = 3 ** 0.5 * sigma / (np.exp(math.lgamma((dim + 1) / k) - math.lgamma(dim / k)))
        self.lambd = 2 * sigma / (math.exp(math.lgamma((dim + 1) / k) - math.lgamma(dim / k)))
        self.gamma_factor = math.exp(math.lgamma((dim + k) / k) - math.lgamma((dim + k - 1) / k))
        self.gamma_dist = Gamma(concentration=torch.tensor(dim / k, device=device),
                                rate=torch.tensor((1 / self.lambd) ** k, device=device))

    def sample(self, shape):
        radius = (self.gamma_dist.sample((shape[0], 1))) ** (1 / self.k)
        noises = (2 * torch.rand(shape, device=self.device) - 1).reshape((shape[0], -1))
        sel_dims = torch.randint(noises.shape[1], size=(noises.shape[0],))
        idxs = torch.arange(0, noises.shape[0], dtype=torch.long)
        noises[idxs, sel_dims] = torch.sign(torch.rand(noises.shape[0], device=self.device) - 0.5) 
        return (noises * radius).reshape(shape)

    def certify(self, prob_lower_bound, d=3*32*32):
        return 2 * self.lambd * self.gamma_factor * (prob_lower_bound - 0.5)


class Exp1Noise(Noise):

    def __init__(self, sigma, device, dim=3*32*32, k=1, **kwargs):
        super().__init__(sigma, device)
        self.dim = dim
        self.k = k
        self.lambd = sigma / (math.exp(math.lgamma((dim + 1) / k) - math.lgamma(dim / k)))
        self.gamma_factor = math.exp(math.lgamma((dim + k) / k) - math.lgamma((dim + k - 1) / k))
        self.dirichlet_dist = Dirichlet(concentration=torch.ones(dim, device=device))
        self.gamma_dist = Gamma(concentration=torch.tensor(dim / k, device=device),
                                rate=torch.tensor((1 / self.lambd) ** k, device=device))

    def sample(self, shape):
        n_samples = int(np.prod(list(shape)) / self.dim)
        radius = (self.gamma_dist.sample((n_samples, 1))) ** (1 / self.k)
        noises = self.dirichlet_dist.sample((n_samples,)) * self.dim
        signs = torch.sign(torch.rand_like(noises) - 0.5)
        return (noises * signs * radius).reshape(shape)

    def certify(self, prob_lower_bound, num_pts=1000, eps=1e-2):
        x = np.linspace(eps, 0.5, num_pts)
        y = sp.stats.gamma.ppf(1 - x, self.dim / self.k)
        y = 1 / (1 - sp.stats.gamma.cdf(y, (self.dim + self.k - 1) / self.k))
        y = np.repeat(y[np.newaxis,:], len(prob_lower_bound), axis=0)
        y[x < 1 - prob_lower_bound.numpy()[:, np.newaxis]] = 0
        integral = torch.tensor(np.trapz(y, dx=1 / num_pts), dtype=torch.float)
        return 2 * self.lambd * self.gamma_factor / self.k * integral

