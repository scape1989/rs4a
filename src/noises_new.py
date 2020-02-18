import math

import numpy as np
import scipy as sp
import scipy.special
import scipy.stats
import torch
from torch.distributions import (Beta, Dirichlet, Gamma, Laplace, Normal,
                                 Pareto, Uniform)


def atanh(x):
    return 0.5 * np.log((1 + x) / (1 - x))


class Noise(object):

    def __init__(self, device, dim, sigma=None, lambd=None):
        self.dim = dim
        self.device = device
        if lambd is None and sigma is not None:
            self.sigma = sigma
            self.lambd = self.get_lambd(sigma)
        elif sigma is None and lambd is not None:
            self.lambd = lambd
            self.sigma = self.get_sigma(lambd)
        else:
            raise ValueError('Please give exactly one of sigma or lambd')

    def _sigma(self):
        '''Calculates the sigma if lambd = 1
        '''
        raise NotImplementedError()

    def get_sigma(self, lambd=None):
        '''Calculates the sigma given lambd
        '''
        if lambd is None:
            lambd = self.lambd
        return lambd * self._sigma()

    def get_lambd(self, sigma=None):
        '''Calculates the lambd given sigma
        '''
        if sigma is None:
            sigma = self.sigma
        return sigma / self._sigma()

    def sample(self, x):
        '''Apply noise to x'''
        raise NotImplementedError()

    def certify(self, prob_lower_bound, adv=None):
        raise NotImplementedError()

    def certifyl1(self, prob_lower_bound):
        return self.certify(prob_lower_bound, adv=1)

    def certifyl2(self, prob_lower_bound):
        return self.certify(prob_lower_bound, adv=2)

    def certifylinf(self, prob_lower_bound):
        return self.certify(prob_lower_bound, adv=float('inf'))

class Clean(Noise):

    def __init__(self, device, dim):
        super().__init__(device, None, sigma=1)

    def sample(self, x):
        return x

    def _sigma(self):
        return 1


class UniformNoise(Noise):
    '''Uniform noise on [-lambda, lambda]^dim
    '''

    def __init__(self, device, dim, sigma=None, lambd=None):
        super().__init__(device, dim, sigma, lambd)

    def _sigma(self):
        return 3 ** -0.5

    def sample(self, x):
        return (torch.rand_like(x, device=self.device) - 0.5) * 2 * self.lambd + x

    def certify(self, prob_lower_bound, adv):
        if adv == float("inf"):
            return 2 * self.lambd * (1 - (1.5 - prob_lower_bound) ** (1 / self.dim))
        if adv > 1:
            raise ValueError(f"Unable to certify UniformNoise for adv={adv}.")
        return 2 * self.lambd * (prob_lower_bound - 0.5)


class GaussianNoise(Noise):
    '''Isotropic Gaussian noise
    '''

    def __init__(self, device, dim, sigma=None, lambd=None):
        super().__init__(device, dim, sigma, lambd)
        self.norm_dist = Normal(loc=torch.tensor(0., device=device),
                                scale=torch.tensor(self.lambd, device=device))

    def _sigma(self):
        return 1

    def sample(self, x):
        return torch.randn_like(x) * self.lambd + x

    def certify(self, prob_lower_bound, adv):
        ppen = 1
        if adv > 2:
            ppen = self.dim ** (0.5 - 1/adv)
        return self.norm_dist.icdf(prob_lower_bound) / ppen



class LaplaceNoise(Noise):
    '''Isotropic Laplace noise
    '''

    def __init__(self, device, dim, sigma=None, lambd=None):
        super().__init__(device, dim, sigma, lambd)
        self.laplace_dist = Laplace(loc=torch.tensor(0.0, device=device),
                                    scale=torch.tensor(self.lambd, device=device))

    def _sigma(self):
        return 2 ** 0.5
        
    def sample(self, x):
        return self.laplace_dist.sample(x.shape) + x

    def certify(self, prob_lower_bound, adv):
        if adv == float("inf"):
            # TODO(Greg): fix
            return self.lambd * Normal(0, 1).icdf(prob_lower_bound) / self.dim ** 0.5
        if adv > 1:
            raise ValueError(f"Unable to certify LaplaceNoise for p={p}.")
        a = 0.5 * self.lambd * torch.log(prob_lower_bound / (1 - prob_lower_bound))
        b = -self.lambd * (torch.log(2 * (1 - prob_lower_bound)))
        return torch.max(a, b)



class ParetoNoise(Noise):
    '''Pareto (i.e. power law) noise in each coordinate, iid.
    '''

    def __init__(self, device, dim, sigma=None, lambd=None, a=3):
        self.a = a
        super().__init__(device, dim, sigma, lambd)
        self.pareto_dist = Pareto(
            scale=torch.tensor(self.lambd, device=device, dtype=torch.float),
            alpha=torch.tensor(self.a, device=device, dtype=torch.float))

    def _sigma(self):
        a = self.a
        if a > 2:
            return (0.5 * (a - 1) * (a - 2)) ** -0.5
        else:
            return np.float('inf')

    def sample(self, x):
        samples = self.pareto_dist.sample(x.shape) - self.lambd
        signs = torch.sign(torch.rand_like(x) - 0.5)
        return samples * signs + x

    def certify(self, prob_lower_bound, adv):
        if adv > 1:
            raise ValueError(f"Unable to certify LomaxNoise for adv={adv}.")
        prob_lower_bound = prob_lower_bound.numpy()
        radius = sp.special.hyp2f1(
                    1, self.a / (self.a + 1), self.a / (self.a + 1) + 1,
                    (2 * prob_lower_bound - 1) ** (1 + 1 / self.a)) * \
                self.lambd * (2 * prob_lower_bound - 1) / self.a
        return torch.tensor(radius, dtype=torch.float)


class UniformBallNoise(Noise):

    def __init__(self, device, dim, sigma=None, lambd=None):
        super().__init__(device, dim, sigma, lambd)
        self.beta_dist = sp.stats.beta(0.5 * (self.dim + 1), 0.5 * (self.dim + 1))

    def _sigma(self):
        return (self.dim + 2) ** -0.5

    def sample(self, x):
        radius = torch.rand((len(x), 1), device=self.device) ** (1 / self.dim)
        radius *= self.lambd
        noise = torch.randn(x.shape, device=self.device).reshape(len(x), -1)
        noise = noise / torch.norm(noise, dim=1, p=2, keepdim=True) * radius
        return noise + x

    def certify(self, prob_lower_bound, adv):
        ppen = 1
        if adv > 2:
            ppen = self.dim ** (0.5 - 1/adv)
        radius = self.lambd * (
            2 - 4 * self.beta_dist.ppf(0.75 - 0.5 * prob_lower_bound.numpy()))
        return torch.tensor(radius, dtype=torch.float) / ppen


class ExpInfNoise(Noise):

    def __init__(self, device, dim, sigma=None, lambd=None, k=1, j=0):
        self.k = k
        self.j = j
        super().__init__(device, dim, sigma, lambd)
        self.gamma_factor = math.exp(
            math.lgamma((dim - j) / k) - math.lgamma((dim - j - 1) / k))
        self.gamma_dist = Gamma(
            concentration=torch.tensor((dim - j) / k, device=device),
            rate=torch.tensor((1 / self.lambd) ** k, device=device))

    def _sigma(self):
        k = self.k
        j = self.j
        dim = self.dim
        r2 = (dim - 1) / 3 + 1
        return np.sqrt(r2 / dim * (
            math.exp(math.lgamma((dim + 2 - j) / k)
            - math.lgamma((dim - j) / k))))

    def sample(self, x):
        radius = (self.gamma_dist.sample((len(x), 1))) ** (1 / self.k)
        noise = (2 * torch.rand(x.shape, device=self.device) - 1
                ).reshape((len(x), -1))
        sel_dims = torch.randint(noise.shape[1], size=(noise.shape[0],))
        idxs = torch.arange(0, noise.shape[0], dtype=torch.long)
        noise[idxs, sel_dims] = torch.sign(
            torch.rand(len(x), device=self.device) - 0.5)
        return (noise * radius).view(x.shape) + x

    def certify(self, prob_lower_bound, adv):
        if adv == float("inf"):
            if self.k == 1 and self.j == 0:
                return self.lambd * torch.log(0.5 / (1 - prob_lower_bound))
            else:
                raise NotImplementedError()
        if adv > 1:
                raise ValueError(f"Unable to certify ExpInfNoise for adv={adv}.")
        return 2 * self.lambd * self.dim / (self.dim - 1) * \
                self.gamma_factor * (prob_lower_bound - 0.5)


class Exp1Noise(Noise):

    def __init__(self, device, dim, sigma=None, lambd=None, k=1):
        self.k = k
        super().__init__(device, dim, sigma, lambd)
        self.gamma_factor = math.exp(math.lgamma(dim / k) - math.lgamma((dim + k - 1) / k))
        self.dirichlet_dist = Dirichlet(concentration=torch.ones(dim, device=device))
        self.gamma_dist = Gamma(concentration=torch.tensor(dim / k, device=device),
                                rate=torch.tensor((1 / self.lambd) ** k, device=device))

    def _sigma(self):
        k = self.k
        dim = self.dim 
        return np.sqrt(2 / (dim + 1) / dim) * (
            math.exp(math.lgamma((dim + 1) / k) - math.lgamma(dim / k)))

    def sample(self, x):
        radius = (self.gamma_dist.sample((len(x), 1))) ** (1 / self.k)
        noise = self.dirichlet_dist.sample((len(x),))
        signs = torch.sign(torch.rand_like(noise) - 0.5)
        return noise * signs * radius + x

    def certify(self, prob_lower_bound, adv, num_pts=1000, eps=1e-4):
        if adv > 1:
            raise ValueError(f"Unable to certify Exp1Noise for adv={adv}.")
        x = np.linspace(eps, 0.5, num_pts)
        y = sp.stats.gamma.ppf(1 - 2 * x, self.dim / self.k)
        y = 1 / (1 - sp.stats.gamma.cdf(y, (self.dim + self.k - 1) / self.k))
        y = np.repeat(y[np.newaxis,:], len(prob_lower_bound), axis=0)
        y[x < 1 - prob_lower_bound.numpy()[:, np.newaxis]] = 0
        integral = torch.tensor(np.trapz(y, dx=0.5 / num_pts), dtype=torch.float)
        return 2 * self.lambd * self.gamma_factor / self.k * integral


if __name__ == '__main__':
    dim = 4
    noise = ExpInfNoise('cpu', dim, lambd=1, k=2, j=2)
    print(noise.certify(0.8, p=1))

