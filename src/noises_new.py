import math

import numpy as np
import scipy as sp
import scipy.special
import scipy.stats
import torch
from scipy.stats import beta, binom, gamma, norm
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

    def certify(self, prob_lb, p=None):
        raise NotImplementedError()

    def certifyl1(self, prob_lb):
        return self.certify(prob_lb, p=1)

    def certifyl2(self, prob_lb):
        return self.certify(prob_lb, p=2)

    def certifylinf(self, prob_lb):
        return self.certify(prob_lb, p='inf')

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

    def certify(self, prob_lb, p):
        if p == float("inf"):
            return 2 * self.lambd * (1 - (1.5 - prob_lb) ** (1 / self.dim))
        if p > 1:
            raise ValueError(f"Unable to certify UniformNoise for p={p}.")
        return 2 * self.lambd * (prob_lb - 0.5)


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

    def certify(self, prob_lb, p):
        ppen = 1
        if p > 2:
            ppen = self.dim ** (0.5 - 1/p)
        return self.norm_dist.icdf(prob_lb) / ppen



class LaplaceNoise(Noise):
    '''Isotropic Laplace noise
    '''

    def __init__(self, device, dim, sigma=None, lambd=None):
        super().__init__(device, dim, sigma, lambd)
        self.laplace_dist = Laplace(loc=torch.tensor(0.0, device=device),
                                    scale=torch.tensor(self.lambd, device=device))
        self.table = self.table_radii = self.table_rho = self._table_info = None

    def _sigma(self):
        return 2 ** 0.5
        
    def sample(self, x):
        return self.laplace_dist.sample(x.shape) + x

    def certify(self, prob_lb, p, mode='approx'):
        if p == float("inf"):
            return self.certifylinf(self, prob_lb, mode='approx')
        if p > 1:
            raise ValueError(f"Unable to certify LaplaceNoise for p={p}.")
        a = 0.5 * self.lambd * torch.log(prob_lb / (1 - prob_lb))
        b = -self.lambd * (torch.log(2 * (1 - prob_lb)))
        return torch.max(a, b)

    def certifylinf(self, prob_lb, mode='approx',
                    inc=0.001, grid_type='radius', upper=3, save=True):
        '''Certify Laplace smoothing against linf adversary.
        There are two modes of certification: "approx" or "integrate".
        The latter computes a table of robust radii from the differential
        method and performs lookup during certification, and is guaranteed
        to be correct. But this table calculation takes a bit of overhead 
        (though it's only done once, and the table will be saved for loading
        in the future).
        The former uses the following approximation which is highly accurate
        in high dimension:
            
            lambda * GaussianCDF(prob_lb) / d**0.5
        
        We verify the quality of this approximation in the `test_noises.py`.
        By default, "approx" mode is used.
        '''
        if mode == 'approx':
            return self.lambd * Normal(0, 1).icdf(prob_lb) / self.dim ** 0.5
        elif mode == 'integrate':
            table_info = dict(inc=inc, grid_type=grid_type, upper=upper)
            if self.table_rho is None or self._table_info != table_info:
                self.make_linf_table(inc, grid_type, upper, save)
                self._table_info = table_info
            prob_lb = prob_lb.numpy()
            idxs = np.searchsorted(self.table_rho, prob_lb, 'right') - 1
            return torch.tensor(self.lambd * self.table_radii[idxs],
                                dtype=torch.float)
        else:
            raise ValueError(f'Unrecognized mode "{mode}"')
    
    def Phi(self, prob):
        def phi(c, d):
            return binom(d, 0.5).sf((c+d)/2)
        def phiinv(p, d):
            return 2 * binom(d, 0.5).isf(p) - d
        d = self.dim
        c = phiinv(prob, d)
        pp = phi(c, d)
    #     print(c, p, pp)
        return c * (prob - pp) + d * phi(c - 1/2, d-1) - d * phi(c, d)

    def _certifylinf_integrate(self, rho):
        return sp.integrate.quad(lambda p: 1/self.Phi(p),
                                1 - rho, 1/2)[0]

    def make_linf_table(self, inc=0.001, grid_type='radius', upper=3, save=True):
        '''Calculate or load a table of robust radii for linf adversary.
        First try to load a table under `./tables/` with the corresponding
        parameters. If this fails, calculate the table.
        Inputs:
            inc: grid increment (default: 0.05)
            grid_type: 'radius' | 'prob' (default: 'radius')
                In a `radius` grid, the probabilities rho are calculated as
                GaussianCDF([0, inc, 2 * inc, ..., upper - inc, upper]).
                In a `prob` grid, the probabilities rho are spaced out evenly
                in increments of `inc`
            upper: if `grid_type == 'radius'`, then the upper limit to the
                radius grid. (default: 3)
            save: whether to save the table computed
        Outputs:
            None, but `self.table`, `self.table_rho`, `self.table_radii`
            are now defined.
        '''
        from os.path import join
        if grid_type == 'radius':
            rho_fname = join('tables',
                    f'laplace_linf_d{self.dim}_inc{inc}'
                    f'_grid{grid_type}_upper{upper}_rho.npy')
            radii_fname = join('tables', 
                    f'laplace_linf_d{self.dim}_inc{inc}'
                    f'_grid{grid_type}_upper{upper}_radii.npy')
        else:
            rho_fname = join('tables',
                    f'laplace_linf_d{self.dim}_inc{inc}'
                    f'_grid{grid_type}_rho.npy')
            radii_fname = join('tables', 
                    f'laplace_linf_d{self.dim}_inc{inc}'
                    f'_grid{grid_type}_radii.npy')
        try:
            self.table_rho = np.load(rho_fname)
            self.table_radii = np.load(radii_fname)
            self.table = dict(zip(self.table_rho, self.table_radii))
            print('Found and loaded saved table: '
                f'Laplace, Linf adv, dim {self.dim}, inc {inc}, grid {grid_type}'
                + f', upper {upper}' if grid_type == 'radius' else '')
        except FileNotFoundError:
            print('Making robust radii table: Laplace, Linf adv')
            self.table = self._make_linf_table(inc, grid_type, upper)
            self.table_rho = np.array(list(self.table.keys()))
            self.table_radii = np.array(list(self.table.values()))
            if save:
                import os
                print('Saving robust radii table')
                os.makedirs('tables', exist_ok=True)
                np.save(rho_fname, self.table_rho)
                np.save(radii_fname, self.table_radii)
    
    def _make_linf_table(self, inc=0.001, grid_type='radius', upper=3):
        import tqdm
        table = {1/2: 0}
        lastrho = 1/2
        if grid_type == 'radius':
            rgrid = np.arange(inc, upper+inc, inc)
            grid = norm.cdf(rgrid)
        elif grid_type == 'prob':
            grid = np.arange(1/2+inc, 1, inc)
        else:
            raise ValueError(f'Unknow grid_type {grid_type}')
        for rho in tqdm.tqdm(grid):
            delta = sp.integrate.quad(lambda p: 1/self.Phi(p),
                                1 - rho, 1 - lastrho)[0]
            table[rho] = table[lastrho] + delta
            lastrho = rho
        return table


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

    def certify(self, prob_lb, p):
        if p > 1:
            raise ValueError(f"Unable to certify LomaxNoise for p={p}.")
        prob_lb = prob_lb.numpy()
        a = self.a
        radius = sp.special.hyp2f1(
                    1, a / (a + 1), a / (a + 1) + 1,
                    (2 * prob_lb - 1) ** (1 + 1 / a)
                ) * self.lambd * (2 * prob_lb - 1) / a
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
        noise = noise / torch.norm(noise, dim=1, p=2).unsqueeze(1) * radius
        return noise + x

    def certify(self, prob_lb, p):
        ppen = 1
        if p > 2:
            ppen = self.dim ** (0.5 - 1/p)
        radius = self.lambd * (
            2 - 4 * self.beta_dist.ppf(0.75 - 0.5 * prob_lb.numpy()))
        return torch.tensor(radius, dtype=torch.float) / ppen

def sample_linf_sphere(device, shape):
    noise = (2 * torch.rand(shape, device=device) - 1
            ).reshape((shape[0], -1))
    sel_dims = torch.randint(noise.shape[1], size=(noise.shape[0],))
    idxs = torch.arange(0, noise.shape[0], dtype=torch.long)
    noise[idxs, sel_dims] = torch.sign(
        torch.rand(shape[0], device=device) - 0.5)
    return noise

class ExpInfNoise(Noise):
    r'''Noise of the form \|x\|_\infty^{-j} e^{-\|x/\lambda\|_\infty^k}
    '''

    def __init__(self, device, dim, sigma=None, lambd=None, k=1, j=0):
        self.k = k
        self.j = j
        super().__init__(device, dim, sigma, lambd)
        if dim > 1:
            self.gamma_factor = dim / (dim - 1) * math.exp(
                math.lgamma((dim - j) / k) - math.lgamma((dim - j - 1) / k))
        elif j == 0:
            self.gamma_factor = math.exp(
                math.lgamma((dim + k) / k) - math.lgamma((dim + k - 1) / k))
        else:
            raise ValueError(
                f'ExpInfNoise(dim={dim}, k={k}, j={j}) is not a distribution.')
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
        noise = sample_linf_sphere(self.device, x.shape)
        return (noise * radius).view(x.shape) + x

    def certify(self, prob_lb, p):
        '''
        Note that if `prob_lb > 1 - 1/self.dim`, then better radii
        are available (see paper), but when `self.dim` is large, like in CIFAR10
        or ImageNet, this almost never happens.
        '''
        if p == float("inf"):
            if self.k == 1 and self.j == 0:
                return self.lambd * torch.log(0.5 / (1 - prob_lb))
            else:
                raise NotImplementedError()
        if p > 1:
            raise ValueError(f"Unable to certify ExpInfNoise for p={p}.")
        return 2 * self.lambd * \
                self.gamma_factor * (prob_lb - 0.5)

class PowerInfNoise(Noise):

    def __init__(self, device, dim, sigma=None, lambd=None, a=None):
        self.a = a
        if a is None:
            raise ValueError('Parameter `a` is required.')
        super().__init__(device, dim, sigma, lambd)
        self.beta_dist = sp.stats.betaprime(dim, a - self.dim)

    def _sigma(self):
        d = self.dim
        a = self.a
        r2 = (d - 1) / 3 + 1
        return np.sqrt(r2 * (d + 1) / (a - d - 1) / (a - d - 2))

    def sample(self, x):
        samples = self.beta_dist.rvs((len(x), 1))
        radius = torch.tensor(samples, dtype=torch.float, device=self.device)
        noise = sample_linf_sphere(self.device, x.shape)
        return (noise * radius * self.lambd).view(x.shape) + x

    def certify(self, prob_lb, p):
        if p > 1:
            raise ValueError(f"Unable to certify PowerLawNoise for p={p}.")
        return self.lambd * 2 * self.dim / (self.a - self.dim) * (prob_lb - 0.5)


if __name__ == '__main__':
    import time
    dim = 3072
    noise = PowerInfNoise('cpu', dim, lambd=1, a=dim+3)
    before = time.time()
    # noise.make_linf_table(0.05)
    # cert1 = noise.certifylinf(torch.arange(0.5, 1, 0.01))
    # cert2 = noise.certifylinf(torch.arange(0.5, 1, 0.01), 'integrate')
    # print(cert1)
    # print(cert2)
    # print((cert1 - cert2).std())
    after = time.time()
    # print(noise.table)
    print(noise.sample(torch.zeros(10000, dim)).std(), noise.sigma)
    print('{:.3}'.format(after - before))
