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
        
        We verify the quality of this approximation in `test_noises.py`.
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
            inc: grid increment (default: 0.001)
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
            raise ValueError(f'Unknown grid_type {grid_type}')
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
    '''Uniform distribution over the l2 ball'''

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
            rate=1)

    def _sigma(self):
        k = self.k
        j = self.j
        d = self.dim
        r2 = (d - 1) / 3 + 1
        return np.sqrt(r2 / d * (
            math.exp(math.lgamma((d + 2 - j) / k)
            - math.lgamma((d - j) / k))))

    def sample(self, x):
        radius = (self.gamma_dist.sample((len(x), 1))) ** (1 / self.k)
        noise = sample_linf_sphere(self.device, x.shape)
        return self.lambd * (noise * radius).view(x.shape) + x

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
    r'''Linf-based power law, with density of the form (1 + \|x\|_\infty)^{-a}'''

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

def sample_l2_sphere(device, shape):
    '''Sample uniformly from the unit l2 sphere.
    Inputs:
        device: 'cpu' | 'cuda' | other torch devices
        shape: a pair (batchsize, dim)
    Outputs:
        matrix of shape `shape` such that each row is a sample.
    '''
    noises = torch.randn(shape)
    noises /= noises.norm(dim=1, keepdim=True)
    return noises


def sample_l1_sphere(device, shape):
    '''Sample uniformly from the unit l1 sphere, i.e. the cross polytope.
    Inputs:
        device: 'cpu' | 'cuda' | other torch devices
        shape: a pair (batchsize, dim)
    Outputs:
        matrix of shape `shape` such that each row is a sample.
    '''
    batchsize, dim = shape
    dirdist = Dirichlet(concentration=torch.ones(dim, device=device))
    noises = dirdist.sample([batchsize])
    signs = torch.sign(torch.rand_like(noises) - 0.5)
    return noises * signs

### Level Set Method

def relu(x):
    if isinstance(x, np.ndarray):
        return np.maximum(x, 0, x)
    else:
        return max(x, 0)


def wfun(r, s, e, d):
    '''W function in the paper.
    Calculates the probability a point sampled from the surface of a ball
    of radius `r` centered at the origin is outside a ball of radius `s`
    with center `e` away from the origin.
    '''
    # import pdb
    # pdb.set_trace()
    # print(r, s, e, d)
    t = ((r+e)**2 - s**2)/(4*e*r)
    # print('\t', t)
    return beta((d-1)/2, (d-1)/2).cdf(t)

def get_radii_from_table(table_rho, table_radii, prob_lb):
    prob_lb = prob_lb.numpy()
    idxs = np.searchsorted(table_rho, prob_lb, 'right') - 1
    return torch.tensor(table_radii[idxs], dtype=torch.float)


def get_radii_from_convex_table(table_rho, table_radii, prob_lb):
    '''
    Assuming 1) radii is a convex function of rho and
    2) table_rho[0] = 1/2, table_radii[0] = 0.
    Uses the basic fact that if f is convex and a < b, then

        f'(b) >= (f(b) - f(a)) / (b - a).
    '''
    prob_lb = prob_lb.numpy()
    idxs = np.searchsorted(table_rho, prob_lb, 'right') - 1
    slope = (table_radii[idxs] - table_radii[idxs-1]) / (
        table_rho[idxs] - table_rho[idxs-1]
    )
    rad = table_radii[idxs] + slope * (prob_lb - table_rho[idxs])
    rad[idxs == 0] = 0
    return torch.tensor(rad, dtype=torch.float)

def plexp(z, mode='lowerbound'):
    '''Computes LambertW(e^z) numerically safely.
    For small value of z, we use `scipy.special.lambertw`.
    For large value of z, we apply the approximation

        z - log(z) < W(e^z) < z - log(z) - log(1 - log(z)/z).
    '''
    # ez = np.exp(z)
    if np.isscalar(z):
        if z > 500:
            if mode == 'lowerbound':
                return z - np.log(z)
            elif mode == 'upperbound':
                return z - np.log(z) - np.log(1 - np.log(z) / z)
            else:
                raise ValueError(f'Unknown mode: {mode}')
        else:
            return sp.special.lambertw(np.exp(z))
    else:
        if mode == 'lowerbound':
            # print(z)
            u = z - np.log(z)
        elif mode == 'upperbound':
            u = z - np.log(z) - np.log(1 - np.log(z) / z)
        else:
            raise ValueError(f'Unknown mode: {mode}')
        w = sp.special.lambertw(np.exp(z))
        w[z > 500] = u[z > 500]
        return w

class Exp2Noise(Noise):
    r'''L2-based distribution of the form \|x\|_2^{-j} e^{\|x/\lambda\|_2^k}'''

    def __init__(self, device, dim, sigma=None, lambd=None, k=1, j=0):
        self.k = k
        self.j = j
        super().__init__(device, dim, sigma, lambd)
        self.gamma_dist = Gamma(
            concentration=torch.tensor((dim - j) / k, device=device),
            rate=1)
        self.table = self.table_radii = self.table_rho = self._table_info = None

    def _sigma(self):
        k = self.k
        j = self.j
        d = self.dim
        return np.sqrt(1 / d * 
                    math.exp(math.lgamma((d + 2 - j) / k)
                            - math.lgamma((d - j) / k)
                        )
                    )

    def sample(self, x):
        radius = (self.gamma_dist.sample((len(x), 1))) ** (1 / self.k)
        noise = sample_l2_sphere(self.device, x.shape)
        return self.lambd * (noise * radius).view(x.shape) + x

    def certifyl2(self, prob_lb, mode='levelset',
                inc=0.01, upper=3, save=True):
        if self.k == 1 and self.j == 0:
            if not hasattr(self, 'beta_dist'):                
                self.beta_dist = sp.stats.beta(0.5 * (self.dim - 1),
                                               0.5 * (self.dim - 1))
            radius = self.lambd * (self.dim - 1) * \
                atanh(1 - 2 * self.beta_dist.ppf(1 - prob_lb.numpy()))
            return torch.tensor(radius, dtype=torch.float)
        elif self.k == 2 and self.j == 0:
            raise NotImplementedError()
        elif mode == 'levelset':
            return self.certifyl2_levelset(prob_lb, inc, upper, save)

    def certifyl2_levelset(self, prob_lb, inc=0.01, upper=3, save=True):
        table_info = dict(inc=inc, upper=upper)
        if self.table_rho is None or self._table_info != table_info:
            self.make_l2_table(inc, upper, save)
            self._table_info = table_info
        return self.lambd * get_radii_from_convex_table(
                        self.table_rho, self.table_radii, prob_lb)


    def _pbig(self, t, e, mode='integrate', nsamples=1000):
        '''Compute the big measure of a Neyman-Pearson set with ratio e^t.
        This function assumes `self.lambd == 1`.
        Inputs:
            t: log(kappa)
            e: the l2 norm of perturbation
            mode: integrate | mc
            nsamples: number of samples when `mode == 'mc'`
        Outputs:
        '''
        d = self.dim
        k = self.k
        j = self.j
        # pl = sp.special.lambertw
        if self.j == 0:
            if mode == 'integrate':
                return gamma(d/k).expect(
                    lambda rpow:
                        wfun(rpow**(1/k), relu(rpow - t)**(1/k), e, d),
                    points=[d/k-1], lb=0, ub=10*(d/k))
            elif mode == 'mc':
                rpow = gamma(d/k).rvs(size=nsamples)
                return np.mean(wfun(
                    rpow**(1/k), relu(rpow - t)**(1/k), e, d)
                    )
            else:
                raise ValueError(f'Unrecognized mode: {mode}')
        else:
            def s(rpow):
                q = k/j * (rpow - t) + np.log(rpow) + np.log(k/j)
                p = relu(plexp(q, mode='lowerbound').real)
                s = (j/k * p)**(1/k)
                return s.real
            if mode == 'integrate':
                return gamma(d/k - j/k).expect(
                    lambda rpow: wfun(rpow**(1/k), s(rpow), e, d),
                    points=[d/k-j/k-1], lb=0, ub=10*(d/k-j/k))
            elif mode == 'mc':
                rpow = gamma(d/k - j/k).rvs(size=nsamples)
                return np.mean(wfun(rpow**(1/k), s(rpow), e, d))
            else:
                raise ValueError(f'Unrecognized mode: {mode}')

        
    def _psmall(self, t, e, mode='integrate', nsamples=1000):
        '''Compute the small measure of a Neyman-Pearson set with ratio e^t.
        This function assumes `self.lambd == 1`.
        Inputs:
            t: log(kappa)
            e: the l2 norm of perturbation
            mode: integrate | mc
            nsamples: number of samples when `mode == 'mc'`
        Outputs:
        '''
        d = self.dim
        k = self.k
        j = self.j
        # pl = sp.special.lambertw
        if self.j == 0:
            if mode == 'integrate':
                return gamma(d/k).expect(
                    lambda rpow:
                        1 - wfun(rpow**(1/k), relu(rpow + t)**(1/k), e, d),
                    points=[d/k-1], lb=0, ub=10*(d/k))
            elif mode == 'mc':
                rpow = gamma(d/k).rvs(size=nsamples)
                return np.mean(1 - wfun(
                    rpow**(1/k), relu(rpow + t)**(1/k), e, d)
                    )
            else:
                raise ValueError(f'Unrecognized mode: {mode}')
        else:
            def s(rpow):
                s = (j/k * plexp(k/j * (rpow + t) + np.log(rpow) + np.log(k/j),
                            mode='upperbound'
                    ))**(1/k)
                return s.real
            if mode == 'integrate':
                return gamma(d/k - j/k).expect(
                    lambda rpow: 1 - wfun(rpow**(1/k), s(rpow), e, d),
                    points=[d/k-j/k-1], lb=0, ub=10*(d/k-j/k))
            elif mode == 'mc':
                rpow = gamma(d/k - j/k).rvs(size=nsamples)
                return np.mean(1 - wfun(rpow**(1/k), s(rpow), e, d))
            else:
                raise ValueError(f'Unrecognized mode: {mode}')
    
    def _find_NP_log_ratio(self, u, x0=0, bracket=(-100, 100)):
        return sp.optimize.root_scalar(
            lambda t: self._pbig(t, u) - 0.5, x0=x0, bracket=bracket)

    def _make_l2_table(self, inc=0.01, upper=3):
        from tqdm import tqdm
        table = {0: {'radius': 0, 'rho': 1/2}}
        prv_root = 0
        for eps in tqdm(np.arange(inc, upper + inc, inc)):
            e = eps * self._sigma()
            t = self._find_NP_log_ratio(e, prv_root)
            table[eps] = {
                't': t.root,
                'radius': e,
                'deg': self.k,
                'dim': self.dim,
                'normalized_radius': eps,
                'converged': t.converged,
                'info': t
            }
            if t.converged:
                table[eps]['rho'] = 1 - self._psmall(t.root, e)
                prv_root = t.root
        return table

    def make_l2_table(self, inc=0.01, upper=3, save=True):
        '''Calculate or load a table of robust radii for l2 adversary.
        First try to load a table under `./tables/` with the corresponding
        parameters. If this fails, calculate the table using level set method.
        Inputs:
            inc: grid increment (default: 0.01)
            upper: if `grid_type == 'radius'`, then the upper limit to the
                radius grid. (default: 3)
            save: whether to save the table computed
        Outputs:
            None, but `self.table`, `self.table_rho`, `self.table_radii`
            are now defined.
        '''
        from os.path import join
        basename = (f'exp2_l2_d{self.dim}_k{self.k}_j{self.j}' 
                    f'_inc{inc}_upper{upper}')
        rho_fname = join('tables', basename + '_rho.npy')
        radii_fname = join('tables', basename + '_radii.npy')
        try:
            self.table_rho = np.load(rho_fname)
            self.table_radii = np.load(radii_fname)
            self.table = dict(zip(self.table_rho, self.table_radii))
            print('Found and loaded saved table: '
                f'Exp2, L2 adv, dim={self.dim}, k={self.k}, j={self.j}, '
                f'inc={inc}, upper={upper}')
        except FileNotFoundError:
            print('Making robust radii table: Exp2, L2 adv')
            table = self._make_l2_table(inc, upper)
            self.table_radii = np.array([x['radius'] for x in table.values()])
            self.table_rho = np.array([x['rho'] for x in table.values()])
            self.table = dict(zip(self.table_rho, self.table_radii))
            if save:
                import os
                print('Saving robust radii table')
                os.makedirs('tables', exist_ok=True)
                np.save(rho_fname, self.table_rho)
                np.save(radii_fname, self.table_radii)

class Power2Noise(Noise):
    r'''L2-based distribution of the form (1 + \|x\|_2^k)^{-a}'''
    
    def __init__(self, device, dim, sigma=None, lambd=None, k=1, a=None):
        self.k = k
        if a is None:
            self.a = dim + 10
        else:
            self.a = a
        super().__init__(device, dim, sigma, lambd)
        self.table = self.table_radii = self.table_rho = self._table_info = None
        self.beta_dist = sp.stats.betaprime(dim / k, self.a - dim / k)
        self.beta_mode = (dim/k - 1) / (self.a - dim/k + 1)

    def _sigma(self):
        k = self.k
        a = self.a
        d = self.dim
        g = math.lgamma
        return np.exp(0.5 * (
                g((d+2)/k) + g(a - (d+2)/k) - g(d/k) - g(a - d/k) - np.log(d)
            ))

    def certifyl2(self, prob_lb, inc=0.01, upper=3, save=True):
        return self.certifyl2_levelset(prob_lb, inc, upper, save)

    def certifyl2_levelset(self, prob_lb, inc=0.01, upper=3, save=True):
        table_info = dict(inc=inc, upper=upper)
        if self.table_rho is None or self._table_info != table_info:
            self.make_l2_table(inc, upper, save)
            self._table_info = table_info
        return self.lambd * get_radii_from_convex_table(
                        self.table_rho, self.table_radii, prob_lb)
        
    def sample(self, x):
        samples = self.beta_dist.rvs((len(x), 1))
        radius = torch.tensor(samples**(1/self.k),
                    dtype=torch.float, device=self.device)
        noise = sample_l2_sphere(self.device, x.shape)
        return (self.lambd * radius * noise).view(x.shape) + x

    def _pbig(self, t, e, mode='integrate', nsamples=1000):
        '''Compute the big measure of a Neyman-Pearson set with ratio e^t.
        This function assumes `self.lambd == 1`.
        Inputs:
            t: log(kappa)
            e: the l2 norm of perturbation
            mode: integrate | mc
            nsamples: number of samples when `mode == 'mc'`
        Outputs:
        '''
        d = self.dim
        k = self.k
        a = self.a
        def s(rpow):
            return relu((1 + rpow) * np.exp(-t/a) - 1)**(1/k)
        def integrand(rpow):
            return wfun(rpow**(1/k), s(rpow), e, d)
        if mode == 'integrate':
            # return self.beta_dist.expect(integrand,
            #     points=[self.beta_mode], lb=0, ub=10*self.beta_mode)
            return self.beta_dist.expect(integrand)
        elif mode == 'mc':
            rpow = self.beta_dist.rvs(size=nsamples)
            return np.mean(wfun(rpow**(1/k), s(rpow), e, d))
        else:
            raise ValueError(f'Unrecognized mode: {mode}')


    def _psmall(self, t, e, mode='integrate', nsamples=1000):
        '''Compute the big measure of a Neyman-Pearson set with ratio e^t.
        This function assumes `self.lambd == 1`.
        Inputs:
            t: log(kappa)
            e: the l2 norm of perturbation
            mode: integrate | mc
            nsamples: number of samples when `mode == 'mc'`
        Outputs:
        '''
        d = self.dim
        k = self.k
        a = self.a
        def s(rpow):
            return relu((1 + rpow) * np.exp(t/a) - 1)**(1/k)
        def integrand(rpow):
            return 1 - wfun(rpow**(1/k), s(rpow), e, d)
        if mode == 'integrate':
            # return self.beta_dist.expect(integrand,
            #     points=[self.beta_mode], lb=0, ub=10*self.beta_mode)
            return self.beta_dist.expect(integrand)
        elif mode == 'mc':
            rpow = self.beta_dist.rvs(size=nsamples)
            return np.mean(integrand(rpow))
        else:
            raise ValueError(f'Unrecognized mode: {mode}')

    def _find_NP_log_ratio(self, u, x0=0, bracket=(-100, 100)):
        return sp.optimize.root_scalar(
            lambda t: self._pbig(t, u) - 0.5, x0=x0, bracket=bracket)

    def _make_l2_table(self, inc=0.01, upper=3):
        from tqdm import tqdm
        table = {0: {'radius': 0, 'rho': 1/2}}
        prv_root = 0
        for eps in tqdm(np.arange(inc, upper + inc, inc)):
            e = eps * self._sigma()
            t = self._find_NP_log_ratio(e, prv_root)
            table[eps] = {
                't': t.root,
                'radius': e,
                'normalized_radius': eps,
                'converged': t.converged,
                'info': t
            }
            if t.converged:
                table[eps]['rho'] = 1 - self._psmall(t.root, e)
                prv_root = t.root
        return table
       
    def make_l2_table(self, inc=0.01, upper=3, save=True):
        '''Calculate or load a table of robust radii for l2 adversary.
        First try to load a table under `./tables/` with the corresponding
        parameters. If this fails, calculate the table using level set method.
        Inputs:
            inc: grid increment (default: 0.01)
            upper: if `grid_type == 'radius'`, then the upper limit to the
                radius grid. (default: 3)
            save: whether to save the table computed
        Outputs:
            None, but `self.table`, `self.table_rho`, `self.table_radii`
            are now defined.
        '''
        from os.path import join
        basename = (f'pow2_l2_d{self.dim}_k{self.k}_a{self.a}' 
                    f'_inc{inc}_upper{upper}')
        rho_fname = join('tables', basename + '_rho.npy')
        radii_fname = join('tables', basename + '_radii.npy')
        try:
            self.table_rho = np.load(rho_fname)
            self.table_radii = np.load(radii_fname)
            self.table = dict(zip(self.table_rho, self.table_radii))
            print('Found and loaded saved table: '
                f'Pow2, L2 adv, dim={self.dim}, k={self.k}, a={self.a}, '
                f'inc={inc}, upper={upper}')
        except FileNotFoundError:
            print('Making robust radii table: Pow2, L2 adv')
            table = self._make_l2_table(inc, upper)
            self.table_radii = np.array([x['radius'] for x in table.values()])
            self.table_rho = np.array([x['rho'] for x in table.values()])
            self.table = dict(zip(self.table_rho, self.table_radii))
            if save:
                import os
                print('Saving robust radii table')
                os.makedirs('tables', exist_ok=True)
                np.save(rho_fname, self.table_rho)
                np.save(radii_fname, self.table_radii)

class Exp1Noise(Noise):
    r'''L1-based distribution of the form \|x\|_1^{-j} e^{\|x/\lambda\|_1^k}'''

    def __init__(self, device, dim, sigma=None, lambd=None, k=1, j=0):
        self.k = k
        self.j = j
        super().__init__(device, dim, sigma, lambd)
        self.gamma_dist = Gamma(
            concentration=torch.tensor((dim - j) / k, device=device),
            rate=1)

    def _sigma(self):
        k = self.k
        j = self.j
        d = self.dim
        return np.sqrt(2 / d / (d+1) * 
                    math.exp(math.lgamma((d + 2 - j) / k)
                            - math.lgamma((d - j) / k)
                        )
                    )

    def sample(self, x):
        radius = (self.gamma_dist.sample((len(x), 1))) ** (1 / self.k)
        radius *= self.lambd
        noises = sample_l1_sphere(self.device, x.shape)
        return noises * radius + x
    

if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt
    dim = 3072
    for j in [3071]:# [1, 2, 4, 8]:# [2048, 3064, 3068, 3071]:
        k=2
        noise = Power2Noise('cpu', dim, sigma=1, k=k, a=dim/k+100)
        before = time.time()
        # noise.make_linf_table(0.05)
        rs = torch.arange(0.5, 1, 0.01)
        cert1 = noise.certifyl2(rs, inc=0.5)
        # cert2 = noise.certifyl2_levelset(rs, inc=1)
        print(cert1)
        # print(cert2)
        # print((cert1 - cert2).std())
        # print(noise.table_rho)
        # # plt.plot(rs, cert1)
        # # plt.plot(rs, cert2)
        # # plt.show()
        # after = time.time()
        # print(noise.table)
        # print(noise.sample(torch.zeros(1000, dim)).std(), noise.sigma)
        # print('{:.3}'.format(after - before))
