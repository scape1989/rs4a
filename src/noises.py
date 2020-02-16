import math
import numpy as np
import scipy as sp
import scipy.special
import scipy.stats
import torch
from torch.distributions import Normal, Uniform, Laplace, Gamma, Dirichlet, Pareto, Beta


def atanh(x):
    return 0.5 * np.log((1 + x) / (1 - x))


class Noise(object):

    def __init__(self, sigma, device, dim, k=None):
        self.sigma = sigma
        self.device = device
        self.dim = dim
        self.k = k

    def sample(self, x):
        raise NotImplementedError

    def certify(self, prob_lower_bound, p=None):
        raise NotImplementedError


class Clean(Noise):

    def __init__(self, sigma, device, dim):
        super().__init__(None, device, None)

    def sample(self, x):
        return x


class RotationNoise(Noise):

    def __init__(self, sigma, device, dim):
        super().__init__(sigma, device, dim)
        try:
            W = np.load("./src/lib/W.npy")
        except FileNotFoundError:
            W, _ = sp.linalg.qr(np.random.randn(dim, dim))
            np.save("./src/lib/W.npy", W)
        self.W = torch.tensor(W, device=device, dtype=torch.float)

    def sample(self, x):
        x_copy = x.view(x.shape[0], -1)
        return (x_copy @ self.W).view(x.shape)


class UniformNoise(Noise):

    def __init__(self, sigma, device, dim):
        super().__init__(sigma, device, dim)
        self.lambd = 3 ** 0.5 * sigma

    def sample(self, x):
        return (torch.rand_like(x) - 0.5) * 2 * self.lambd + x

    def certify(self, prob_lower_bound, p):
        if p == float("inf"):
            return 2 * self.lambd * (1 - (1.5 - prob_lower_bound) ** (1 / self.dim))
        if p > 1:
            raise ValueError(f"Unable to certify UniformNoise for p={p}.")
        return 2 * self.lambd * (prob_lower_bound - 0.5)


class RotatedUniformNoise(Noise):

    def __init__(self, sigma, device, dim):
        super().__init__(sigma, device, dim)
        self.lambd = sigma * 3 ** 0.5
        try:
            W = np.load("./src/lib/W.npy")
        except FileNotFoundError:
            W, _ = sp.linalg.qr(np.random.randn(dim, dim))
            np.save("./src/lib/W.npy", W)
        self.W = torch.tensor(W, device=device, dtype=torch.float)

    def sample(self, x):
        noise = (torch.rand_like(x) - 0.5) * 2 * self.lambd
        noise = noise @ self.W
        return x + noise


class GaussianNoise(Noise):

    def __init__(self, sigma, device, dim):
        super().__init__(sigma, device, dim)
        self.lambd = sigma
        self.norm_dist = Normal(loc=torch.tensor(0., device=device),
                                scale=torch.tensor(self.lambd, device=device))

    def sample(self, x):
        return torch.randn_like(x) * self.lambd + x

    def certify(self, prob_lower_bound, p):
        if p == float("inf"):
            return self.norm_dist.icdf(prob_lower_bound) / self.dim ** 0.5
        if p > 2:
            raise ValueError(f"Unable to certify UniformNoise for p={p}.")
        return self.norm_dist.icdf(prob_lower_bound)


class LaplaceNoise(Noise):

    def __init__(self, sigma, device, dim):
        super().__init__(sigma, device, dim)
        self.lambd = 2 ** (-0.5) * sigma
        self.laplace_dist = Laplace(loc=torch.tensor(0.0, device=device),
                                    scale=torch.tensor(self.lambd, device=device))

    def sample(self, x):
        return self.laplace_dist.sample(x.shape) + x

    def certify(self, prob_lower_bound, p):
        if p == float("inf"):
            return self.lambd * Normal(0, 1).icdf(prob_lower_bound) / self.dim ** 0.5
        if p > 1:
            raise ValueError(f"Unable to certify LaplaceNoise for p={p}.")
        a = 0.5 * self.lambd * torch.log(prob_lower_bound / (1 - prob_lower_bound))
        b = -self.lambd * (torch.log(2 * (1 - prob_lower_bound)))
        return torch.max(a, b)

class RotatedLaplaceNoise(Noise):

    def __init__(self, sigma, device, dim):
        super().__init__(self, device, None)
        self.lambd = sigma * 2 ** (-0.5)
        self.laplace_dist = Laplace(loc=torch.tensor(0., device=device),
                                    scale=torch.tensor(self.lambd, device=device))
        try:
            W = np.load("./src/lib/W.npy")
        except FileNotFoundError:
            W, _ = sp.linalg.qr(np.random.randn(dim, dim))
            np.save("./src/lib/W.npy", W)
        self.W = torch.tensor(W, device=device, dtype=torch.float)

    def sample(self, x):
        noise = self.laplace_dist.sample(x.shape)
        noise = noise @ self.W
        return x + noise

class LomaxNoise(Noise):

    def __init__(self, sigma, device, dim, k=3):
        super().__init__(sigma, device, None, k)
        self.lambd = math.sqrt(0.5 * (k - 1) * (k - 2)) * sigma if k > 2 else 0 / 0
        self.pareto_dist = Pareto(scale=torch.tensor(self.lambd, device=device, dtype=torch.float),
                                  alpha=torch.tensor(self.k, device=device, dtype=torch.float))

    def sample(self, x):
        samples = self.pareto_dist.sample(x.shape) - self.lambd
        signs = torch.sign(torch.rand_like(x) - 0.5)
        return samples * signs + x

    def certify(self, prob_lower_bound, p):
        if p > 1:
            raise ValueError(f"Unable to certify LomaxNoise for p={p}.")
        prob_lower_bound = prob_lower_bound.numpy()
        radius = sp.special.hyp2f1(1, self.k / (self.k + 1), self.k / (self.k + 1) + 1,
                                   (2 * prob_lower_bound - 1) ** (1 + 1 / self.k)) * \
                 self.lambd * (2 * prob_lower_bound - 1) / self.k
        return torch.tensor(radius, dtype=torch.float)


class GammaNoise(Noise):

    def __init__(self, sigma, device, dim, k=1):
        super().__init__(sigma, device, dim)
        self.lambd = (1 + k) ** 0.5 / (k * sigma)
        self.gamma_dist = Gamma(torch.tensor(1 / k, dtype=torch.float, device=device), self.lambd)

    def sample(self, x):
        noises = self.gamma_dist.sample(x.shape)
        sgns = torch.sign(torch.rand_like(x) - 0.5)
        return noises * sgns + x


class UniformBallNoise(Noise):

    def __init__(self, sigma, device, dim):
        super().__init__(sigma, device, dim)
        self.lambd = (dim + 2) ** 0.5 * sigma
        self.beta_dist = sp.stats.beta(0.5 * (self.dim + 1), 0.5 * (self.dim + 1))

    def sample(self, x):
        radius = torch.rand((len(x), 1), device=self.device) ** (1 / self.dim) * self.lambd
        noise = torch.randn(x.shape, device=self.device).reshape(len(x), -1)
        noise = noise / torch.norm(noise, dim=1, p=2).unsqueeze(1) * radius
        return noise + x

    def certify(self, prob_lower_bound, p):
        if p > 2:
            raise ValueError(f"Unable to certify UniformBall for p={p}.")
        radius = self.lambd * (2 - 4 * self.beta_dist.ppf(0.75 - 0.5 * prob_lower_bound.numpy()))
        return torch.tensor(radius, dtype=torch.float)


class ExpInfNoise(Noise):

    def __init__(self, sigma, device, dim, k=1):
        super().__init__(sigma, device, dim, k)
        self.lambd = 3 ** 0.5 * sigma / (math.exp(math.lgamma((dim + 1) / k) - math.lgamma(dim / k)))
        self.gamma_factor = math.exp(math.lgamma((dim + k) / k) - math.lgamma((dim + k - 1) / k))
        self.gamma_dist = Gamma(concentration=torch.tensor(dim / k, device=device),
                                rate=torch.tensor((1 / self.lambd) ** k, device=device))

    def sample(self, x):
        radius = (self.gamma_dist.sample((len(x), 1))) ** (1 / self.k)
        noise = (2 * torch.rand(x.shape, device=self.device) - 1).reshape((len(x), -1))
        sel_dims = torch.randint(noise.shape[1], size=(noise.shape[0],))
        idxs = torch.arange(0, noise.shape[0], dtype=torch.long)
        noise[idxs, sel_dims] = torch.sign(torch.rand(len(x), device=self.device) - 0.5)
        return (noise * radius).view(x.shape) + x

    def certify(self, prob_lower_bound, p):
        if p == float("inf"):
           return self.lambd * torch.log(0.5 / (1 - prob_lower_bound))
        if p > 1:
            raise ValueError(f"Unable to certify ExpInfNoise for p={p}.")
        return 2 * self.lambd * self.gamma_factor * (prob_lower_bound - 0.5)


class Exp1Noise(Noise):

    def __init__(self, sigma, device, dim, k=1):
        super().__init__(sigma, device, dim, k)
        self.lambd = sigma / (math.exp(math.lgamma((dim + 1) / k) - math.lgamma(dim / k)))
        self.lambd *= (0.5 * dim * (dim + 1)) ** 0.5
        self.gamma_factor = math.exp(math.lgamma(dim / k) - math.lgamma((dim + k - 1) / k))
        self.dirichlet_dist = Dirichlet(concentration=torch.ones(dim, device=device))
        self.gamma_dist = Gamma(concentration=torch.tensor(dim / k, device=device),
                                rate=torch.tensor((1 / self.lambd) ** k, device=device))

    def sample(self, x):
        radius = (self.gamma_dist.sample((len(x), 1))) ** (1 / self.k)
        noises = self.dirichlet_dist.sample((len(x),))
        signs = torch.sign(torch.rand_like(noises) - 0.5)
        return noises * signs * radius + x

    def certify(self, prob_lower_bound, p, num_pts=1000, eps=1e-4):
        if p > 1:
            raise ValueError(f"Unable to certify Exp1Noise for p={p}.")
        x = np.linspace(eps, 0.5, num_pts)
        y = sp.stats.gamma.ppf(1 - 2 * x, self.dim / self.k)
        y = 1 / (1 - sp.stats.gamma.cdf(y, (self.dim + self.k - 1) / self.k))
        y = np.repeat(y[np.newaxis,:], len(prob_lower_bound), axis=0)
        y[x < 1 - prob_lower_bound.numpy()[:, np.newaxis]] = 0
        integral = torch.tensor(np.trapz(y, dx=0.5 / num_pts), dtype=torch.float)
        return 2 * self.lambd * self.gamma_factor / self.k * integral


class Exp2Noise(Noise):

    def __init__(self, sigma, device, dim, k=1):
        super().__init__(sigma, device, dim, k)
        self.lambd = sigma / math.exp(math.lgamma((dim + 1) / k) - math.lgamma(dim / k))
        self.lambd *= dim ** 0.5
        self.beta_dist = sp.stats.beta(0.5 * (self.dim - 1), 0.5 * (self.dim - 1))
        self.gamma_dist = Gamma(concentration=torch.tensor(dim / k, device=device),
                                rate=torch.tensor((1 / self.lambd) ** k, device=device))

    def sample(self, x):
        radius = (self.gamma_dist.sample((len(x), 1))) ** (1 / self.k)
        noises = torch.randn_like(x)
        noises = noises / (noises ** 2).sum(dim=1).unsqueeze(1) ** 0.5
        return noises * radius + x

    def certify(self, prob_lower_bound, p):
        if p > 2:
            raise ValueError(f"Unable to certify Exp2Noise for p={p}.")
        radius = self.lambd * (self.dim - 1) * \
                 atanh(1 - 2 * self.beta_dist.ppf(1 - prob_lower_bound.numpy()))
        return torch.tensor(radius, dtype=torch.float)


class PowerLawNoise(Noise):

    def __init__(self, sigma, device, dim, k=2):
        super().__init__(sigma, device, dim, k)
        self.beta_dist = sp.stats.betaprime(dim, k)
        self.lambd = 3 ** 0.5 * (k - 1) * sigma / dim

    def sample(self, x):
        samples = self.beta_dist.rvs((len(x), 1))
        radius = torch.tensor(samples, dtype=torch.float, device=self.device)
        noise = (2 * torch.rand(x.shape, device=self.device) - 1).reshape((len(x), -1))
        sel_dims = torch.randint(noise.shape[1], size=(len(x),))
        idxs = torch.arange(0, len(x), dtype=torch.long)
        noise[idxs, sel_dims] = torch.sign(torch.rand(len(x), device=self.device) - 0.5)
        return (noise * radius * self.lambd).view(x.shape) + x

    def certify(self, prob_lower_bound, p):
        if p > 1:
            raise ValueError(f"Unable to certify PowerLawNoise for p={p}.")
        return self.lambd * 2 * self.dim / self.k * (prob_lower_bound - 0.5)


class PTailNoise(Noise):

    def __init__(self, sigma, device, dim, k=2):
        super().__init__(sigma, device, dim, k)
        self.beta_dist = sp.stats.betaprime(0.5 * dim, k)
        self.lambd = (2 * (k - 1)) ** 0.5 * sigma
        self.prob_lower_bounds, self.radii = np.load("./src/lib/radii_ptail.npy", allow_pickle=True)

    def sample(self, x):
        samples = self.beta_dist.rvs((len(x), 1))
        radius = torch.tensor(samples, dtype=torch.float, device=self.device) ** 0.5
        noise = torch.randn_like(x)
        noise = noise / torch.norm(noise, dim=1, p=2).unsqueeze(1)
        return noise * radius * self.lambd + x

    def certify(self, prob_lower_bound, p):
        if p > 2:
            raise ValueError(f"Unable to certify PTailNoise for p={p}.")
        prob_lower_bound = prob_lower_bound.numpy()
        alpha = self.k + 0.5 * self.dim
        idxs = np.searchsorted(self.prob_lower_bounds[alpha], prob_lower_bound)
        idxs = np.minimum(idxs, len(self.prob_lower_bounds[alpha]) - 1)
        x_deltas = self.prob_lower_bounds[alpha][idxs] - self.prob_lower_bounds[alpha][idxs - 1]
        y_deltas = self.radii[alpha][idxs] - self.radii[alpha][idxs - 1]
        pcts = (prob_lower_bound - self.prob_lower_bounds[alpha][idxs - 1]) / x_deltas
        radius = (pcts * y_deltas + self.radii[alpha][idxs - 1]) * self.lambd
        return torch.tensor(radius, dtype=torch.float)


class Exp2PolyNoise(Noise):

    def __init__(self, sigma, device, dim, k=1):
        super().__init__(sigma, device, dim, k)
        self.lambd = (2 * dim / k) ** 0.5 * sigma
        self.gamma_dist = Gamma(torch.tensor(0.5 * self.k, device=device),
                                torch.tensor((1 / self.lambd) ** 2, device=device))
        self.prob_lower_bounds, self.radii = np.load("./src/lib/radii_exp2poly.npy",
                                                     allow_pickle=True)

    def sample(self, x):
        radius = self.gamma_dist.sample((len(x), 1)) ** 0.5
        noise = torch.randn(x.shape, device=self.device).reshape(len(x), -1)
        noise = noise / torch.norm(noise, dim=1, p=2).unsqueeze(1) * radius
        return noise + x

    def certify(self, prob_lower_bound, p):
        if p > 2:
            raise ValueError(f"Unable to certify Exp2PolyNoise for p={p}.")
        prob_lower_bound = prob_lower_bound.numpy()
        alpha = self.dim - self.k
        idxs = np.searchsorted(self.prob_lower_bounds[alpha], prob_lower_bound)
        idxs = np.minimum(idxs, len(self.prob_lower_bounds[alpha]) - 1)
        x_deltas = self.prob_lower_bounds[alpha][idxs] - self.prob_lower_bounds[alpha][idxs - 1]
        y_deltas = self.radii[alpha][idxs] - self.radii[alpha][idxs - 1]
        pcts = (prob_lower_bound - self.prob_lower_bounds[alpha][idxs - 1]) / x_deltas
        radius = (pcts * y_deltas + self.radii[alpha][idxs - 1]) * self.lambd
        return torch.tensor(radius, dtype=torch.float)


class MaskGaussianNoise(Noise):

    def __init__(self, sigma, device, dim, k=2):
        """
        k is reciprocal of fraction of pixels retained (default 1/2)
        """
        super().__init__(sigma, device, dim, k)
        self.n_pixels_retained = int(dim // k)
        self.lambd = sigma
        if dim == 3072:
            self.impute_val = 0.4734
        elif dim == 150528:
            self.impute_val = 0.449
        elif dim == 784:
            self.impute_val = 0.1307
        else:
            raise ValueError

    def sample(self, x):
        x_copy = x.view(len(x), -1)
        noise = x_copy + torch.randn_like(x_copy) * self.lambd
        perm = torch.stack([torch.randperm(self.dim) for _ in range(len(x))])
        idxs = perm[:, :self.dim - self.n_pixels_retained]
        for batch_no in range(len(x)):
            noise[batch_no, idxs[batch_no]] = self.impute_val
        return noise.reshape(x.shape)

    def certify(self, prob_lower_bound, p):
        return self.lambd * Normal(0, 1).icdf(prob_lower_bound) / self.n_pixels_retained ** 0.5



class MaskGaussianNoisePixel(Noise):

    def __init__(self, sigma, device, dim, k=2):
        """
        k is reciprocal of fraction of pixels retained (default 1/10)
        """
        super().__init__(sigma, device, dim, k)
        self.n_pixels_retained = int((dim / 3) // k)
        self.lambd = sigma

    def sample(self, x):
        x_copy = x.view(len(x), -1)
        noise = x_copy + torch.randn_like(x_copy) * self.lambd
        perm = torch.stack([torch.randperm(self.dim // 3) for _ in range(len(x))])
        idxs = perm[:, :int((self.dim // 3) - self.n_pixels_retained)]
        for batch_no in range(len(x)):
            noise[batch_no, idxs[batch_no]] = 0.4734
            noise[batch_no, idxs[batch_no] + 1024] = 0.4734
            noise[batch_no, idxs[batch_no] + 2048] = 0.4734
        return noise.reshape(x.shape)

    def certify(self, prob_lower_bound, p):
        return self.lambd * Normal(0, 1).icdf(prob_lower_bound) / (self.n_pixels_retained * 3) ** 0.5


class MaskGaussianNoisePatch(Noise):

    def __init__(self, sigma, device, dim, k=2):
        """
        k is reciprocal of fraction of pixels retained (default 1/10)
        """
        super().__init__(sigma, device, dim, k)
        self.patch_width = int((dim / 3 / k) ** 0.5)
        self.n_pixels_retained = self.patch_width ** 2 * 3
        self.lambd = sigma

    def sample(self, x):
        x = x.view(len(x), -1)
        noise = torch.randn_like(x) * self.lambd + x
        top_left_x = torch.randint(int((self.dim / 3)** 0.5) - self.patch_width + 1, (len(x),))
        top_left_y = torch.randint(int((self.dim / 3)** 0.5) - self.patch_width + 1, (len(x),))
        for batch_no in range(len(x)):
            noise[batch_no, :, :top_left_x[batch_no], :] = np.nan
            noise[batch_no, :, :, :top_left_y[batch_no]] = np.nan
            noise[batch_no, :, top_left_x[batch_no] + self.patch_width:, :] = np.nan
            noise[batch_no, :, :, top_left_y[batch_no] + self.patch_width:] = np.nan
        return noise

    def certify(self, prob_lower_bound, p):
        return self.lambd * Normal(0, 1).icdf(prob_lower_bound) / self.n_pixels_retained ** 0.5


class MaskGaussianNoisePatchSmall(Noise):

    def __init__(self, sigma, device, dim, k=2):
        """
        k is reciprocal of fraction of pixels retained (default 1/10)
        """
        super().__init__(sigma, device, dim, k)
        self.patch_width = 5
        self.num_patches = int(dim / 3 / k / self.patch_width ** 2)
        self.n_pixels_retained = self.patch_width ** 2 * 3 * self.num_patches
        self.lambd = sigma
        self.norm_dist = Normal(loc=torch.tensor(0., device=device),
                                scale=torch.tensor(self.lambd, device=device))

    def sample(self, x):
        x_copy = x.reshape(-1, 3, 32, 32)
        batch_size = x_copy.shape[0]
        noise = self.norm_dist.sample(x_copy.shape)
        sample = torch.zeros_like(x_copy)
        for _ in range(self.num_patches):
            top_left_x = torch.randint(int((self.dim / 3)** 0.5) - self.patch_width + 1,
                                       (batch_size,))
            top_left_y = torch.randint(int((self.dim / 3)** 0.5) - self.patch_width + 1,
                                       (batch_size,))
            for batch_no in range(batch_size):
                i = top_left_x[batch_no]
                j = top_left_y[batch_no]
                sample[batch_no, :, i:i+self.patch_width, j:j+self.patch_width] = noise[batch_no,:,i:i+self.patch_width, j:j+self.patch_width] + x_copy[batch_no,:,i:i+self.patch_width, j:j+self.patch_width]
        return sample.reshape(x.shape)

    def certify(self, prob_lower_bound, p):
        return self.lambd * Normal(0, 1).icdf(prob_lower_bound) / self.n_pixels_retained ** 0.5


