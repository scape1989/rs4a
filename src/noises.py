import numpy as np
import torch
from torch.distributions import Normal, Exponential, Uniform


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
    """
    Appropriate for L2 certification.
    """
    def __init__(self, sigma, device):
        super().__init__(sigma, device)
        self.norm_dist = Normal()

    def sample(self, shape):
        return (torch.rand(shape, device=self.device) - 0.5) * 2 * self.sigma

    def certify(self, prob_lower_bound):
        return self.sigma * self.norm_dist.icdf(prob_lower_bound)

 
class LaplaceNoise(Noise):
    """
    Appropriate for L1 certification.
    """
    def certify(self, prob_lower_bound):
        return torch.max(
            0.5 * self.sigma * torch.log(prob_lower_bound) - 
                torch.log(1 - prob_lower_bound),
            -self.sigma * (torch.log(1 - prob_lower_bound) + torch.log(2))
        )


class ExpInfNoise(Noise):
    """
    Appropriate for L1 certification.
    """
    def sample(self, shape):
        nsamples = np.prod(np.asarray(shape))
        # uniformly sample in unit cube
        points = 2 * torch.rand(nsamples, d) - 1
        # uniformly sample a direction to project
        idxs = torch.randint(0, d, [nsamples])
        points[list(range(nsamples)), idxs] = 2 * torch.randint(0, 2, [nsamples]).float() - 1
        # sample radii
        radii = rdist.sample([nsamples])
        points *= radii[:, None]
        finalshape = list(shape) + [d]
        return points.reshape(*finalshape)

    def certify(self, prob_lower_bound):
        pass

class UniformNoise(Noise):
    """
    Appropriate for L1 certification.
    """
    def sample(self, shape):
        return (torch.rand(shape, device=self.device) - 0.5) * 2 * self.sigma

    def certify(self, prob_lower_bound):
        return self.sigma * (prob_lower_bound - 0.5)

