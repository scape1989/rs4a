import numpy as np
import scipy as sp
import scipy.stats
import itertools
from argparse import ArgumentParser
from matplotlib import pyplot as plt


def pdf(x, sigma=1.):
    return np.exp(-np.linalg.norm(x, ord=np.inf) / sigma)

def sample(n, sigma=1.):
    radius = sp.stats.gamma.rvs(size=(n, 1), a=2, scale=sigma)
    x = (2 * np.random.rand(n, 2) - 1) * radius
    idxs = np.random.choice(2, n)
    x[np.arange(n), idxs] = np.random.choice((-1, 1), n) * radius[:,0]
    return x

if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--sigma", default=1., type=float)
    argparser.add_argument("--n-samples", default=5000, type=int)
    args = argparser.parse_args()

    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    x_grid, y_grid = np.meshgrid(x, y)
    z_grid = np.zeros((50, 50))

    for i, j in itertools.product(range(50), range(50)):
        z_grid[i, j] = pdf([x[i], y[j]], args.sigma)

    x = sample(args.n_samples, args.sigma)

    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.contourf(x_grid, y_grid, z_grid)
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.hist2d(x[:,0], x[:,1], range=((-2, 2), (-2, 2)), density=True, bins=20)
    plt.colorbar()
    plt.show()

