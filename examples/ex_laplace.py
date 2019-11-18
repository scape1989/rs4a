import numpy as np
import scipy as sp
import scipy.stats
import itertools
from argparse import ArgumentParser
from matplotlib import pyplot as plt


def pdf_laplace(x, sigma=1., k=1.):
    sigma = sigma / 2 ** 0.5
    return np.exp(-(np.linalg.norm(x, ord=1) / sigma) ** k)

def pdf_gaussian(x, sigma=1., k=2.):
    return np.exp(-(np.linalg.norm(x, ord=2) / sigma) ** k)


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--sigma", default=1., type=float)
    argparser.add_argument("--k", default=1., type=float)
    argparser.add_argument("--n-samples", default=5000, type=int)
    args = argparser.parse_args()

    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    x_grid, y_grid = np.meshgrid(x, y)
    z_grid = np.zeros((50, 50))
    z_grid_gaussian = np.zeros((50, 50))

    for i, j in itertools.product(range(50), range(50)):
        z_grid[i, j] = pdf_laplace([x[i], y[j]], args.sigma, args.k)
        z_grid_gaussian[i, j] = pdf_gaussian([x[i], y[j]])

    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.contourf(x_grid, y_grid, z_grid)
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.contourf(x_grid, y_grid, z_grid_gaussian)
    plt.colorbar()
    plt.show()

