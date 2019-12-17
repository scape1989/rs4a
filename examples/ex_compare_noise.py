import numpy as np
import seaborn as sns
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from src.noises import *


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--dim", type=int, default=3000)
    argparser.add_argument("--batch-size", type=int, default=40)
    argparser.add_argument("--sigma", type=float, default=0.25)
    args = argparser.parse_args()

    #noises = ["GaussianNoise", "LaplaceNoise", "UniformNoise"]
    noises = ["LomaxNoise3", "LomaxNoise5", "LomaxNoise7"]
    #noises = ["GammaNoise3", "GammaNoise4", "GammaNoise5"]

    sns.set_style("whitegrid")
    sns.set_palette("husl")

    plt.figure(figsize=(8, 3))
    axis = np.linspace(0, 2, 200)

    for noise_str in noises:

        if noise_str[-1].isdigit():
            k = int(noise_str[-1])
            noise = eval(noise_str[:-1])(sigma=args.sigma, device="cpu", p=2, k=k, dim=args.dim)
        else:
            noise = eval(noise_str)(sigma=args.sigma, device="cpu", p=2)

        rvs = noise.sample((args.batch_size, 2, args.dim))
        rvs = rvs.reshape((args.batch_size, -1))

        plt.plot(axis, gaussian_kde(np.abs(rvs).flatten())(axis), label=noise_str)

    plt.legend()
    plt.show()
