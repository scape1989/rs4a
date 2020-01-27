import numpy as np
import pandas as pd
import seaborn as sns
from argparse import ArgumentParser
from collections import defaultdict
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from src.noises import *
from src.utils import get_trailing_number


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--dim", type=int, default=3000)
    argparser.add_argument("--sigma", type=float, default=0.25)
    args = argparser.parse_args()

    noises = ["GaussianNoise", "LaplaceNoise", "UniformNoise"]#, "LomaxNoise"]

    sns.set_style("whitegrid")
    sns.set_palette("husl")

    plt.figure(figsize=(4, 2))
    axis = np.linspace(0.5, 1.0, 400)

    df = defaultdict(list)

    for noise_str in noises:

        k = get_trailing_number(noise_str)
        if k:
            noise = eval(noise_str[:-len(str(k))])(sigma=args.sigma, device="cpu", p=2,
                                                   dim=args.dim, k=k)
        else:
            noise = eval(noise_str)(sigma=args.sigma, device="cpu", p=2, dim=args.dim)

        radii = noise.certify(torch.tensor(axis)).numpy()
        df["radius"] += radii.tolist()
        df["axis"] += axis.tolist()
        df["noise"] += [noise_str] * len(axis)

    df = pd.DataFrame(df)
    sns.lineplot(x="axis", y="radius", hue="noise", style="noise", data=df)
    plt.xlabel("Probability lower bound")
    plt.ylabel("Certified radius")
    plt.tight_layout()
    plt.savefig("./figs/ex_certification.eps")
    plt.show()

