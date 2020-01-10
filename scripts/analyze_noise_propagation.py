import numpy as np
import pandas as pd
import seaborn as sns
import itertools
from collections import defaultdict
from matplotlib import pyplot as plt


if __name__ == "__main__":

    noises = ["GaussianNoise", "LaplaceNoise", "UniformNoise"]
    sigmas = (0.25, 0.5, 1.0, 1.25)

    df = defaultdict(list)

    for noise, sigma in itertools.product(noises, sigmas):

        exp_name = f"cifar_{noise}_0.15"
        diffs_l2_mean = np.load(f"cifar_l2/{exp_name}/diffs_l2_mean_{noise}_{sigma}.npy")
        df["noise"].append(noise)
        df["sigma"].append(sigma)
        df["diff_l2_mean"].append(diffs_l2_mean.mean())

    df = pd.DataFrame(df)
    sns.lineplot(x="sigma", y="diff_l2_mean", hue="noise", style="noise", markers=True,
                 data=df, alpha=1)
    plt.show()

