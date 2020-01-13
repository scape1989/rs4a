
import numpy as np
import pandas as pd
import os
import pickle
import matplotlib as mpl
import seaborn as sns
from argparse import ArgumentParser
from collections import defaultdict
from dfply import *
from matplotlib import pyplot as plt


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--dir", default="./ckpts", type=str)
    argparser.add_argument("--debug", action="store_true")
    argparser.add_argument("--eps-max", default=4.0, type=float)
    args = argparser.parse_args()

    dataset = args.dir.split("_")[0]
    experiment_names = list(filter(lambda x: x.startswith(dataset), os.listdir(args.dir)))

    sns.set_style("whitegrid")
    sns.set_palette("husl")

    df = defaultdict(list)
    eps_range = np.linspace(0, 1, 50)

    for experiment_name in experiment_names:

        save_path = f"{args.dir}/{experiment_name}"
        experiment_args = pickle.load(open(f"{args.dir}/{experiment_name}/args.pkl", "rb"))
        results = {}

        for k in ("prob_lower_bound",):
            results[k] = np.load(f"{save_path}/{k}.npy")

        for eps in eps_range:

            df["experiment_name"].append(experiment_name)
            df["sigma"].append(experiment_args.sigma)
            df["noise"].append(experiment_args.noise)
            df["eps"].append(eps)
            df["proportion_over"].append((results["prob_lower_bound"] > eps).mean())

    df = pd.DataFrame(df)
    if args.debug:
        breakpoint()

    sns.relplot(x="eps", y="proportion_over", hue="noise", kind="line", col="sigma",
                col_wrap=2, data=df, height=2, aspect=1.5)
    plt.suptitle(args.dir)
    plt.tight_layout()
    plt.xlabel("p_A")
    plt.show()

