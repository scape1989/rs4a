import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
import os
import pickle
import matplotlib as mpl
import seaborn as sns
from argparse import ArgumentParser
from dfply import *
from matplotlib import pyplot as plt


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--dataset", default="cifar", type=str)
    argparser.add_argument("--dir", default="./ckpts", type=str)
    args = argparser.parse_args()

    sns.set_style("white")
    sns.set_palette("husl")

    df = pd.DataFrame({"noise": [], "sigma": [], "lower_bounds": []})

    experiment_names = list(filter(lambda x: x.startswith(args.dataset), os.listdir(args.dir)))
    for i, experiment_name in enumerate(experiment_names):

        save_path = f"{args.dir}/{experiment_name}"
        experiment_args = pickle.load(open(f"{args.dir}/{experiment_name}/args.pkl", "rb"))
        results = {}

        for k in ("preds_smooth", "labels"):
            results[k] = np.load(f"{save_path}/{k}.npy")

        lower_bounds = results["preds_smooth"][np.arange(10000), results["labels"].astype(int)]

        df >>= bind_rows(pd.DataFrame({
            "noise": experiment_args.noise,
            "sigma": experiment_args.sigma,
            "lower_bounds": lower_bounds,
            "idx": np.arange(10000)}))

    df = df >> spread(X.noise, X.lower_bounds) >> drop(X.idx)
    grid = sns.FacetGrid(df, col="sigma", col_wrap=2, height=3, aspect=1.5)
    grid.map_dataframe(lambda data, **kwargs: sns.heatmap((data >> drop(X.sigma)).corr(), **kwargs),
                       cmap="viridis", vmin=0.75, vmax=1.0)
    plt.tight_layout()
    plt.show()

