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

    experiment_names = list(filter(lambda x: x.startswith(args.dataset), os.listdir(args.dir)))

    sns.set_style("white")
    sns.set_palette("husl")

    losses_df = pd.DataFrame({"noise": [], "sigma": [], "losses_train": [], "iter": []})

    for experiment_name in experiment_names:

        save_path = f"{args.dir}/{experiment_name}"
        experiment_args = pickle.load(open(f"{args.dir}/{experiment_name}/args.pkl", "rb"))
        results = {}

        for k in ("prob_lower_bound",):
            results[k] = np.load(f"{save_path}/{k}.npy")

       # preds = results["preds_smooth"][np.arange(len(results["preds_smooth"])),
       #                                 results["labels"].astype(int)]
        axis = np.linspace(0, 1, 500)
        kde = sp.stats.gaussian_kde(results["prob_lower_bound"])(axis)

        losses_df >>= bind_rows(pd.DataFrame({
            "experiment_name": experiment_name,
            "noise": experiment_args.noise,
            "sigma": experiment_args.sigma,
            "kde": kde,
            "axis": axis}))

    # show training curves
    losses_df >>= mask((X.sigma >= 0.15) & (X.sigma <= 1.25))
    sns.relplot(x="axis", y="kde", hue="noise", data=losses_df, col="sigma",
                col_wrap=2, kind="line", height=1.5, aspect=3.5, alpha=0.5)
    plt.tight_layout()
    plt.show()

