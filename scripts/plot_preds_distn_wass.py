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
    argparser.add_argument("--dir", default="./ckpts", type=str)
    argparser.add_argument("--target", default="prob_correct", type=str)
    argparser.add_argument("--use-pdf", action="store_true")
    args = argparser.parse_args()

    dataset = args.dir.split("_")[0]
    experiment_names = list(filter(lambda x: x.startswith(dataset), os.listdir(args.dir)))

    sns.set_style("white")
    sns.set_palette("husl")

    losses_df = pd.DataFrame({"noise": [], "sigma": [], "losses_train": [], "iter": []})

    for experiment_name in experiment_names:

        if not experiment_name.endswith("txt"):
            continue

        noise = experiment_name.split("_")[5]
        sigma = float(experiment_name.split("_")[7])
        tsv = pd.read_csv(f"{args.dir}/{experiment_name}", sep="\t")

        p_correct = tsv["P_c"]
        p_top = tsv["P_a"]

        if args.target == "prob_lower_bound":
            tgt = p_top
            axis = np.linspace(0, 1, 500)
        elif args.target == "prob_correct":
            tgt = p_correct
            axis = np.linspace(0, 1, 500)
        elif args.target == "radius_smooth":
            tgt = tsv["radius"]
            tgt = tgt[~np.isnan(tgt)]
            axis = np.linspace(0, 4.0, 500)
        else:
            raise ValueError

        if args.use_pdf:
            cdf = sp.stats.gaussian_kde(tgt)(axis)
        else:
            cdf = (tgt[np.newaxis,:] < axis[:, np.newaxis]).mean(axis=1)

        losses_df >>= bind_rows(pd.DataFrame({
            "experiment_name": experiment_name,
            "noise": noise,
            "sigma": sigma,
            "cdf": cdf,
            "axis": axis}))

    # show training curves
    sns.relplot(x="axis", y="cdf", hue="noise", data=losses_df, col="sigma",
                col_wrap=2, kind="line", height=1.5, aspect=2.5, alpha=0.5)
    plt.show()

