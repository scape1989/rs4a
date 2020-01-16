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

        save_path = f"{args.dir}/{experiment_name}"
        experiment_args = pickle.load(open(f"{args.dir}/{experiment_name}/args.pkl", "rb"))
        results = {}

        for k in ("prob_lower_bound", "preds_smooth", "labels", "radius_smooth"):
            results[k] = np.load(f"{save_path}/{k}.npy")

        p_correct = results["preds_smooth"][np.arange(len(results["preds_smooth"])),
                                            results["labels"].astype(int)]
        p_top = results["prob_lower_bound"]

        if args.target == "prob_lower_bound":
            tgt = p_top
            axis = np.linspace(0.5, 1, 500)
        elif args.target == "prob_correct":
            tgt = p_correct
            axis = np.linspace(0.5, 1, 500)
        elif args.target == "radius_smooth":
            tgt = results["radius_smooth"]
            tgt = tgt[~np.isnan(tgt)]
            axis = np.linspace(0.5, 4.0, 500)
        else:
            raise ValueError

        if args.use_pdf:
            cdf = sp.stats.gaussian_kde(tgt)(axis)
        else:
            cdf = (tgt < axis[:, np.newaxis]).mean(axis=1)

        losses_df >>= bind_rows(pd.DataFrame({
            "experiment_name": experiment_name,
            "noise": experiment_args.noise,
            "sigma": experiment_args.sigma,
            "cdf": cdf,
            "axis": axis}))

    losses_df >>= mask(((X.noise == "GaussianNoise") & (X.sigma == 1.0)) |
                       ((X.noise == "LaplaceNoise") & (X.sigma == 0.5)) |
                       ((X.noise == "UniformNoise") & (X.sigma == 1.0)) |
                       ((X.noise == "LomaxNoise") & (X.sigma == 2.0)))
    losses_df >>= unite("label", [X.noise, X.sigma], sep=" ")

    # show training curves
    sns.lineplot(x="axis", y="cdf", hue="label", data=losses_df, alpha=0.5)
    plt.show()

