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
    argparser.add_argument("--dir", default="./cifar_cross", type=str)
    args = argparser.parse_args()

    sns.set_style("white")
    sns.set_palette("husl")

    df = pd.DataFrame({"train_noise": [], "sigma": [], "test_noise": [], "top_1_acc": []})

    experiment_names = list(filter(lambda x: x.startswith(args.dataset), os.listdir(args.dir)))
    for i, experiment_name in enumerate(experiment_names):

        save_path = f"{args.dir}/{experiment_name}"
        experiment_args = pickle.load(open(f"{args.dir}/{experiment_name}/args.pkl", "rb"))
        results = {}

        for k in ("preds_smooth", "labels"):
            results[k] = np.load(f"{save_path}/{k}.npy")

        top_1_preds_smooth = np.argmax(results["preds_smooth"], axis=1)
        top_1_acc_pred = (top_1_preds_smooth == results["labels"]).mean()

        if np.isnan(top_1_acc_pred):
            breakpoint()

        df >>= bind_rows(pd.DataFrame({
            "train_noise": [experiment_name.split("_")[1]],
            "test_noise": [experiment_name.split("_")[2]],
            "sigma": [experiment_args.sigma],
            "top_1_acc": [top_1_acc_pred]}))

    plt.figure(figsize=(8, 8))
    for i, sigma in enumerate(sorted(list(set(df["sigma"])))):
        plt.subplot(3, 2, i + 1)
        curr_df = df >> mask(X.sigma == sigma) >> drop(X.sigma) >> spread(X.test_noise, X.top_1_acc)
        curr_df = curr_df.fillna(0).set_index("train_noise")
        sns.heatmap(curr_df, cmap="inferno", vmin=0, vmax=1)
        plt.title(f"Sigma = {sigma}")
    plt.tight_layout()
    plt.show()

