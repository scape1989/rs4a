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
from src.utils import get_trailing_number


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--dir", default="./ckpts", type=str)
    argparser.add_argument("--debug", action="store_true")
    argparser.add_argument("--eps-max", default=0.08, type=float)
    args = argparser.parse_args()

    experiment_names = os.listdir(args.dir)

    sns.set_style("whitegrid")
    sns.set_palette("husl")

    df = defaultdict(list)
    eps_range = np.linspace(0, args.eps_max, 81)

    for experiment_name in experiment_names:

        if not experiment_name.endswith("txt"):
            continue

        noise = experiment_name.split("_")[5]
        sigma = float(experiment_name.split("_")[7])

        tsv = pd.read_csv(f"{args.dir}/{experiment_name}", sep="\t")

        top_1_acc_pred = (tsv["predict"] == tsv["label"]).mean()

        for eps in eps_range:

            top_1_acc_cert = ((tsv["radius"] >= eps) & \
                              (tsv["predict"] == tsv["label"])).mean()
#            df["sigma"].append(experiment_args.sigma / (3 * 32 * 32 / k) ** 0.5)
            df["sigma"].append(sigma)
            df["noise"].append(noise)
            df["eps"].append(eps)
            df["top_1_acc_cert"].append(top_1_acc_cert)
            df["top_1_acc_pred"].append(top_1_acc_pred)

    # save the experiment results
    df = pd.DataFrame(df) >> arrange(X.noise)
    df.to_csv(f"{args.dir}/results.csv", index=False)

    if args.debug:
        breakpoint()

    # plot certified accuracies
#    selected = df >> mask(X.noise != "Clean")
#    sns.relplot(x="eps", y="top_1_acc_cert", hue="noise", kind="line", col="sigma",
#                col_wrap=2, data=selected, height=2, aspect=1.5)
#    plt.ylim((0, 1))
#    plt.suptitle(args.dir)
#    plt.tight_layout()
#    plt.show()

    # plot top certified accuracy per epsilon, per type of noise
    grouped = df >> mask(X.noise != "Clean") \
                 >> mask(X.sigma < 0.1) \
                 >> group_by(X.eps, X.noise) \
                 >> arrange(X.top_1_acc_cert, ascending=False) \
                 >> summarize(top_1_acc_cert=first(X.top_1_acc_cert),
                              noise=first(X.noise))

    plt.figure(figsize=(3, 3))
    sns.lineplot(x="eps", y="top_1_acc_cert", data=grouped, hue="noise", style="noise")
    plt.ylim((0, 1))
    plt.xlabel("$\epsilon$")
    plt.ylabel("Top-1 certified accuracy")
    plt.tight_layout()
    plt.savefig(f"{args.dir}/certified_accuracies_l1.eps")
    plt.show()

