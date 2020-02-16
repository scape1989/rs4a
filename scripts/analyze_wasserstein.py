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
            df["train_acc"].append(tsv["train_acc"][0] / 100)

    # save the experiment results
    df = pd.DataFrame(df) >> arrange(X.noise)
    df.to_csv(f"{args.dir}/results.csv", index=False)

    if args.debug:
        breakpoint()

    # plot clean training and testing accuracy
    grouped = df >> group_by(X.noise, X.sigma) \
                 >> summarize(noise=first(X.noise),
                              sigma=first(X.sigma),
                              top_1_acc_train=first(X.train_acc),
                              top_1_acc_pred=first(X.top_1_acc_pred))

    plt.figure(figsize=(6.5, 2.5))
    plt.subplot(1, 2, 1)
    sns.lineplot(x="sigma", y="top_1_acc_train", hue="noise", markers=True, dashes=False,
                 style="noise", data=grouped, alpha=1)
    plt.xlabel("$\sigma$")
    plt.ylabel("Top-1 training accuracy")
    plt.ylim((0, 1))
    plt.subplot(1, 2, 2)
    sns.lineplot(x="sigma", y="top_1_acc_pred", hue="noise", markers=True, dashes=False,
                 style="noise", data=grouped, alpha=1, legend=False)
    plt.xlabel("$\sigma$")
    plt.ylabel("Top-1 testing accuracy")
    plt.ylim((0, 1))
    plt.tight_layout()
    plt.savefig(f"{args.dir}/train_test_accuracies.pdf")
    plt.show()
#
#    tmp = df >> mask(X.eps.isin((0.015, 0.02, 0.025, 0.03))) >> \
#                mutate(top1accpred=X.top_1_acc_pred, top1acccert=X.top_1_acc_cert)
#    fig = sns.relplot(data=tmp, kind="scatter", x="top1accpred", y="top1acccert",
#                      hue="noise", col="eps", col_wrap=2, aspect=1, height=3, size="sigma")
#    fig.map_dataframe(plt.plot, (plt.xlim()[0], plt.xlim()[1]), (plt.xlim()[0], plt.xlim()[1]), 'k--').set_axis_labels("top1acctrain", "top1acccert").add_legend()
#    plt.show()
#
    # plot clean training accuracy against certified accuracy at eps
#    tmp = df >> mask(X.eps == 0.025) >> arrange(X.noise)
#    plt.figure(figsize=(3, 3))
#    sns.scatterplot(x="top_1_acc_pred", y="top_1_acc_cert", hue="noise", style="noise",
#                    size="sigma", data=tmp, legend=False)
#    plt.plot(np.linspace(0.0, 1.0), np.linspace(0.0, 1.0), "--", color="gray")
#    plt.ylim((0, 1))
#    plt.xlim((0.0, 1.0))
#    plt.xlabel("Top-1 training accuracy")
#    plt.ylabel("Top-1 certified accuracy, $\epsilon$ = 0.25")
#    plt.tight_layout()
#    plt.savefig(f"{args.dir}/train_vs_certified.eps")
#    plt.show()
#
    # plot certified accuracies
    selected = df >> mutate(certacc=X.top_1_acc_cert)
    sns.relplot(x="eps", y="certacc", hue="noise", kind="line", col="sigma",
                col_wrap=4, data=selected, height=2, aspect=1.5)
    plt.ylim((0, 1))
    plt.legend()
    plt.tight_layout()
#
    # plot top certified accuracy per epsilon, per type of noise
    grouped = df >> mask(X.noise != "expinf") \
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
    plt.savefig(f"{args.dir}/certified_accuracies_l1.pdf")
    plt.show()

