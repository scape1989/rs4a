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
    argparser.add_argument("--eps-max", default=5.0, type=float)
    args = argparser.parse_args()

    dataset = args.dir.split("_")[0]
    experiment_names = list(filter(lambda x: x.startswith(dataset), os.listdir(args.dir)))

    sns.set_style("whitegrid")
    sns.set_palette("husl")

    df = defaultdict(list)
    eps_range = np.linspace(0, args.eps_max, 81)

    for experiment_name in experiment_names:

        save_path = f"{args.dir}/{experiment_name}"
        results = {}

        for k in  ("preds_smooth", "labels", "radius_smooth"):
            results[k] = np.load(f"{save_path}/{k}.npy")

        top_1_preds_smooth = np.argmax(results["preds_smooth"], axis=1)
        top_1_acc_pred = (top_1_preds_smooth == results["labels"]).mean()

        _, _, noise, sigma, adv_eps = experiment_name.split("_")
        sigma = float(sigma)
        adv_eps = float(adv_eps)
        noise = noise.replace("Noise", "")

        for eps in eps_range:

            top_1_acc_cert = ((results["radius_smooth"] >= eps) & \
                              (top_1_preds_smooth == results["labels"])).mean()
            df["experiment_name"].append(experiment_name)
#            df["sigma"].append(experiment_args.sigma / (3 * 32 * 32 / k) ** 0.5)
            df["sigma"].append(sigma)
            df["noise"].append(noise)
            df["eps"].append(eps)
            df["adv_eps"].append(adv_eps)
            df["top_1_acc_train"].append(0.0)
            df["top_1_acc_cert"].append(top_1_acc_cert)
            df["top_1_acc_pred"].append(top_1_acc_pred)

    # save the experiment results
    df = pd.DataFrame(df) >> arrange(X.noise) >> mask(X.noise != "Lomax")
    df.to_csv(f"{args.dir}/results_{dataset}.csv", index=False)

    if args.debug:
        breakpoint()

    # plot clean training and testing accuracy
    grouped = df >> group_by(X.experiment_name) \
                 >> summarize(experiment_name=first(X.experiment_name),
                              noise=first(X.noise),
                              sigma=first(X.sigma),
                              top_1_acc_train=first(X.top_1_acc_train),
                              top_1_acc_pred=first(X.top_1_acc_pred))
#
#    plt.figure(figsize=(6.5, 2.5))
#    plt.subplot(1, 2, 1)
#    sns.lineplot(x="sigma", y="top_1_acc_train", hue="noise", markers=True, dashes=False,
#                 style="noise", data=grouped, alpha=1)
#    plt.xlabel("$\sigma$")
#    plt.ylabel("Top-1 training accuracy")
#    plt.ylim((0, 1))
#    plt.subplot(1, 2, 2)
#    sns.lineplot(x="sigma", y="top_1_acc_pred", hue="noise", markers=True, dashes=False,
#                 style="noise", data=grouped, alpha=1, legend=False)
#    plt.xlabel("$\sigma$")
#    plt.ylabel("Top-1 testing accuracy")
#    plt.ylim((0, 1))
#    plt.tight_layout()
#    plt.savefig(f"{args.dir}/train_test_accuracies.eps")
#    plt.show()
#
    # plot top certified accuracy per epsilon, per type of noise
    grouped = df >> mask(X.sigma == 0.25) \
                 >> group_by(X.eps, X.adv_eps) \
                 >> arrange(X.top_1_acc_cert, ascending=False) \
                 >> summarize(top_1_acc_cert=first(X.top_1_acc_cert),
                              adveps=first(X.adv_eps))

    plt.figure(figsize=(3, 3))
    sns.lineplot(x="eps", y="top_1_acc_cert", data=grouped, hue="adveps", style="adveps")
    plt.ylim((0, 1))
    plt.xlabel("$\epsilon$")
    plt.ylabel("Top-1 certified accuracy")
    plt.tight_layout()
    plt.savefig(f"{args.dir}/certified_accuracies_l1.eps")
    plt.show()

