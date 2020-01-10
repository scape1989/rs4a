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
    argparser.add_argument("--eps-max", default=4.0, type=float)
    args = argparser.parse_args()

    dataset = args.dir.split("_")[0]
    experiment_names = list(filter(lambda x: x.startswith(dataset), os.listdir(args.dir)))

    sns.set_style("whitegrid")
    sns.set_palette("husl")

    df = defaultdict(list)
    eps_range = np.linspace(0, args.eps_max, 50)

    for experiment_name in experiment_names:

        save_path = f"{args.dir}/{experiment_name}"
        experiment_args = pickle.load(open(f"{args.dir}/{experiment_name}/args.pkl", "rb"))
        results = {}

        for k in  ("preds_smooth", "labels", "radius_smooth", "acc_train"):
            results[k] = np.load(f"{save_path}/{k}.npy")

        top_1_preds_smooth = np.argmax(results["preds_smooth"], axis=1)
        top_1_acc_pred = (top_1_preds_smooth == results["labels"]).mean()

#        k = get_trailing_number(experiment_args.noise)

        for eps in eps_range:

            top_1_acc_cert = ((results["radius_smooth"] >= eps) & \
                              (top_1_preds_smooth == results["labels"])).mean()
            df["experiment_name"].append(experiment_name)
#            df["sigma"].append(experiment_args.sigma / (3 * 32 * 32 / k) ** 0.5)
            df["sigma"].append(experiment_args.sigma)
            df["noise"].append(experiment_args.noise)
            df["eps"].append(eps)
            df["top_1_acc_train"].append(results["acc_train"][0])
            df["top_1_acc_cert"].append(top_1_acc_cert)
            df["top_1_acc_pred"].append(top_1_acc_pred)

    # save the experiment results
    df = pd.DataFrame(df)
    df.to_csv(f"{args.dir}/results_{dataset}.csv", index=False)

    if args.debug:
        breakpoint()

    # plot clean training and testing accuracy
    grouped = df >> mask(X.noise != "Clean") \
                 >> group_by(X.experiment_name) \
                 >> summarize(experiment_name=first(X.experiment_name),
                              noise=first(X.noise),
                              sigma=first(X.sigma),
                              top_1_acc_train=first(X.top_1_acc_train),
                              top_1_acc_pred=first(X.top_1_acc_pred))

    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    sns.lineplot(x="sigma", y="top_1_acc_train", hue="noise", markers=True, dashes=False,
                 style="noise", data=grouped, alpha=1)
    plt.xlabel("$\sigma$")
    plt.ylabel("Top-1 training accuracy")
    plt.ylim((0, 1))
    plt.subplot(2, 1, 2)
    sns.lineplot(x="sigma", y="top_1_acc_pred", hue="noise", markers=True, dashes=False,
                 style="noise", data=grouped, alpha=1)
    plt.xlabel("$\sigma$")
    plt.ylabel("Top-1 testing accuracy")
    plt.ylim((0, 1))
    plt.suptitle(args.dir)
    plt.tight_layout()
    plt.show()

    # plot certified accuracies
    selected = df >> mask(X.noise != "Clean")
    sns.relplot(x="eps", y="top_1_acc_cert", hue="noise", kind="line", col="sigma",
                col_wrap=2, data=selected, height=2, aspect=1.5)
    plt.ylim((0, 1))
    plt.suptitle(args.dir)
    plt.tight_layout()
    plt.show()

    # plot top certified accuracy per epsilon, per type of noise
    grouped = df >> mask(X.noise != "Clean") \
                 >> group_by(X.eps, X.noise) \
                 >> arrange(X.top_1_acc_cert, ascending=False) \
                 >> summarize(top_1_acc_cert=first(X.top_1_acc_cert),
                              noise=first(X.noise))

    sns.lineplot(x="eps", y="top_1_acc_cert", data=grouped, hue="noise", style="noise")
    plt.title(args.dir)
    plt.xlabel("prob_lower_bound")
    plt.tight_layout()
    plt.show()

