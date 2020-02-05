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

    sns.set_context("notebook", rc={"lines.linewidth": 1.5})
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    df = defaultdict(list)
    eps_range = np.linspace(0, args.eps_max, 81)

    for experiment_name in experiment_names:

        save_path = f"{args.dir}/{experiment_name}"
#        experiment_args = pickle.load(open(f"{args.dir}/{experiment_name}/args.pkl", "rb"))
        results = {}

        for k in  ("preds_smooth", "labels", "radius_smooth", "acc_train"):
            results[k] = np.load(f"{save_path}/{k}.npy")

        top_1_preds_smooth = np.argmax(results["preds_smooth"], axis=1)
        top_1_acc_pred = (top_1_preds_smooth == results["labels"]).mean()

        _, noise, sigma = experiment_name.split("_")
        sigma = float(sigma)
        noise = noise.replace("Noise", "")
#
#        if noise == "PTail2":
#            continue
#        if noise == "PTail8":
#            noise = "PowerLaw $\ell_2$"
#        if noise == "Exp2Poly1024":
#            noise = "Exponential $\ell_2$"
#        if noise == "Exp2Poly1":
#            continue
#        if noise == "UniformBall":
#            noise = "Uniform $\ell_2$"
#
        for eps in eps_range:

            top_1_acc_cert = ((results["radius_smooth"] >= eps) & \
                              (top_1_preds_smooth == results["labels"])).mean()
            df["experiment_name"].append(experiment_name)
#            df["sigma"].append(experiment_args.sigma / (3 * 32 * 32 / k) ** 0.5)
            df["sigma"].append(sigma)
            df["noise"].append(noise)
            df["eps"].append(eps)
            df["top_1_acc_train"].append(results["acc_train"][0])
            df["top_1_acc_cert"].append(top_1_acc_cert)
            df["top_1_acc_pred"].append(top_1_acc_pred)

    # save the experiment results
    df = pd.DataFrame(df) >> arrange(X.noise)
    df.to_csv(f"{args.dir}/results_{dataset}.csv", index=False)

    # print top-1 certified accuracies
    print(df >> mask(X.eps.isin((0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0))) \
             >> group_by(X.eps, X.noise) >> arrange(X.top_1_acc_cert, ascending=False) >> head(1))

    if args.debug:
        breakpoint()

    # plot clean training accuracy against certified accuracy at eps
    tmp = df >> mask(X.eps == 0.25) >> arrange(X.noise)
    plt.figure(figsize=(3, 3))
    sns.scatterplot(x="top_1_acc_train", y="top_1_acc_cert", hue="noise", style="noise",
                    size="sigma", data=tmp, legend=False)
    plt.plot(np.linspace(0.0, 1.0), np.linspace(0.0, 1.0), "--", color="gray")
    plt.ylim((0.2, 1.0))
    plt.xlim((0.2, 1.0))
    plt.xlabel("Top-1 training accuracy")
    plt.ylabel("Top-1 certified accuracy, $\epsilon$ = 0.25")
    plt.tight_layout()
    plt.savefig(f"{args.dir}/train_vs_certified.eps")
    plt.show()
#
#    tmp = df >> mask(X.eps.isin((0.25, 0.5, 0.75, 1.0))) >> \
#                mutate(tr=X.top_1_acc_train, cert=X.top_1_acc_cert)
#    fig = sns.relplot(data=tmp, kind="scatter", x="tr", y="cert",
#                      hue="noise", col="eps", col_wrap=2, aspect=1, height=3, size="sigma")
#    fig.map_dataframe(plt.plot, (plt.xlim()[0], plt.xlim()[1]), (plt.xlim()[0], plt.xlim()[1]), 'k--').set_axis_labels("tr", "cert").add_legend()
#    plt.show()
#
    # plot clean training and testing accuracy
    grouped = df >> group_by(X.experiment_name) \
                 >> summarize(experiment_name=first(X.experiment_name),
                              noise=first(X.noise),
                              sigma=first(X.sigma),
                              top_1_acc_train=first(X.top_1_acc_train),
                              top_1_acc_pred=first(X.top_1_acc_pred))

#    fig = sns.relplot(x="top_1_acc_train", y="top_1_acc_pred", hue="noise", col="sigma",
#                      style="noise", col_wrap=2, height=2, aspect=1, data=grouped)
#    fig.map_dataframe(plt.plot, (plt.xlim()[0], 1), (plt.xlim()[0],1), 'k--').set_axis_labels("Top-1 training accuracy", "Top-1 testing accuracy").add_legend()
#    plt.show()

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
    plt.savefig(f"{args.dir}/train_test_accuracies.eps")
    plt.show()

    # plot certified accuracies
    selected = df >> mutate(certacc=X.top_1_acc_cert)# >> mask(X.sigma <= 1.25) >> mask(X.noise != "ExpInf") >> mask(X.noise != "Lomax")
    sns.relplot(x="eps", y="certacc", hue="noise", kind="line", col="sigma",
                col_wrap=2, data=selected, height=2, aspect=1.5)
    plt.ylim((0, 1))
    plt.tight_layout()
    plt.savefig(f"{args.dir}/per_sigma.eps")
    plt.show()
#
    # plot top certified accuracy per epsilon, per type of noise
    grouped = df >> mask(X.noise != "Clean", X.noise != "ExpInf", X.noise != "Lomax") \
                 >> group_by(X.eps, X.noise) \
                 >> arrange(X.top_1_acc_cert, ascending=False) \
                 >> summarize(top_1_acc_cert=first(X.top_1_acc_cert),
                              noise=first(X.noise))

    grouped = pd.concat((grouped >> mask(X.noise == "Gaussian"), grouped >> mask(X.noise != "Gaussian")))

    plt.figure(figsize=(3, 3))
    sns.lineplot(x="eps", y="top_1_acc_cert", data=grouped, hue="noise", style="noise")
    plt.ylim((0, 1))
    plt.xlabel("$\epsilon$")
    plt.ylabel("Top-1 certified accuracy")
    plt.tight_layout()
    plt.savefig(f"{args.dir}/certified_accuracies_l1.eps")
    plt.show()

