import numpy as np
import pandas as pd
import pickle
import matplotlib as mpl
import seaborn as sns
from argparse import ArgumentParser
from collections import defaultdict
from dfply import *
from matplotlib import pyplot as plt


experiments = [
    "cifar", "cifar_laplace_025", "cifar_expinf_025", "cifar_uniform_025",
    "cifar_laplace_05", "cifar_expinf_05", "cifar_uniform_05",
    "cifar_laplace_075", "cifar_expinf_075", "cifar_uniform_075",
    "cifar_laplace_015", "cifar_expinf_015", "cifar_uniform_015",
    "cifar_laplace_1", "cifar_expinf_1", "cifar_uniform_1",
    "cifar_gaussian_025", "cifar_gaussian_05", "cifar_gaussian_075",
    "cifar_gaussian_015", "cifar_gaussian_1",
    "cifar_gaussian_035", "cifar_expinf_035", "cifar_laplace_035", "cifar_uniform_035"
]

if __name__ == "__main__":

    sns.set_style("white")
    mpl.style.use("seaborn-dark-palette")

    df = defaultdict(list)
    eps_range = np.linspace(0, 1.5, 50)

    for experiment_name in experiments:

        save_path = f"ckpts/{experiment_name}"
        args = pickle.load(open(f"ckpts/{experiment_name}/args.pkl", "rb"))
        results = {}

        for k in  ("preds_smooth", "labels", "radius_smooth", "acc_train"):
            results[k] = np.load(f"{save_path}/{k}.npy")

        top_1_preds_smooth = np.argmax(results["preds_smooth"], axis=1)

        for eps in eps_range:

            top_1_acc_cert = ((results["radius_smooth"] >= eps) & \
                              (top_1_preds_smooth == results["labels"])).mean()
            top_1_acc_pred = (top_1_preds_smooth == results["labels"]).mean()
            df["experiment_name"].append(experiment_name)
            df["sigma"].append(args.sigma)
            df["noise"].append(args.noise)
            df["eps"].append(eps)
            df["top_1_acc_train"].append(results["acc_train"][0])
            df["top_1_acc_cert"].append(top_1_acc_cert)
            df["top_1_acc_pred"].append(top_1_acc_pred)

    df = pd.DataFrame(df)
    df.to_csv("./ckpts/results.csv", index=False)

    grouped = df >> mask(X.noise != "Clean") \
                 >> group_by(X.experiment_name) \
                 >> summarize(experiment_name=first(X.experiment_name),
                              noise=first(X.noise),
                              sigma=first(X.sigma),
                              top_1_acc_train=first(X.top_1_acc_train),
                              top_1_acc_pred=first(X.top_1_acc_pred))

    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    sns.lineplot(x="sigma", y="top_1_acc_train", hue="noise", markers=True, 
                 dashes=False, style="noise", data=grouped)
    plt.xlabel("$\sigma$")
    plt.ylabel("Top-1 training accuracy")
    plt.ylim((0, 1))
    plt.subplot(2, 1, 2)
    sns.lineplot(x="sigma", y="top_1_acc_pred", hue="noise", markers=True, 
                 dashes=False, style="noise", data=grouped)
    plt.xlabel("$\sigma$")
    plt.ylabel("Top-1 testing accuracy")
    plt.ylim((0, 1))
    plt.tight_layout()
    plt.show()

    selected = df >> mask(X.noise != "Clean") 
    sns.relplot(x="eps", y="top_1_acc_cert", hue="noise", kind="line", col="sigma",
                col_wrap=2, data=selected, height=2, aspect=1.5)
    plt.ylim((0, 1))
    plt.show()

    selected = df >> mask(X.noise != "Clean")
    sns.relplot(x="sigma", y="eps", hue="top_1_acc_cert", kind="scatter", row="noise", 
                data=selected, height=3, aspect=1.5, legend=False, sizes=(5,), palette="viridis")
    plt.tight_layout()
    plt.show()

