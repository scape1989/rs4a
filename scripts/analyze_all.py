import numpy as np
import pandas as pd
import pickle
import matplotlib as mpl
from argparse import ArgumentParser
from collections import defaultdict
from dfply import *
from matplotlib import pyplot as plt


experiments = [
    "cifar", "cifar_laplace", "cifar_expinf", "cifar_uniform",
]

if __name__ == "__main__":

    df = defaultdict(list)
    eps_range = np.linspace(0, 2.5, 50)

    for experiment_name in experiments:

        save_path = f"ckpts/{experiment_name}"
        args = pickle.load(open(f"ckpts/{experiment_name}/args.pkl", "rb"))
        results = {}

        for k in  ("preds_smooth", "labels", "radius_smooth", "acc_train"):
            results[k] = np.load(f"{save_path}/{k}.npy")

        top_1_preds_smooth = np.argmax(results["preds_smooth"], axis=1)

        for eps in eps_range:

            top_1_acc = ((results["radius_smooth"] >= eps) & \
                         (top_1_preds_smooth == results["labels"])).mean()
            df["experiment_name"].append(experiment_name)
            df["sigma"].append(args.sigma)
            df["noise"].append(args.noise)
            df["acc_train"].append(results["acc_train"][0])
            df["eps"].append(eps)
            df["top_1_acc"].append(top_1_acc)

    df = pd.DataFrame(df)
    df.to_csv("./ckpts/results.csv", index=False)

    plt.figure(figsize=(8, 3))
    mpl.style.use("seaborn-dark-palette")

    grouped = df >> group_by(X.experiment_name) \
                 >> summarize(experiment_name=first(X.experiment_name),
                              sigma=first(X.sigma),
                              acc_train=first(X.acc_train))

    for experiment_name in experiments:

        if experiment_name == "cifar":
            continue

        selected = df >> mask(X.experiment_name == experiment_name)
        plt.plot(selected.eps, selected.top_1_acc,
                 label=first(selected.experiment_name).replace("_", "_"))

    plt.axhline(df["top_1_acc"][0], label="clean", 
                color="grey", linestyle="--")
    plt.legend()
    plt.ylabel("Certified top-1 accuracy")
    plt.xlabel("$\ell_1$ radius")
    plt.tight_layout()
    plt.ylim((0, 1))
    plt.show()

