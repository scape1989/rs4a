import numpy as np
import pandas as pd
import matplotlib as mpl
from argparse import ArgumentParser
from collections import defaultdict
from matplotlib import pyplot as plt
from statsmodels.stats.proportion import proportion_confint


experiments = [
    "cifar", "cifar_uniform", "cifar_laplace", "cifar_expinf"
]

if __name__ == "__main__":


    label_names = ["plane", "car", "bird", "cat", "deer", "dog", "frog",
                   "horse", "ship", "truck"]

    aggregated_results = defaultdict(list)
    axis = np.linspace(0, 2.5, 50)

    for experiment_name in experiments:

        save_path = f"ckpts/{experiment_name}"
        results = {}

        for k in ("preds", "preds_adv", "preds_smooth", "imgs", "imgs_adv", 
                  "labels", "radius_smooth"):
            results[k] = np.load(f"{save_path}/{k}.npy")

        top_1_preds = np.argmax(results["preds"], axis=1)
        top_1_preds_adv = np.argmax(results["preds_adv"], axis=1)
        top_1_preds_smooth = np.argmax(results["preds_smooth"], axis=1)

        lower, _ = proportion_confint(np.max(results["preds_smooth"], axis=1) * 
                                      1024, 1024, alpha=0.001, method="beta")
        top_1_preds_smooth[lower < 0.5] = -1

        top_1_acc = np.mean(top_1_preds == results["labels"])
        top_1_acc_adv = np.mean(top_1_preds_adv == results["labels"])
        top_1_acc_smooth = np.mean(top_1_preds_smooth == results["labels"])

        aggregated_results["experiment_name"].append(experiment_name)
        aggregated_results["top_1_acc"].append(top_1_acc)
        aggregated_results["top_1_acc_adv"].append(top_1_acc_adv)
        aggregated_results["top_1_acc_smooth"].append(top_1_acc_smooth)

        accs = np.zeros_like(axis)
        sel = results["radius_smooth"][:,np.newaxis] > axis[np.newaxis,:]
        for i in range(len(axis)):
            accs[i] = np.sum(top_1_preds[sel[:,i]] == results["labels"][sel[:,i]]) / 10000
        aggregated_results["accs"].append(accs.tostring())

    df = pd.DataFrame(aggregated_results)
    print(df.drop(["accs"], axis=1))

    plt.figure(figsize=(8, 3))
    mpl.style.use("seaborn-dark-palette")

    for (i, row) in df.iterrows():
        if i == 0:
            continue
        plt.plot(axis, np.fromstring(row["accs"]),
                 label=row["experiment_name"].replace("_", "\_"))

    plt.axhline(df["top_1_acc"][0], label="clean", 
                color="grey", linestyle="--")
    plt.legend()
    plt.ylabel("Certified top-1 accuracy")
    plt.xlabel("$\ell_1$ radius")
    plt.tight_layout()
    plt.ylim((0, 1))
    plt.savefig("./ckpts/curve.png")
#    plt.show()
