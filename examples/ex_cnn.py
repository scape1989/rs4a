import numpy as np
import pandas as pd
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
from argparse import ArgumentParser
from collections import defaultdict
from matplotlib import pyplot as plt
from src.noises import *
from src.datasets import *


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--batch-size", type=int, default=200)
    args = argparser.parse_args()

    x = torch.randn((args.batch_size, 3, 32, 32))

    conv_layer = nn.Sequential(
        nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(),
        nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU()
    )
    mlp_layer = nn.Sequential(
        nn.Linear(3 * 32 * 32, 4096),
        nn.ReLU(),
        nn.Linear(4096, 1024),
        nn.ReLU()
    )

    y_conv = conv_layer(x)
    y_mlp = mlp_layer(x.reshape(args.batch_size, -1))

    print("Conv:", y_conv.shape)
    print("MLP:", y_mlp.shape)

    df = defaultdict(list)

    for noise, sigma in itertools.product(("GaussianNoise", "UniformNoise", "LaplaceNoise"),
                                          (1.0, 2.0, 3.0, 4.0)):

        x_noisy = eval(noise)(sigma=sigma, device="cpu", dim=3 * 32 * 32, p=2).sample(x)

        diff_conv = conv_layer(x_noisy).reshape(args.batch_size, -1) - y_conv.reshape(args.batch_size, -1)
        diff_mlp = mlp_layer(x_noisy.reshape(args.batch_size, -1)) - y_mlp

        norm_conv = torch.norm(diff_conv, dim=1) / diff_conv.shape[1]
        norm_mlp = torch.norm(diff_mlp, dim=1) / diff_mlp.shape[1]

        df["sigma"].append(sigma)
        df["noise"].append(noise)
        df["diff_l2_mean"].append(norm_conv.mean().data.numpy())
        df["diff_l2_std"].append(norm_conv.std().data.numpy())
        df["model"].append("conv")

        df["sigma"].append(sigma)
        df["noise"].append(noise)
        df["diff_l2_mean"].append(norm_mlp.mean().data.numpy())
        df["diff_l2_std"].append(norm_mlp.std().data.numpy())
        df["model"].append("mlp")

    df = pd.DataFrame(df)
    df["diff_l2_mean"] = df["diff_l2_mean"].astype("float")
    df["diff_l2_std"] = df["diff_l2_std"].astype("float")
    sns.relplot(x="sigma", y="diff_l2_mean", hue="noise", kind="line", col="model",
                col_wrap=1, data=df, height=3, aspect=1.5)
    plt.show()
