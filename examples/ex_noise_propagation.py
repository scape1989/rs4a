"""
Randomly initialize a base model and track how the noise propagates through the network.

Measures L2 norm and Linf norm.
"""
import torch
import torch.nn.functional as F
import pandas as pd
import itertools
import seaborn as sns
from argparse import ArgumentParser
from collections import defaultdict
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from src.attacks import *
from src.noises import *
from src.models import *
from src.datasets import get_dataset
from src.utils import get_trailing_number


def get_final_layer(model, x):
    out = model.model.conv1(x)
    out = model.model.block1(out)
    out = model.model.block2(out)
    out = model.model.block3(out)
    out = model.model.relu(model.model.bn1(out))
    out = F.avg_pool2d(out, 8)
    return out.view(-1, model.model.nChannels)


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--device", default="cpu", type=str)
    argparser.add_argument("--batch-size", default=4, type=int),
    argparser.add_argument("--sample-size", default=64, type=int)
    argparser.add_argument("--dataset", default="cifar", type=str)
    argparser.add_argument("--model", default="ResNet", type=str)
    args = argparser.parse_args()

    test_dataset = get_dataset(args.dataset, "test")
    test_subset = Subset(test_dataset, list(range(0, len(test_dataset), 10)))
    test_loader = DataLoader(test_subset, shuffle=False, batch_size=args.batch_size,
                             num_workers=4)

    model = eval(args.model)(dataset=args.dataset, device=args.device)
    model.eval()

    noises = ["GaussianNoise", "UniformNoise", "LaplaceNoise"]
    sigmas = (0.05, 0.25, 0.5, 1.0, 1.25)
    results = defaultdict(list)

    for noise_str, sigma in itertools.product(noises, sigmas):

        for i, (x, y) in tqdm(enumerate(test_loader), total=len(test_loader)):

            noise = eval(noise_str)(sigma=sigma, device=args.device, p=2,
                                    dim=get_dim(args.dataset))

            x, y = x.to(args.device), y.to(args.device)
            lower, upper = i * args.batch_size, (i + 1) * args.batch_size

            rep_true = get_final_layer(model, x)

            v = x.unsqueeze(1).expand((args.batch_size, args.sample_size, 3, 32, 32))
            rep_noisy = get_final_layer(model, noise.sample(v.reshape(args.batch_size, -1)).reshape(torch.Size([-1]) + v.shape[2:]))
            rep_noisy = rep_noisy.reshape(args.batch_size, args.sample_size, -1)
            diffs = rep_noisy - rep_true.unsqueeze(1)

            l2_mean = torch.norm(diffs, dim=2).mean(dim=1).data
            linf_mean = torch.norm(diffs, dim=2, p=float("inf")).mean(dim=1).data
            results["diffs_l2_mean"] += l2_mean.cpu().tolist()
            results["diffs_linf_mean"] += linf_mean.cpu().tolist()
            results["noise"] += [noise_str] * args.batch_size
            results["sigma"] += [sigma] * args.batch_size

    results = pd.DataFrame(results)
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    sns.lineplot(x="sigma", y="diffs_l2_mean", hue="noise", style="noise", data=results)
    plt.subplot(2, 1, 2)
    sns.lineplot(x="sigma", y="diffs_linf_mean", hue="noise", style="noise", data=results)
    plt.tight_layout()
    plt.show()



