import torch
from argparse import ArgumentParser
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
from src.noises import *
from src.datasets import get_dataset


def plot_image(x, dataset):
    if dataset == "cifar":
        plt.imshow(x[0].numpy().transpose(1, 2, 0))
    if dataset == "mnist":
        plt.imshow(x[0,0].numpy())


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--idx", type=int, default=1)
    argparser.add_argument("--sigma", type=float, default=0.1)
    argparser.add_argument("--n-samples", type=int, default=4)
    argparser.add_argument("--dataset", type=str, default="cifar")
    args = argparser.parse_args()

    dataset = get_dataset(args.dataset, "test")

    plt.figure(figsize=(3 * args.n_samples, 9))

    x, y = dataset[args.idx]
    x = x.unsqueeze(0)

    noise = MaskGaussianNoisePatch(sigma=0.0, device="cpu", dim=3*32*32, p=2, k=2)
    sample = noise.sample(x).clamp(0, 1)
    plt.imshow(sample[0].numpy().transpose((1,2,0)))
    breakpoint()

    for i in range(args.n_samples):

        noise = LaplaceNoise(sigma=args.sigma + 0.1 * i, device="cpu", p=2)
        sample = (x + noise.sample(x.shape)).clamp(0, 1)
        plt.subplot(4, args.n_samples, i + 1)
        plot_image(sample, args.dataset)
        plt.axis("off")
        plt.title(f"Laplace {args.sigma + 0.1 * i:.1f}")

    for i in range(args.n_samples):

        noise = GaussianNoise(sigma=args.sigma + 0.1 * i, device="cpu", p=2)
        sample = (x + noise.sample(x.shape)).clamp(0, 1)
        plt.subplot(4, args.n_samples, i + 1 + args.n_samples)
        plot_image(sample, args.dataset)
        plt.axis("off")
        plt.title(f"Gaussian {args.sigma + 0.1 * i:.1f}")

    for i in range(args.n_samples):

        noise = ExpInfNoise(sigma=args.sigma + 0.1 * i, device="cpu", k=1, p=2)
        sample = (x + noise.sample(x.shape)).clamp(0, 1)
        plt.subplot(4, args.n_samples, i + 1 + args.n_samples * 2)
        plot_image(sample, args.dataset)
        plt.axis("off")
        plt.title(f"ExpInf {args.sigma + 0.1 * i:.1f}")

    for i in range(args.n_samples):

        noise = UniformNoise(sigma=args.sigma + 0.1 * i, device="cpu", p=2)
        sample = (x + noise.sample(x.shape)).clamp(0, 1)
        plt.subplot(4, args.n_samples, i + 1 + args.n_samples * 3)
        plot_image(sample, args.dataset)
        plt.axis("off")
        plt.title(f"Uniform {args.sigma + 0.1 * i:.1f}")

    plt.show()

