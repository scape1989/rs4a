import torch
from argparse import ArgumentParser
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
from src.noises import *


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--idx", type=int, default=1)
    argparser.add_argument("--sigma", type=float, default=0.1)
    argparser.add_argument("--n-samples", type=int, default=4)
    args = argparser.parse_args()
    
    train_dataset = datasets.CIFAR10("./data/cifar_10", train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor()
                                      ]))

    plt.figure(figsize=(3 * args.n_samples, 9))

    x, y = train_dataset[args.idx]
    x = x.unsqueeze(0)
    
    for i in range(args.n_samples):

        noise = LaplaceNoise(sigma=args.sigma + 0.1 * i, device="cpu")
        sample = (x + noise.sample(x.shape)).clamp(0, 1)
        plt.subplot(4, args.n_samples, i + 1)
        plt.imshow(sample[0].numpy().transpose(1, 2, 0))
        plt.axis("off")
        plt.title(f"Laplace {args.sigma + 0.1 * i:.1f}")

    for i in range(args.n_samples):

        noise = GaussianNoise(sigma=args.sigma + 0.1 * i, device="cpu")
        sample = (x + noise.sample(x.shape)).clamp(0, 1)
        plt.subplot(4, args.n_samples, i + 1 + args.n_samples)
        plt.imshow(sample[0].numpy().transpose(1, 2, 0))
        plt.axis("off")
        plt.title(f"Gaussian {args.sigma + 0.1 * i:.1f}")

    for i in range(args.n_samples):

        noise = Exp1Noise(sigma=args.sigma + 0.1 * i, device="cpu", k=100)
        sample = (x + noise.sample(x.shape)).clamp(0, 1)
        plt.subplot(4, args.n_samples, i + 1 + args.n_samples * 2)
        plt.imshow(sample[0].numpy().transpose(1, 2, 0))
        plt.axis("off")
        plt.title(f"ExpInf {args.sigma + 0.1 * i:.1f}")

    for i in range(args.n_samples):

        noise = UniformNoise(sigma=args.sigma + 0.1 * i, device="cpu", k=1)
        sample = (x + noise.sample(x.shape)).clamp(0, 1)
        plt.subplot(4, args.n_samples, i + 1 + args.n_samples * 3)
        plt.imshow(sample[0].numpy().transpose(1, 2, 0))
        plt.axis("off")
        plt.title(f"Uniform {args.sigma + 0.1 * i:.1f}")

    plt.show()

