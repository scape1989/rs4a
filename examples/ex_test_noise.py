from argparse import ArgumentParser
from src.noises import *


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--noise", type=str, default="GaussianNoise")
    argparser.add_argument("--dim", type=int, default=40)
    argparser.add_argument("--batch-size", type=int, default=2000)
    argparser.add_argument("--sigma", type=float, default=0.25)
    args = argparser.parse_args()

    if args.noise[-1].isdigit():
        k = int(args.noise[-1])
        noise = eval(args.noise[:-1])(sigma=args.sigma, device="cpu", p=2, k=k, dim=args.dim)
    else:
        noise = eval(args.noise)(sigma=args.sigma, device="cpu", p=2)

    rvs = noise.sample((args.batch_size, args.dim))
    print(rvs.norm(p=2, dim=1).pow(2).mean())

    if args.noise[-1].isdigit():
        k = int(args.noise[-1])
        noise = eval(args.noise[:-1])(sigma=args.sigma, device="cpu", p=1, k=k, dim=args.dim)
    else:
        noise = eval(args.noise)(sigma=args.sigma, device="cpu", p=1)

    rvs = noise.sample((args.batch_size, args.dim))
    print(rvs.norm(p=1, dim=1).mean())
