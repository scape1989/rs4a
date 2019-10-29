import numpy as np
import torch
import torch.nn as nn
from argparse import ArgumentParser
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.models import ResNet
from src.attacks import *
from src.smooth import *
from src.noises import *
from tqdm import tqdm


if __name__ == "__main__":
    
    argparser = ArgumentParser()
    argparser.add_argument("--device", default="cuda:0", type=str)
    argparser.add_argument("--batch-size", default=64, type=int)
    argparser.add_argument("--num-workers", default=4, type=int)
    argparser.add_argument("--sigma", default=0.25, type=float)
    argparser.add_argument("--norm", default=1, type=int)
    argparser.add_argument("--noise", default="Clean", type=str)
    argparser.add_argument("--experiment-name", default="cifar", type=str)
    args = argparser.parse_args()

    test_dataset = datasets.CIFAR10("./data/cifar_10", train=False,
                                    download=True, 
                                    transform=transforms.ToTensor())

    test_loader = DataLoader(test_dataset, shuffle=False, 
                             batch_size=args.batch_size, 
                             num_workers=args.num_workers)

    save_path = f"ckpts/{args.experiment_name}/model_ckpt.torch"
    model = ResNet(num_classes=10, device=args.device)
    model.load_state_dict(torch.load(save_path))
    noise = eval(args.noise)(args.sigma, args.device)

    results = {
        "preds": np.zeros((len(test_dataset), 10)),
        "preds_adv": np.zeros((len(test_dataset), 10)),
        "preds_smooth": np.zeros((len(test_dataset), 10)),
        "imgs": np.zeros((len(test_dataset), 3, 32, 32)),
        "imgs_adv": np.zeros((len(test_dataset), 3, 32, 32)),
        "labels": np.zeros(len(test_dataset)),
        "embeds": np.zeros((len(test_dataset), 256)),
        "embeds_adv": np.zeros((len(test_dataset), 256)),
        "radius_smooth": np.zeros(len(test_dataset)),
    }

    for i, (x, y) in tqdm(enumerate(test_loader), total=len(test_loader)):

        x, y = x.to(args.device), y.to(args.device)
        preds = model.forward(x)
        x_adv = pgd_attack(model, x, y, args.sigma, p=args.norm)
        preds_adv = model.forward(x_adv)
        #embeds = model.embed(x).data
        #embeds_adv = model.embed(x_adv).data
        x.requires_grad = False
        preds_smooth = smooth_predict_hard(model, x, noise, sample_size=32)
        cats = preds_smooth.probs.argmax(dim=1)
        radii = certify_smoothed(model, x, cats, alpha=0.001, noise=noise, 
                                 sample_size=32)

        lower, upper = i * args.batch_size, (i + 1) * args.batch_size
        results["preds"][lower:upper,:] = preds.probs.data.cpu().numpy()
        results["preds_adv"][lower:upper,:] = preds_adv.probs.data.cpu().numpy()
        results["preds_smooth"][lower:upper,:] = preds_smooth.probs.data.cpu().numpy()
        results["labels"][lower:upper] = y.data.cpu().numpy()
        results["imgs"][lower:upper,:,:,:] = x.data.cpu().numpy()
        results["imgs_adv"][lower:upper,:,:,:] = x_adv.data.cpu().numpy()
#        results["embeds"][lower:upper,:] = embeds.cpu().numpy()
#        results["embeds_adv"][lower:upper,:] = embeds_adv.cpu().numpy()
        results["radius_smooth"][lower:upper] = radii.cpu().numpy()

    save_path = f"ckpts/{args.experiment_name}"
    for k, v in results.items():
        np.save(f"{save_path}/{k}.npy", v)

