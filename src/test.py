import numpy as np
import os
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
    argparser.add_argument("--batch-size", default=2, type=int)
    argparser.add_argument("--num-workers", default=os.cpu_count(), type=int)
    argparser.add_argument("--sample-size-pred", default=64, type=int)
    argparser.add_argument("--sample-size-cert", default=1024, type=int)
    argparser.add_argument("--sigma", default=0.25, type=float)
    argparser.add_argument("--noise", default="Clean", type=str)
    argparser.add_argument("--experiment-name", default="cifar", type=str)
    argparser.add_argument('--output-dir', type=str, 
                           default=os.getenv("PT_OUTPUT_DIR"))
    args = argparser.parse_args()

    test_dataset = datasets.CIFAR10("./data/cifar_10", train=False,
                                    download=True, 
                                    transform=transforms.ToTensor())

    test_loader = DataLoader(test_dataset, shuffle=False, 
                             batch_size=args.batch_size, 
                             num_workers=args.num_workers)

    save_path = f"{args.output_dir}/{args.experiment_name}/model_ckpt.torch"
    model = ResNet(dataset="cifar", device=args.device)
    model.load_state_dict(torch.load(save_path))
    model.eval()
    noise = eval(args.noise)(**args.__dict__)

    results = {
        "preds": np.zeros((len(test_dataset), 10)),
        "preds_smooth": np.zeros((len(test_dataset), 10)),
        "imgs": np.zeros((len(test_dataset), 3, 32, 32)),
        "labels": np.zeros(len(test_dataset)),
        "radius_smooth": np.zeros(len(test_dataset)),
    }

    for i, (x, y) in tqdm(enumerate(test_loader), total=len(test_loader)):

        x, y = x.to(args.device), y.to(args.device)
        preds = model.forecast(model.forward(x))
        preds_smooth = smooth_predict_hard(model, x, noise, 
                                           sample_size=args.sample_size_pred)
        top_cats = preds_smooth.probs.argmax(dim=1)
        radii = certify_smoothed(model, x, top_cats, alpha=0.001, noise=noise, 
                                 sample_size=args.sample_size_cert)

        lower, upper = i * args.batch_size, (i + 1) * args.batch_size
        results["preds"][lower:upper,:] = preds.probs.data.cpu().numpy()
        results["preds_smooth"][lower:upper,:] = preds_smooth.probs.data.cpu().numpy()
        results["labels"][lower:upper] = y.data.cpu().numpy()
        results["imgs"][lower:upper,:,:,:] = x.data.cpu().numpy()
        results["radius_smooth"][lower:upper] = radii.cpu().numpy()

    save_path = f"{args.output_dir}/{args.experiment_name}"
    for k, v in results.items():
        np.save(f"{save_path}/{k}.npy", v)

