import numpy as np
import os
import torch
import torch.nn as nn
from argparse import ArgumentParser
from pathlib import Path
from torch.utils.data import DataLoader
from src.models import *
from src.attacks import *
from src.smooth import *
from src.noises import *
from src.datasets import get_dataset
from tqdm import tqdm


if __name__ == "__main__":
    
    argparser = ArgumentParser()
    argparser.add_argument("--device", default="cuda:0", type=str)
    argparser.add_argument("--batch-size", default=2, type=int)
    argparser.add_argument("--num-workers", default=os.cpu_count(), type=int)
    argparser.add_argument("--sample-size-pred", default=64, type=int)
    argparser.add_argument("--sample-size-cert", default=100000, type=int)
    argparser.add_argument("--sigma", default=0.0, type=float)
    argparser.add_argument("--noise", default="Clean", type=str)
    argparser.add_argument("--k", default=1, type=int)
    argparser.add_argument("--p", default=1, type=int)
    argparser.add_argument("--experiment-name", default="cifar", type=str)
    argparser.add_argument("--dataset", default="cifar", type=str)
    argparser.add_argument("--model", default="ResNet", type=str)
    argparser.add_argument('--output-dir', type=str, default=os.getenv("PT_OUTPUT_DIR"))
    args = argparser.parse_args()

    test_dataset = get_dataset(args.dataset, "test")
    test_loader = DataLoader(test_dataset, shuffle=False, 
                             batch_size=args.batch_size, 
                             num_workers=args.num_workers)

    save_path = f"{args.output_dir}/{args.experiment_name}/model_ckpt.torch"
    model = eval(args.model)(dataset=args.dataset, device=args.device)
    model.load_state_dict(torch.load(save_path))
    model.eval()
    noise = eval(args.noise)(**args.__dict__)

    results = {
        "preds_smooth": np.zeros((len(test_dataset), 10)),
        "labels": np.zeros(len(test_dataset)),
        "radius_smooth": np.zeros(len(test_dataset)),
    }

    for i, (x, y) in tqdm(enumerate(test_loader), total=len(test_loader)):

        x, y = x.to(args.device), y.to(args.device)
        preds = model.forecast(model.forward(x))
        preds_smooth = smooth_predict_hard(model, x, noise, args.sample_size_pred)
        top_cats = preds_smooth.probs.argmax(dim=1)
        radii = certify_smoothed(model, x, top_cats, 0.001, noise, args.sample_size_cert)

        lower, upper = i * args.batch_size, (i + 1) * args.batch_size
        results["preds_smooth"][lower:upper,:] = preds_smooth.probs.data.cpu().numpy()
        results["labels"][lower:upper] = y.data.cpu().numpy()
        results["radius_smooth"][lower:upper] = radii.cpu().numpy()

    save_path = f"{args.output_dir}/{args.experiment_name}"
    for k, v in results.items():
        np.save(f"{save_path}/{k}.npy", v)

