import numpy as np
import pathlib
import os
import sys
import torch
import torch.nn as nn
from argparse import ArgumentParser
from torchnet import meter
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from src.models import *
from src.attacks import *
from src.smooth import *
from src.noises import *
from src.datasets import *
from src.utils import get_trailing_number


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--device", default="cuda", type=str)
    argparser.add_argument("--batch-size", default=2, type=int)
    argparser.add_argument("--num-workers", default=min(os.cpu_count(), 8), type=int)
    argparser.add_argument("--sample-size-pred", default=64, type=int)
    argparser.add_argument("--sample-size-cert", default=1024, type=int)
    argparser.add_argument("--noise-batch-size", default=512, type=int)
    argparser.add_argument("--sigma", default=0.0, type=float)
    argparser.add_argument("--noise", default="Clean", type=str)
    argparser.add_argument("--p", default=2, type=int)
    argparser.add_argument("--dataset-skip", default=1, type=int)
    argparser.add_argument("--experiment-name", default="cifar", type=str)
    argparser.add_argument("--dataset", default="cifar", type=str)
    argparser.add_argument("--model", default="ResNet", type=str)
    argparser.add_argument("--rotate", action="store_true")
    argparser.add_argument("--test-set-only", action="store_true")
    argparser.add_argument("--output-dir", type=str, default=os.getenv("PT_OUTPUT_DIR"))
    argparser.add_argument("--save-path", type=str, default=None)
    args = argparser.parse_args()

    test_dataset = get_dataset(args.dataset, "test")
    test_dataset = Subset(test_dataset, list(range(0, len(test_dataset), args.dataset_skip)))
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size,
                             num_workers=args.num_workers)

    if not args.save_path:
        save_path = f"{args.output_dir}/{args.experiment_name}/model_ckpt.torch"
    else:
        save_path = args.save_path

    model = eval(args.model)(dataset=args.dataset, device=args.device)
    saved_dict = torch.load(save_path)
    try: # janky change for backwards compatibility
        model.load_state_dict(saved_dict)
    except:
        norm_dim = get_normalization_shape(args.dataset)
        norm_stats = get_normalization_stats(args.dataset)
        mu = torch.tensor(norm_stats["mu"], device=args.device).reshape(norm_dim)
        log_sig = torch.log(torch.tensor(norm_stats["sigma"], device=args.device).reshape(norm_dim))
        try:
            saved_dict["norm.mu"] = mu
            saved_dict["norm.log_sig"] = log_sig
            model.load_state_dict(saved_dict)
        except:
            del saved_dict["norm.mu"]
            del saved_dict["norm.log_sig"]
            saved_dict["norm.module.mu"] = mu
            saved_dict["norm.module.log_sig"] = log_sig
            model.load_state_dict(saved_dict)
    model.eval()

    k = get_trailing_number(args.noise)
    if k:
        noise = eval(args.noise[:-len(str(k))])(sigma=args.sigma, device=args.device, p=args.p,
                                                dim=get_dim(args.dataset), k=k)
    else:
        noise = eval(args.noise)(sigma=args.sigma, device=args.device, p=args.p,
                                 dim=get_dim(args.dataset))

    results = {
        "preds_smooth": np.zeros((len(test_dataset), get_num_labels(args.dataset))),
        "labels": np.zeros(len(test_dataset)),
        "prob_lower_bound": np.zeros(len(test_dataset)),
        "radius_smooth": np.zeros(len(test_dataset)),
        "preds_nll": np.zeros(len(test_dataset))
    }

    if args.rotate:
        rotate_noise = RotationNoise(0.0, args.device, dim=get_dim(args.dataset), p=None)

    for i, (x, y) in tqdm(enumerate(test_loader), total=len(test_loader)):

        x, y = x.to(args.device), y.to(args.device)
        x = rotate_noise.sample(x) if args.rotate else x

        preds_smooth = smooth_predict_hard(model, x, noise, args.sample_size_pred,
                                           noise_batch_size=args.noise_batch_size)
        top_cats = preds_smooth.probs.argmax(dim=1)
        p_a, radii = certify_smoothed(model, x, top_cats, 0.001, noise, args.sample_size_cert,
                                      noise_batch_size=args.noise_batch_size)

        lower, upper = i * args.batch_size, (i + 1) * args.batch_size
        results["preds_smooth"][lower:upper, :] = preds_smooth.probs.data.cpu().numpy()
        results["labels"][lower:upper] = y.data.cpu().numpy()
        results["prob_lower_bound"][lower:upper] = p_a.cpu().numpy()
        results["radius_smooth"][lower:upper] = radii.cpu().numpy()
        results["preds_nll"][lower:upper] = -preds_smooth.log_prob(y).cpu().numpy()

    save_path = f"{args.output_dir}/{args.experiment_name}"
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    for k, v in results.items():
        np.save(f"{save_path}/{k}.npy", v)

    if args.test_set_only:
        sys.exit()

    train_dataset = get_dataset(args.dataset, "train")
    train_loader = DataLoader(train_dataset, shuffle=False, batch_size=args.batch_size,
                              num_workers=args.num_workers)
    acc_meter = meter.AverageValueMeter()

    for x, y in tqdm(train_loader):

        x, y = x.to(args.device), y.to(args.device)
        x = rotate_noise.sample(x) if args.rotate else x

        preds_smooth = smooth_predict_hard(model, x, noise, args.sample_size_pred,
                                           args.noise_batch_size)
        top_cats = preds_smooth.probs.argmax(dim=1)
        acc_meter.add(torch.sum(top_cats == y).cpu().data.numpy(), n=len(x))

    print("Training accuracy: ", acc_meter.value())
    save_path = f"{args.output_dir}/{args.experiment_name}"
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    np.save(f"{save_path}/acc_train.npy",  acc_meter.value())

