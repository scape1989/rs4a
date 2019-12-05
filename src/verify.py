import torch
import os
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.attacks import *
from src.noises import *
from src.models import *
from src.datasets import get_dataset


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--device", default="cuda:0", type=str)
    argparser.add_argument("--batch-size", default=16, type=int),
    argparser.add_argument("--num-workers", default=os.cpu_count(), type=int)
    argparser.add_argument("--sample-size-pred", default=64, type=int)
    argparser.add_argument("--sigma", default=0.0, type=float)
    argparser.add_argument("--noise", default="Clean", type=str)
    argparser.add_argument("--k", default=1, type=int)
    argparser.add_argument("--p", default=2, type=int)
    argparser.add_argument("--experiment-name", default="cifar", type=str)
    argparser.add_argument("--dataset", default="cifar", type=str)
    argparser.add_argument("--model", default="ResNet", type=str)
    argparser.add_argument("--output-dir", type=str, default=os.getenv("PT_OUTPUT_DIR"))
    args = argparser.parse_args()

    test_dataset = get_dataset(args.dataset, "test")
    test_dataset.data = test_dataset.data[:200] # todo: fix
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size,
                             num_workers=args.num_workers)

    save_path = f"{args.output_dir}/{args.experiment_name}/model_ckpt.torch"
    model = eval(args.model)(dataset=args.dataset, device=args.device)
    model.load_state_dict(torch.load(save_path))
    model.eval()
    noise = eval(args.noise)(**args.__dict__)

    eps_range = (8.0, 6.0, 4.0, 2.0, 1.0)

    results = {f"preds_adv_{eps}": np.zeros((len(test_dataset), 10)) for eps in eps_range}

    for i, (x, y) in tqdm(enumerate(test_loader), total=len(test_loader)):

        x, y = x.to(args.device), y.to(args.device)
        lower, upper = i * args.batch_size, (i + 1) * args.batch_size

        for eps in eps_range:
            x_adv = pgd_attack_smooth(model, x, y, eps=eps, noise=noise, sample_size=128,
                                      steps=20, p=1, clamp=(0, 1))
            preds_adv = smooth_predict_hard(model, x_adv, noise, args.sample_size_pred)
            results[f"preds_adv_{eps}"][lower:upper, :] = preds_adv.probs.data.cpu().numpy()
            assert ((x - x_adv).reshape(x.shape[0], -1).norm(dim=1, p=1) <= eps + 1e-3).all()

    save_path = f"{args.output_dir}/{args.experiment_name}"
    for k, v in results.items():
        np.save(f"{save_path}/{k}.npy", v)

