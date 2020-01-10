import torch
import torch.nn.functional as F
import os
from argparse import ArgumentParser
from collections import defaultdict
from torch.utils.data import DataLoader
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
    argparser.add_argument("--device", default="cuda:0", type=str)
    argparser.add_argument("--batch-size", default=4, type=int),
    argparser.add_argument("--num-workers", default=os.cpu_count(), type=int)
    argparser.add_argument("--sample-size", default=64, type=int)
    argparser.add_argument("--noise", default="Clean", type=str)
    argparser.add_argument("--p", default=2, type=int)
    argparser.add_argument("--experiment-name", default="cifar", type=str)
    argparser.add_argument("--dataset", default="cifar", type=str)
    argparser.add_argument("--model", default="ResNet", type=str)
    argparser.add_argument("--output-dir", type=str, default=os.getenv("PT_OUTPUT_DIR"))
    args = argparser.parse_args()

    test_dataset = get_dataset(args.dataset, "test")
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, # todo: fix
                             num_workers=args.num_workers)

    save_path = f"{args.output_dir}/{args.experiment_name}/model_ckpt.torch"
    true_model = eval(args.model)(dataset=args.dataset, device=args.device)
    true_model.load_state_dict(torch.load(save_path))
    true_model.eval()

    l2_norms = np.zeros(len(test_loader))

    sigma_range = (0.25, 0.5, 1.0, 1.25)
    results = defaultdict(lambda: np.zeros(len(test_dataset)))

    for sigma in sigma_range:

        noise = eval(args.noise)(sigma=sigma, device=args.device, p=args.p,
                                 dim=get_dim(args.dataset))

        experiment_name = f"cifar_{args.noise}_{sigma}"
        save_path = f"{args.output_dir}/{experiment_name}/model_ckpt.torch"
        model = eval(args.model)(dataset=args.dataset, device=args.device)
        model.load_state_dict(torch.load(save_path))
        model.eval()

        for i, (x, y) in tqdm(enumerate(test_loader), total=len(test_loader)):

            x, y = x.to(args.device), y.to(args.device)
            lower, upper = i * args.batch_size, (i + 1) * args.batch_size

            rep_true = get_final_layer(true_model, x)

            v = x.unsqueeze(1).expand((args.batch_size, args.sample_size, 3, 32, 32))
            v = v.reshape((-1, 3, 32, 32))
            rep_noisy = get_final_layer(model, noise.sample(v))
            rep_noisy = rep_noisy.reshape(args.batch_size, -1, rep_true.shape[-1])
            diffs = rep_noisy - rep_true.unsqueeze(1)

            means = torch.norm(diffs, dim=2).mean(dim=1).data
            sds = torch.norm(diffs, dim=2).std(dim=1).data
            results[f"diffs_l2_mean_{args.noise}_{sigma}"][lower:upper] = means.cpu().numpy()
            results[f"diffs_l2_sd_{args.noise}_{sigma}"][lower:upper] = sds.cpu().numpy()

    save_path = f"{args.output_dir}/{args.experiment_name}"
    for k, v in results.items():
        np.save(f"{save_path}/{k}.npy", v)

