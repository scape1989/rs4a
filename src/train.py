import logging
import pathlib
import pickle
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from argparse import ArgumentParser
from torchnet import meter
from torch.utils.data import DataLoader
from src.models import *
from src.noises import *
from src.smooth import *
from src.attacks import pgd_attack_smooth
from src.datasets import get_dataset


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--device", default="cuda:0", type=str)
    argparser.add_argument("--lr", default=0.1, type=float)
    argparser.add_argument("--batch-size", default=64, type=int)
    argparser.add_argument("--num-workers", default=os.cpu_count(), type=int)
    argparser.add_argument("--num-epochs", default=120, type=int)
    argparser.add_argument("--print-every", default=20, type=int)
    argparser.add_argument("--save-every", default=50, type=int)
    argparser.add_argument("--experiment-name", default="cifar", type=str)
    argparser.add_argument("--noise", default="Clean", type=str)
    argparser.add_argument("--sigma", default=0.0, type=float)
    argparser.add_argument("--eps", default=1.0, type=float)
    argparser.add_argument("--p", default=1, type=int)
    argparser.add_argument("--k", default=1.0, type=float)
    argparser.add_argument("--model", default="ResNet", type=str)
    argparser.add_argument("--dataset", default="cifar", type=str)
    argparser.add_argument("--adversarial", action="store_true")
    argparser.add_argument("--direct", action="store_true")
    argparser.add_argument('--output-dir', type=str, default=os.getenv("PT_OUTPUT_DIR"))
    args = argparser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    model = eval(args.model)(dataset=args.dataset, device=args.device)
    model.train()

    train_loader = DataLoader(get_dataset(args.dataset, "train"), shuffle=True,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers)

    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=0.9,
                          weight_decay=1e-4,
                          nesterov=True)
    annealer = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs)

    loss_meter = meter.AverageValueMeter()
    time_meter = meter.TimeMeter(unit=False)
    noise = eval(args.noise)(**args.__dict__)

    train_losses = []

    for epoch in range(args.num_epochs):

        for i, (x, y) in enumerate(train_loader):

            x, y = x.to(args.device), y.to(args.device)

            if args.adversarial and epoch > args.num_epochs // 2:
                x = pgd_attack_smooth(model, x, y, args.eps, noise, sample_size=2, p=args.p)
            elif not args.direct:
                x = x + noise.sample(x.shape)

                # random rotation matrix
#                W, _ = sp.linalg.qr(np.random.randn(784, 784))
#                delta = noise.sample(x.shape)
#                delta = delta.reshape(x.shape[0], -1)
#                delta = delta @ torch.tensor(W, device=args.device, dtype=torch.float)
#                x = x + delta.view(x.shape)

            optimizer.zero_grad()
            if args.direct:
                loss = -smooth_predict_soft(model, x, noise, sample_size=32).log_prob(y).mean()
            else:
                loss = model.loss(x, y).mean()
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.cpu().data.numpy(), n=1)

            if i % args.print_every == 0:
                logger.info(f"Epoch: {epoch}\t" +
                            f"Itr: {i} / {len(train_loader)}\t" +
                            f"Loss: {loss_meter.value()[0]:.2f}\t"
                            f"Mins: {(time_meter.value() / 60):.2f}\t" +
                            f"Experiment: {args.experiment_name}")
                train_losses.append(loss_meter.value()[0])
                loss_meter.reset()

        if epoch % args.save_every == 0:
            save_path = f"{args.output_dir}/{args.experiment_name}/{epoch}/"
            pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), f"{save_path}/model_ckpt.torch")

        annealer.step()

    save_path = f"{args.output_dir}/{args.experiment_name}/model_ckpt.torch"
    torch.save(model.state_dict(), save_path)
    args_path = f"{args.output_dir}/{args.experiment_name}/args.pkl"
    pickle.dump(args, open(args_path, "wb"))

    model.eval()
    acc_meter = meter.AverageValueMeter()

    for x, y in train_loader:

         x, y = x.to(args.device), y.to(args.device)
         preds_smooth = smooth_predict_hard(model, x, noise, sample_size=32)
         top_cats = preds_smooth.probs.argmax(dim=1)
         acc_meter.add(torch.sum(top_cats == y).cpu().data.numpy(), n=len(x))

    print("Training accuracy: ", acc_meter.value())
    save_path = f"{args.output_dir}/{args.experiment_name}/acc_train.npy"
    np.save(save_path,  acc_meter.value())

    save_path = f"{args.output_dir}/{args.experiment_name}/losses_train.npy"
    np.save(save_path, np.array(train_losses))

