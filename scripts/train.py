import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
from argparse import ArgumentParser
from torchnet import meter
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.models import ResNet
from src.noises import *
from src.attacks import pgd_attack_smooth


if __name__ == "__main__":
    
    argparser = ArgumentParser()
    argparser.add_argument("--device", default="cuda:0", type=str)
    argparser.add_argument("--lr", default=0.1, type=float)
    argparser.add_argument("--batch-size", default=256, type=int)
    argparser.add_argument("--num-workers", default=os.cpu_count(), type=int)
    argparser.add_argument("--num-epochs", default=90, type=int)
    argparser.add_argument("--print-every", default=20, type=int)
    argparser.add_argument("--experiment-name", default="cifar", type=str)
    argparser.add_argument("--noise", default="Clean", type=str)
    argparser.add_argument("--sigma", default=0.25, type=float)
    argparser.add_argument("--eps", default=5.0, type=float)
    argparser.add_argument("--p", default=1, type=int)
    argparser.add_argument("--adversarial", action="store_true")
    args = argparser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    train_dataset = datasets.CIFAR10("./data/cifar_10", 
                                     train=True, 
                                     download=True, 
                                     transform=transforms.Compose([
                                         transforms.RandomCrop(32, padding=4),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor()
                                      ]))

    model = ResNet(dataset="cifar", device=args.device)
    
    train_loader = DataLoader(train_dataset, shuffle=True, 
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
    noise = eval(args.noise)(args.sigma, args.device)

    for epoch in range(args.num_epochs):

        for i, (x, y) in enumerate(train_loader):

            x, y = x.to(args.device), y.to(args.device)

            if args.adversarial and epoch > args.num_epochs // 2:
                x = pgd_attack_smooth(model, x, y, args.eps, noise,     
                                      sample_size=4, p=args.p)
            else:
                x = x + noise.sample(x.shape)

            optimizer.zero_grad()
            loss = model.loss(x, y).mean()
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.cpu().data)

            if i % args.print_every == 0:
                logger.info(f"Epoch: {epoch}\t" + 
                            f"Itr: {i} / {len(train_loader)}\t" + 
                            f"Loss: {loss_meter.value()[0]:.2f}\t"
                            f"Mins: {(time_meter.value() / 60):.2f}")
                loss_meter.reset()

        if epoch % 10 == 0:
            save_path = f"ckpts/{args.experiment_name}/{epoch}/"
            Path(save_path).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), f"{save_path}/model_ckpt.torch")

        annealer.step()

    save_path = f"ckpts/{args.experiment_name}/model_ckpt.torch"
    torch.save(model.state_dict(), save_path)

