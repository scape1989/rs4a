import math
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical, Bernoulli
from torchvision import models as base_models
from src.wide_resnet import WideResNet


CIFAR_10_MU = [0.4914, 0.4822, 0.4465]
CIFAR_10_SIGMA = [0.2023, 0.1994, 0.2010]

IMAGENET_MU = [0.485, 0.456, 0.406]
IMAGENET_SIGMA = [0.229, 0.224, 0.225]

MNIST_MU = [0.1307,]
MNIST_SIGMA = [0.3081,]


class ResNet(nn.Module):

    def __init__(self, dataset, device):
        super().__init__()
        self.device = device
        if dataset == "cifar":
            self.model = nn.Sequential(
                NormalizeLayer((3, 1, 1), device, CIFAR_10_MU, CIFAR_10_SIGMA),
                WideResNet(depth=40, num_classes=10, widen_factor=2))
        elif dataset == "imagenet":
            self.model = nn.Sequential(
                NormalizeLayer((3, 1, 1), device, IMAGENET_MU, IMAGENET_SIGMA),
                WideResNet(depth=40, num_classes=10, widen_factor=2))
        else:
            raise ValueError
        self.model.to(device)

    def forecast(self, theta):
        return Categorical(logits=theta)

    def forward(self, x):
        return self.model(x)

    def loss(self, x, y):
        forecast = self.forecast(self.forward(x))
        return -forecast.log_prob(y)

    def hard_loss(self, x, y):
        forecast = self.forecast(self.forward(x))

class LinearModel(nn.Module):

    def __init__(self, dataset, device):
        super().__init__()
        self.device = device
        if dataset == "cifar":
            self.norm = NormalizeLayer((3, 1, 1), device, CIFAR_10_MU, CIFAR_10_SIGMA)
            self.model = nn.Linear(3 * 32 * 32, 10)
        elif dataset == "mnist":
            self.norm = NormalizeLayer((1, 1, 1), device, MNIST_MU, MNIST_SIGMA)
            self.model = nn.Linear(28 * 28, 10)
        else:
            raise ValueError
        self.model.to(device)

    def forecast(self, theta):
        return Categorical(logits=theta)

    def forward(self, x):
        x = self.norm(x).view(x.shape[0], -1)
        return self.model(x)

    def loss(self, x, y):
        forecast = self.forecast(self.forward(x))
        return -forecast.log_prob(y)


class AlexNet(nn.Module):
    def __init__(self, dataset, device):
        super().__init__()
        self.device = device
        self.norm = NormalizeLayer((3, 1, 1), device, CIFAR_10_MU, CIFAR_10_SIGMA)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10),
        )

    def forecast(self, theta):
        return Categorical(logits=theta)

    def forward(self, x):
        x = self.norm(x)
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def loss(self, x, y):
        forecast = self.forecast(self.forward(x))
        return -forecast.log_prob(y)

class MLP(nn.Module):

    def __init__(self, dataset, device):
        super().__init__()
        self.device = device
        if dataset == "cifar":
            self.norm = NormalizeLayer((3, 1, 1), device, CIFAR_10_MU, CIFAR_10_SIGMA)
            self.model = nn.Sequential(
                nn.Linear(3 * 32 * 32, 2048),
                nn.ReLU(),
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Linear(512, 10))
        elif dataset == "mnist":
            self.norm = NormalizeLayer((1, 1, 1), device, MNIST_MU, MNIST_SIGMA)
            self.model = nn.Sequential(
                nn.Linear(28 * 28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10))
        else:
            raise ValueError
        self.model.to(device)

    def forecast(self, theta):
        return Categorical(logits=theta)

    def forward(self, x):
        x = self.norm(x).view(x.shape[0], -1)
        return self.model(x)

    def loss(self, x, y):
        forecast = self.forecast(self.forward(x))
        return -forecast.log_prob(y)


class NormalizeLayer(nn.Module):
    """
    Normalizes across the first non-batch axis.

    Examples:
        (64, 3, 32, 32) [CIFAR] => normalizes across channels
        (64, 8) [UCI]  => normalizes across features
    """
    def __init__(self, dim, device, mu=None, sigma=None):
        super().__init__()
        self.dim = dim
        if mu and sigma:
            self.mu = torch.tensor(mu, device=device).reshape(dim)
            self.log_sig = torch.tensor(sigma, device=device)
            self.log_sig = torch.log(self.log_sig).reshape(dim)
            self.initialized = True
        else:
            self.mu = nn.Parameter(torch.zeros(dim, device=device))
            self.log_sig = nn.Parameter(torch.zeros(dim, device=device))
            self.initialized = False

    def forward(self, x):
        if not self.initialized:
            self.initialize_parameters(x)
            self.initialized = True
        return (x - self.mu) / torch.exp(self.log_sig)

    def initialize_parameters(self, x):
        with torch.no_grad():
            mu = x.view(x.shape[0], x.shape[1], -1).mean((0, 2))
            std = x.view(x.shape[0], x.shape[1], -1).std((0, 2))
            self.mu.copy_(mu.data.view(self.dim))
            self.log_sig.copy_(torch.log(std).data.view(self.dim))


class ForecastNN(nn.Module):

    def __init__(self, in_dim, hidden_dim=50, device="cpu"):
        super().__init__()
        self.network = nn.Sequential(
            NormalizeLayer(in_dim, device),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forecast(self, theta):
        mean, sd = torch.split(theta, 1, dim=1)
        return Normal(loc=mean, scale=torch.exp(sd))

    def forward(self, x):
        return self.network(x)

    def loss(self, x, y):
        forecast = self.forecast(self.forward(x))
        return -forecast.log_prob(y)


class LogisticRegression(nn.Module):

    def __init__(self, in_dim, device="cpu"):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forecast(self, theta):
        return Bernoulli(logits=theta)

    def forward(self, x):
        return self.linear(x)

    def loss(self, x, y):
        return -self.forecast(self.forward(x)).log_prob(y.unsqueeze(1))

