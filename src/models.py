import math
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical, Bernoulli
from torchvision import models as base_models
from src.wide_resnet import WideResNet


CIFAR_10_MU = [0.4914, 0.4822, 0.4465]
CIFAR_10_SIG = [0.2023, 0.1994, 0.2010]


class ResNet(nn.Module):
    
    def __init__(self, dataset, device):
        super().__init__()
        self.device = device
        if dataset == "cifar":
            self.model = nn.Sequential(
                NormalizeLayer((3, 1, 1), device, CIFAR_10_MU, CIFAR_10_SIG),
                WideResNet(depth=40, num_classes=10, widen_factor=2))
        elif dataset == "imagenet":
            pass
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


class LinearModel(nn.Module):

    def __init__(self, dataset, device):
        super().__init__()
        self.device = device
        if dataset == "cifar":
            self.model = nn.Sequential(
                NormalizeLayer((3, 1, 1), device, CIFAR_10_MU, CIFAR_10_SIG),
                nn.Linear(3 * 32 * 32, 10))
        elif dataset == "cifar":
            pass
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

