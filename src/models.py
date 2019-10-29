import math
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical, Bernoulli
from torchvision import models as base_models
from src.cifar_resnet import ResNet as cifar_resnet


class ResNet(nn.Module):
    
    def __init__(self, num_classes, device):
        super().__init__()
        self.device = device
        self.model = nn.Sequential(
            NormalizeLayer((3, 1, 1), device),
            cifar_resnet(depth=20, num_classes=10))
#            base_models.resnet18(pretrained=False))
#        self.model[-1].fc = nn.Linear(512, num_classes, bias=True)
        self.model.to(device)

    def forward(self, x):
        return Categorical(logits=self.model(x))

    def loss(self, x, y):
        forecast = self.forward(x)
        return -forecast.log_prob(y)

    def embed(self, x):
#        x = self.model[0](x)
#        x = self.model[1].conv1(x)
#        x = self.model[1].bn1(x)
#        x = self.model[1].relu(x)
#        x = self.model[1].maxpool(x)
#        x = self.model[1].layer1(x)
#        x = self.model[1].layer2(x)
#        x = self.model[1].layer3(x)
#        x = self.model[1].layer4(x)
#        x = self.model[1].avgpool(x)
#        x = torch.flatten(x, 1)
        return x

class NormalizeLayer(nn.Module):
    """
    Normalizes across the first non-batch axis.

    Example:
        (64, 3, 32, 32) [CIFAR] => normalizes across channels
        (64, 8) [UCI]  => normalizes across features
    """
    def __init__(self, dim, device):
        super().__init__()
        self.dim = dim
        self.mu = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float, device=device).reshape((3, 1, 1))
        self.log_sig = torch.log(torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float, device=device)).reshape((3, 1, 1))
        self.initialized = True

    def forward(self, x):
        if not self.initialized:
            self.initialize_parameters(x)
            self.initialized = True
        return (x - self.mu) / torch.exp(self.log_sig)
    
    def initialize_parameters(self, x):
        with torch.no_grad():
            mu = x.view(x.shape[0], x.shape[1], -1).mean((0, 2))
            std = x.view(x.shape[0], x.shape[1], -1).std((0, 2))
            self.mu.copy_(mu.data.view(*self.mu.shape))
            self.log_sig.copy_(torch.log(std).data.view(*self.mu.shape))


class ForecastNN(nn.Module):

    def __init__(self, in_dim, hidden_dim=50, device="cpu"):
        super().__init__()
        self.network = nn.Sequential(
            NormalizeLayer(in_dim, device),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        params = self.network(x)
        mean, sd = torch.split(params, 1, dim=1)
        return Normal(loc=mean, scale=torch.exp(sd))

    def loss(self, x, y):
        forecast = self.forward(x)
        return -forecast.log_prob(y)


class LogisticRegression(nn.Module):
    
    def __init__(self, in_dim, device="cpu"):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x):
        return Bernoulli(logits=self.linear(x))

    def loss(self, x, y):
        return -self.forward(x).log_prob(y.unsqueeze(1))
