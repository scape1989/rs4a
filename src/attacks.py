import torch
import torch.nn as nn
from torch.autograd import grad
from src.smooth import *


def project_onto_ball(x, eps, p="inf"):
    """
    Note that projection onto inf-norm and 2-norm take O(d) time, and
    projection onto 1-norm takes O(dlogd) using the sorting-based algorithm 
    given in [Duchi et al. 2008].
    """
    original_shape = x.shape
    x = x.view(x.shape[0], -1)
    if p == "inf":
        x = x.clamp(-eps, eps)
    elif p == 2:
        x = x.renorm(p=2, dim=0, maxnorm=eps)
    elif p == 1:
        mask = (torch.norm(x, p=1, dim=1) < eps).float().unsqueeze(1)
        sgns = torch.sign(x)
        mu, _ = torch.sort(torch.abs(x), dim=1, descending=True)
        cumsum = torch.cumsum(mu, dim=1)
        arange = torch.arange(1, x.shape[1] + 1, device=x.device)
        rho, _ = torch.max((mu * arange > (cumsum - eps)) * arange, dim=1)
        rho_cpu = rho.cpu()
        theta = (cumsum[torch.arange(x.shape[0]), rho_cpu] - eps) / rho
        proj = (x - theta.unsqueeze(1)).clamp(min=0)
        x = mask * x + (1 - mask) * proj * sgns
    else:
        raise ValueError("Can only project onto 1,2,inf norm balls.")
    return x.view(original_shape)

def fgsm_attack(model, x, y, eps):
    x.requires_grad = True
    grads = grad(model.loss(x, y).mean(), x)[0]
    x = x + eps * torch.sign(grads) 
    return x.detach()

def pgd_attack(model, x, y, eps, steps=20, p="inf", clamp=(0, 1)):
    step_size = 2 * eps / steps
    x.requires_grad = True
    x_orig = x.clone().detach()
    for _ in range(steps):
        grads = grad(model.loss(x, y).mean(), x)[0]
        grads_norm = torch.norm(grads.view(x.shape[0], -1), dim=1, p=2)
        shape = (x.shape[0],) + (len(grads.shape) - 1) * (1,)
        grads = grads / grads_norm.view(shape)
        diff = (x + step_size * grads).clamp(*clamp) - x_orig
        diff = project_onto_ball(diff, eps, p)
        x = x_orig + diff
    x = x.detach()
    x.requires_grad = False
    return x

def pgd_attack_smooth(model, x, y, eps, noise, sample_size, steps=20,  
                      p="inf", clamp=(0, 1)):
    step_size = 2 * eps / steps
    x.requires_grad = True
    x_orig = x.clone().detach()
    for _ in range(steps):
        forecast = smooth_predict_soft(model, x, noise, sample_size)
        loss = -forecast.log_prob(y).mean()
        grads = grad(loss, x)[0]
        grads_norm = torch.norm(grads.view(x.shape[0], -1), dim=1, p=2)
        shape = (x.shape[0],) + (len(grads.shape) - 1) * (1,)
        grads = grads / grads_norm.view(shape)
        diff = (x + step_size * grads).clamp(*clamp) - x_orig
        diff = project_onto_ball(diff, eps, p)
        x = x_orig + diff
    return x.detach()

