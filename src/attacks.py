import torch
import torch.nn as nn
from torch.autograd import grad



def fgsm_attack(model, x, y, eps):
    """
    L-inf attack.
    """
    x.requires_grad = True
    grads = grad(model.loss(x, y).mean(), x)[0]
    x = x + eps * torch.sign(grads) 
    return x.detach()

def pgd_attack(model, x, y, eps, step_size=0.1, steps=100, p="inf"):
    """
    L-inf attack.
    """
    x.requires_grad = True
    x_orig = x.clone().detach()
    for _ in range(steps):
        grads = grad(model.loss(x, y).mean(), x)[0]
        diff = (x + step_size * grads).clamp(0, 1) - x_orig
        if p == "inf":
            diff = torch.clamp(diff, -eps, eps)
        else:
            diff = diff.view(x.shape[0], -1).renorm(p=p, dim=-1, maxnorm=eps)
        x = x_orig + diff.view(x.shape)
    return x.detach()

def pgd_attack_smooth(model, x, y, eps, step_size=0.1, steps=100, p="inf"):
    x.requires_grad = True
