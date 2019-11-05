import torch
import torch.nn as nn
from torch.autograd import grad
from src.smooth import *



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
        if p == "inf":
            diff = torch.clamp(diff, -eps, eps)
        else:
            diff = diff.view(x.shape[0], -1).renorm(p=p, dim=0, maxnorm=eps)
        x = x_orig + diff.view(x.shape)
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
        if p == "inf":
            diff = torch.clamp(diff, -eps, eps)
        else:
            diff = diff.view(x.shape[0], -1).renorm(p=p, dim=0, maxnorm=eps)
        x = x_orig + diff.view(x.shape)
    return x.detach()

