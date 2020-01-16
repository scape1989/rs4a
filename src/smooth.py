import numpy as np
import scipy as sp
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Bernoulli
from statsmodels.stats.proportion import proportion_confint


def direct_train_log_lik(model, x, y, noise, sample_size=16):
    samples_shape = torch.Size([x.shape[0], sample_size]) + x.shape[1:]
    samples = x.unsqueeze(1).expand(samples_shape)
    samples = samples.reshape(torch.Size([-1]) + samples.shape[2:])
    samples = noise.sample(samples)
    thetas = model.forward(samples).view(x.shape[0], sample_size, -1)
    return torch.logsumexp(thetas[torch.arange(x.shape[0]), :, y] - \
                           torch.logsumexp(thetas, dim=2), dim=1) - \
           torch.log(torch.tensor(sample_size, dtype=torch.float, device=x.device))

def smooth_predict_soft(model, x, noise, sample_size=64):
    samples_shape = torch.Size([x.shape[0], sample_size]) + x.shape[1:]
    samples = x.unsqueeze(1).expand(samples_shape)
    samples = samples.reshape(torch.Size([-1]) + samples.shape[2:])
    samples = noise.sample(samples)
    thetas = model.forward(samples).view(x.shape[0], sample_size, -1)
    return Categorical(probs=model.forecast(thetas).probs.mean(dim=1))

def smooth_predict_hard_binary(model, x, noise, sample_size=64):
    batch_size = x.shape[0]
    samples_shape = [1, sample_size] + ([1] * (len(x.shape) - 1))
    samples = x.unsqueeze(1).repeat(samples_shape)
    samples = (samples + noise.sample(samples.shape))
    samples = samples.view(*[-1] + [*samples.shape][2:])
    logits = model.forward(samples).view(batch_size, sample_size, -1)
    probs = torch.sigmoid(logits).round()
    return Bernoulli(probs=probs.mean(dim=1))

def smooth_predict_hard(model, x, noise, sample_size=64, noise_batch_size=512, num_cats=10):

    counts = torch.zeros(x.shape[0], num_cats, dtype=torch.float, device=x.device)
    num_samples_left = sample_size

    while num_samples_left > 0:

        shape = torch.Size([x.shape[0], min(num_samples_left, noise_batch_size)]) + x.shape[1:]
        samples = x.unsqueeze(1).expand(shape)
        samples = samples.reshape(torch.Size([-1]) + samples.shape[2:])
        samples = noise.sample(samples.view(len(samples), -1)).view(samples.shape)
#        samples = torch.cat((samples, 1 - samples), dim=2) # for 2 channels
#        samples[torch.isnan(samples)] = 0
#        samples[torch.isnan(samples)] = 0
        logits = model.forward(samples).view(shape[:2] + torch.Size([-1]))
        top_cats = torch.argmax(logits, dim=2)
        counts += F.one_hot(top_cats, num_cats).float().sum(dim=1)
        num_samples_left -= noise_batch_size

    return Categorical(probs=counts)

def certify_smoothed(model, x, top_cats, alpha, noise, sample_size, noise_batch_size=512):
    preds = smooth_predict_hard(model, x, noise, sample_size, noise_batch_size)
    top_probs = preds.probs.gather(1, top_cats.unsqueeze(1)).detach().cpu()
    lower, _ = proportion_confint(top_probs * sample_size, sample_size, alpha=alpha, method="beta")
    lower = torch.tensor(lower.squeeze(), dtype=torch.float)
    return lower, noise.certify(lower)

