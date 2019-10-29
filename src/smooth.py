import scipy as sp
import scipy.stats
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal, Bernoulli
from statsmodels.stats.proportion import proportion_confint



def smooth_predict_soft(model, x, sigma=0.1, sample_size=64):
    samples = x.unsqueeze(1).repeat(1, sample_size, 1)
    noise = torch.randn_like(samples) * sigma
    samples = (samples + noise).clamp(0, 1).view(-1, 2)
    preds = model.forward(samples).logits.view(-1, sample_size)
    return Bernoulli(logits=preds.mean(dim=1))

#def smooth_predict_soft(model, x, sigma=0.1, sample_size=64):
#    samples = x.unsqueeze(1).repeat(1, sample_size, 1, 1, 1)
#    noise = torch.randn_like(samples) * sigma
#    samples = (samples + noise).clamp(0, 1).view(-1, 3, 32, 32)
#    preds = model.forward(samples).logits.view(-1, sample_size, 10)
#    return Categorical(logits=preds.mean(dim=1))

def smooth_predict_hard(model, x, noise, sample_size=64):
    samples = x.unsqueeze(1).repeat(1, sample_size, 1, 1, 1)
    samples = (samples + noise.sample(samples.shape)).clamp(0, 1)
    samples = samples.view(-1, 3, 32, 32)
    logits = model.forward(samples).logits.view(-1, sample_size, 10)
    probs = torch.softmax(logits, dim=2).round()
    return Categorical(probs=probs.mean(dim=1))

def certify_smoothed(model, x, cats, alpha, noise, sample_size=64):
    preds = smooth_predict_hard(model, x, noise, sample_size)
    cat_probs = preds.probs.gather(1, cats.unsqueeze(1)).detach().cpu()
    lower, _ = proportion_confint(cat_probs * sample_size, sample_size, 
                                  alpha=alpha, method="beta")
    return noise.certify(torch.tensor(lower.squeeze(), dtype=torch.float))

