import scipy as sp
import scipy.stats
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal, Bernoulli
from statsmodels.stats.proportion import proportion_confint



def smooth_predict_soft(model, x, noise, sample_size=64, clamp=(0, 1)):
    # todo: support mixture distributions
    batch_size = x.shape[0]
    samples_shape = [1, sample_size] + ([1] * (len(x.shape) - 1))
    samples = x.unsqueeze(1).repeat(samples_shape)
    samples = (samples + noise.sample(samples.shape)).clamp(*clamp)
    samples = samples.view(*[-1] + [*samples.shape][2:])
    thetas = model.forward(samples).view(batch_size, sample_size, -1)
    return model.forecast(thetas.mean(dim=1))

def smooth_predict_hard_binary(model, x, noise, sample_size=64, clamp=(0, 1)):
    batch_size = x.shape[0]
    samples_shape = [1, sample_size] + ([1] * (len(x.shape) - 1))
    samples = x.unsqueeze(1).repeat(samples_shape)
    samples = (samples + noise.sample(samples.shape)).clamp(*clamp)
    samples = samples.view(*[-1] + [*samples.shape][2:])
    logits = model.forward(samples).view(batch_size, sample_size, -1)
    probs = torch.sigmoid(logits).round()
    return Bernoulli(probs=probs.mean(dim=1))

def smooth_predict_hard(model, x, noise, sample_size=64, clamp=(0, 1)):
    samples_shape = [1, sample_size] + ([1] * (len(x.shape) - 1))
    samples = x.unsqueeze(1).repeat(samples_shape)
    samples = (samples + noise.sample(samples.shape)).clamp(*clamp)
    samples = samples.view(*[-1] + [*samples.shape][2:])
    logits = model.forward(samples).view(x.shape[0], sample_size, -1)
    num_cats = logits.shape[-1]
    top_cats = torch.argmax(logits, dim=2)
    counts = nn.functional.one_hot(top_cats, num_cats).float().sum(dim=1)
    return Categorical(probs=counts / counts.shape[1])

def certify_smoothed(model, x, top_cats, alpha, noise, sample_size):
    preds = smooth_predict_hard(model, x, noise, sample_size)
    top_probs = preds.probs.gather(1, top_cats.unsqueeze(1)).detach().cpu()
    lower, _ = proportion_confint(top_probs * sample_size, sample_size, alpha=alpha, method="beta")
    return noise.certify(torch.tensor(lower.squeeze(), dtype=torch.float))

