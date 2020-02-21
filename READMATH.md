### Randomized Smoothing of All Shapes and Sizes

Last update: February 2020.

---

Code to accompany our paper.

#### Experiments

To reproduce our SOTA $\ell_1$ results on CIFAR-10, we need to train models over 
$$
\sigma \in \{0.15, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,2.0,2.25, 2.5,2.75, 3.0,3.25,3.5\},
$$
For each value, run the following:

```
python3 -m src.train
--noise=UniformNoise
--sigma={sigma}
--experiment-name=cifar_UniformNoise_{sigma}

python3 -m src.test
--noise=UniformNoise
--sigma={sigma}
--experiment-name=cifar_UniformNoise_{sigma}
--sample-size-cert=100000
--sample-size-pred=64
--noise-batch-size=512
```

Results will be saved to the `ckpts/` directory. 

To draw a comparison to the benchmark noises, replace `UniformNoise` above with `GaussianNoise` and `LaplaceNoise`. Then to plot the  figures, run:

```
python3 -m scripts.analyze --dir=ckpts --show --adv=1
```

Note that other noises will need to be instantiated with the appropriate arguments when the appropriate training/testing code is invoked. For example:

```
python3 -m src.train
--noise=ExpInfNoise
--k=10
--j=100
--sigma=0.5
--experiment-name=cifar_ExpInfNoise_0.5
```

#### Trained Models

Our pre-trained models are available. 

ImageNet (ResNet-50):

- [[Uniform, Sigma=0.25]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_uniform_025.pt)
- [[Uniform, Sigma=0.5]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_uniform_050.pt)
- [[Uniform, Sigma=0.75]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_uniform_075.pt)
- [[Uniform, Sigma=1.0]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_uniform_100.pt)
- [[Uniform, Sigma=1.25]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_uniform_125.pt)
- [[Uniform, Sigma=1.5]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_uniform_155.pt)
- [[Uniform, Sigma=1.75]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_uniform_175.pt)
- [[Uniform, Sigma=2.0]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_uniform_200.pt)
- [[Uniform, Sigma=2.25]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_uniform_225.pt)
- [[Uniform, Sigma=2.5]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_uniform_250.pt)
- [[Uniform, Sigma=2.75]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_uniform_275.pt)
- [[Uniform, Sigma=3.0]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_uniform_300.pt)
- [[Uniform, Sigma=3.25]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_uniform_325.pt)
- [[Uniform, Sigma=3.50]](http://www.tonyduan.com/resources/2020_rs4a_ckpts/imagenet_uniform_350.pt)

CIFAR-10 (ResNet-18): coming soon.

An example of usage:

```python
from src.models import ResNet
from src.noises import UniformNoise
from src.smooth import *

# load the model
model = ResNet(dataset="imagenet", device="cuda")
saved_dict = torch.load("imagenet_uniform_05.pt")
model.load_state_dict(saved_dict)
model.eval()

# instantiation of noise
noise = UniformNoise(device="cpu", dim=3072, sigma=0.5)

# training code, to generate samples
noisy_x = noise.sample(x)

# testing code, certify for L1 adversary
preds = smooth_predict_hard(model, x, 64)
top_cats = preds.probs.argumax(dim=1)
prob_lb = certify_prob_lb(model, x, top_cats, 0.001, noise, 100000)
radius = noise.certify(prob_lb, adv=1)
```

#### Repository

1. `ckpts/` is used to store experiment checkpoints and results.
2. `data/` is used to store image datasets.
3. `examples/` contains toy visualizations of noisy data and sanity checks.
4. `tables/` contains caches of pre-calculated tables of certified radii.
5. `src/` contains the main souce code.
6. `scripts/` contains the analysis and plotting code.

Within the `src/` directory, the most salient files are:

1. `train.py` is used to train models and save to `ckpts/`.
2. `test.py` is used to test and compute robust certificates for $\ell_1,\ell_2,\ell_\infty$ adversaries.
3. `noises.py` is a library of noises derived for randomized smoothing.
4. `test_noises.py` is a unit test for the noises we include. 

#### Randomized Smoothing Preliminaries

Let $P_\theta(y|x)$ denote the categorical forecast, parameterized by neural network $\theta \in \Theta$. 

Note $\arg\max P_\theta(y|x)$ denotes the mode of the forecast.

A soft smoothed classifier is the mixture model of forecasts smoothed by noise on the inputs,
$$
F_\mathrm{soft}(x) \triangleq \mathbb{E}_{\delta}[P_\theta(y|x + \delta)]\quad\quad\delta\sim q
$$

It turns out this classifier has smoothness properties that yield robustness [1].

Now consider the zero-one loss.
$$
\mathcal{S}(P_\theta, y) = 1\{ \arg\max P_\theta(y|x) = y\}.
$$

We'd like to obtain guarantees on this loss. To do so we'll need to use:

A hard smoothed classifier is the mixture model of forecast modes smoothed by noise on the inputs, 
$$
F_\mathrm{hard}(x) \triangleq \mathbb{E}_{\delta}[\arg\max P_\theta(y|x + \delta)]\quad\quad\delta\sim q
$$
By interpreting this hard smoothed classifier as a soft smooth classifier, we can get guarantees on the modes that are predicted by the hard smoothed classifier. 

In practice we need to resort to Monte Carlo samping to approximate $F_\mathrm{hard}$. 

Certification bounds depend on $\rho$, which we denote to be
$$
\rho \triangleq \mathrm{Pr}_\delta[\arg\max P_\theta(y|x+\delta) = c],
$$


where $c$ is the top predicted category by $F_\mathrm{hard}$. 

We obtain a lower bound on $\rho$ from Monte Carlo samples of the forecasted modes $c_1,\dots,c_m$, at a confidence level $\alpha$ with,
$$
\hat \rho_\mathrm{lower} =\inf \left\{\rho :\mathrm{Pr}\left[\mathrm{Binomial}(m, \rho) \geq \frac{1}{m}\sum_{i=1}^mc_i\right]\geq \alpha\right\}
$$
We note that this one-sided Clopper-Pearson confidence interval is generally conservative.

Of course, we need to estimate $c$ as well, which is done with additional Monte Carlo sampling.

#### References

[1] Cohen, J., Rosenfeld, E., and Kolter, Z. (2019). Certified Adversarial Robustness via Randomized Smoothing. In International Conference on Machine Learning, pp. 1310–1320.
