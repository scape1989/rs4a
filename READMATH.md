### Randomized Smoothing of All Shapes and Sizes

Last update: February 2020.

---

Code to accompany our paper.

#### Experiments

To reproduce our SOTA $\ell_1$ results on CIFAR-10, we need to train models over 
$$
\sigma \in \{0.15, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,2.0,2.25, 2.5,2.75, 3.0,3.25,3.5\}.
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

Results will be saved to the `ckpts/` directory. Then to plot the  figures, run:

```
python3 scripts/analyze.py --dir=ckpts --show
```

Further examples of training/testing scripts can be found in the `jobs/` directory.

#### Trained Models

Our pre-trained models will be released shortly.

#### Repository



#### Randomized Smoothing Preliminaries

Let $P_\theta(y|x)$ denote the categorical forecast, parameterized by neural network $\theta \in \Theta$. Let $\arg\max P_\theta(y|x)$ denote the mode of the forecast and $\max P_\theta(y|x)$ denote its corresponding predicted probability.

A soft smoothed classifier is the mixture model of forecasts smoothed by noise on the inputs,
$$
F_\mathrm{soft}(x) \triangleq \mathbb{E}_{\delta}[P_\theta(y|x + \delta)]\quad\quad\delta\sim q
$$

It turns out this classifier has smoothness properties that yield robustness [1].

Now consider the zero-one loss (note that while this is a proper scoring rule, it is not *strictly* proper).
$$
\mathcal{S}(P_\theta, y) = 1\{ \arg\max P_\theta(y|x) = y\}.
$$

We'd like to obtain guarantees on this loss. To do so we'll need to use the:

A hard smoothed classifier is the mixture model of forecast modes smoothed by noise on the inputs, 
$$
F_\mathrm{hard}(x) \triangleq \mathbb{E}_{\epsilon}[\arg\max P_\theta(y|x + \delta)]\quad\quad\delta\sim q
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

Of course, we need to estimate $c$ as well, which is done with more Monte Carlo sampling.

#### Examples

Below we show an example of how to use our implemented noises.

```python
from src.noises import UniformNoise

# instantiation
noise = UniformNoise(device="cpu", dim=3072, sigma=0.5)

# training code, to generate samples
noisy_x = noise.sample(x)

# testing code, for L1 adversary
prob_lower_bound = noise.certify(prob_lower_bound, adv=1)
```

#### References

[1] Cohen, J., Rosenfeld, E., and Kolter, Z. (2019). Certified Adversarial Robustness via Randomized Smoothing. In International Conference on Machine Learning, pp. 1310–1320.