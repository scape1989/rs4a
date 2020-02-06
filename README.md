### Randomized Smoothing of All Shapes and Sizes

Last update: February 2020.

---

Code to accompany our paper.

#### Experiments

To reproduce our SOTA <img alt="$\ell_1$" src="svgs/839a0dc412c4f8670dd1064e0d6d412f.svg" align="middle" width="13.40191379999999pt" height="22.831056599999986pt"/> results on CIFAR-10, we need to train models over 
<p align="center"><img alt="$$&#10;\sigma \in \{0.15, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0\}.&#10;$$" src="svgs/005f5f4214b48a2ddaac2f7971ec8602.svg" align="middle" width="339.66355995pt" height="16.438356pt"/></p>
For each, run the following:

```
python3 -m src.train
--noise=UniformNoise
--sigma={sigma}
--experiment-name=cifar_UniformNoise_{sigma}

python3 -m src.test
--noise=UniformNoise
--sigma={sigma}
--experiment-name=cifar_UniformNoise_{sigma}
--p=1
--sample-size-cert=100000
--sample-size-pred=64
--noise-batch-size=512

```

Results will be saved to the `ckpts/` directory. Then to reproduce plots, run:

```
python3 scripts/analyze.py --dir=ckpts --show
```

Further examples of training/testing scripts can be found in the `jobs/` directory.

#### Randomized Smoothing Preliminaries

Let <img alt="$P_\theta(y|x)$" src="svgs/ba3062180eb2cf7d620b00a9aaea814c.svg" align="middle" width="53.38666574999999pt" height="24.65753399999998pt"/> denote the categorical forecast, parameterized by neural network <img alt="$\theta \in \Theta$" src="svgs/1694fa79d012a58f8baabdf8e4974216.svg" align="middle" width="41.05009919999999pt" height="22.831056599999986pt"/>. Let <img alt="$\arg\max P_\theta(y|x)$" src="svgs/de5bf9581ffc4e81f1f60bb66f9fef02.svg" align="middle" width="112.56474899999998pt" height="24.65753399999998pt"/> denote the mode of the forecast and <img alt="$\max P_\theta(y|x)$" src="svgs/6cbe611d60edc2b341743a89d999631f.svg" align="middle" width="86.72003835pt" height="24.65753399999998pt"/> denote its corresponding predicted probability.

A soft smoothed classifier is the mixture model of forecasts smoothed by noise on the inputs,
<p align="center"><img alt="$$&#10;F_\mathrm{soft}(x) \triangleq \mathbb{E}_{\delta}[P_\theta(y|x + \delta)]\quad\quad\delta\sim q&#10;$$" src="svgs/4e71ca05b5e1ebb3579d11fed7cb6620.svg" align="middle" width="255.82543634999996pt" height="19.178118299999998pt"/></p>

It turns out this classifier has smoothness properties that yield robustness [1].

Now consider the zero-one loss (note that while this is a proper scoring rule, it is not *strictly* proper).
<p align="center"><img alt="$$&#10;\mathcal{S}(P_\theta, y) = 1\{ \arg\max P_\theta(y|x) = y\}.&#10;$$" src="svgs/35afab4a47af77d24c4952d215843383.svg" align="middle" width="252.19156710000001pt" height="16.438356pt"/></p>

We'd like to obtain guarantees on this loss. To do so we'll need to use the:

A hard smoothed classifier is the mixture model of forecast modes smoothed by noise on the inputs, 
<p align="center"><img alt="$$&#10;F_\mathrm{hard}(x) \triangleq \mathbb{E}_{\epsilon}[\arg\max P_\theta(y|x + \delta)]\quad\quad\delta\sim q&#10;$$" src="svgs/8a90d19a936a95690ecab61f2df2ca32.svg" align="middle" width="319.32771585pt" height="19.178118299999998pt"/></p>
By interpreting this hard smoothed classifier as a soft smooth classifier, we can get guarantees on the modes that are predicted by the hard smoothed classifier. 

In practice we need to resort to Monte Carlo samping to approximate <img alt="$F_\mathrm{hard}$" src="svgs/f370369da8812afdb3f69806ea24b29f.svg" align="middle" width="36.780984899999986pt" height="22.465723500000017pt"/>. 

Certification bounds depend on <img alt="$\rho$" src="svgs/6dec54c48a0438a5fcde6053bdb9d712.svg" align="middle" width="8.49888434999999pt" height="14.15524440000002pt"/>, which we denote to be
<p align="center"><img alt="$$&#10;\rho \triangleq \mathrm{Pr}_\delta[\arg\max P_\theta(y|x+\delta) = c],&#10;$$" src="svgs/7f3199cea50227a3dacef5613141636b.svg" align="middle" width="238.57029404999997pt" height="19.178118299999998pt"/></p>


where <img alt="$c$" src="svgs/3e18a4a28fdee1744e5e3f79d13b9ff6.svg" align="middle" width="7.11380504999999pt" height="14.15524440000002pt"/> is the top predicted category by <img alt="$F_\mathrm{hard}$" src="svgs/f370369da8812afdb3f69806ea24b29f.svg" align="middle" width="36.780984899999986pt" height="22.465723500000017pt"/>. 

We obtain a lower bound on <img alt="$\rho$" src="svgs/6dec54c48a0438a5fcde6053bdb9d712.svg" align="middle" width="8.49888434999999pt" height="14.15524440000002pt"/> from Monte Carlo samples of the forecasted modes <img alt="$c_1,\dots,c_m$" src="svgs/f09b71056793c2733ff7df921c93df4d.svg" align="middle" width="69.79633319999999pt" height="14.15524440000002pt"/>, at a confidence level <img alt="$\alpha$" src="svgs/c745b9b57c145ec5577b82542b2df546.svg" align="middle" width="10.57650494999999pt" height="14.15524440000002pt"/> with,
<p align="center"><img alt="$$&#10;\hat \rho_\mathrm{lower} =\inf \left\{\rho :\mathrm{Pr}\left[\mathrm{Binomial}(m, \rho) \geq \frac{1}{m}\sum_{i=1}^mc_i\right]\geq \alpha\right\}&#10;$$" src="svgs/2852180f02b7693c4af1dd8058dee6d8.svg" align="middle" width="394.3703412pt" height="49.315569599999996pt"/></p>
We note that this one-sided Clopper-Pearson confidence interval is generally conservative.

Of course, we need to estimate <img alt="$c$" src="svgs/3e18a4a28fdee1744e5e3f79d13b9ff6.svg" align="middle" width="7.11380504999999pt" height="14.15524440000002pt"/> as well, which is done with more Monte Carlo sampling.

#### Noise distributions

In this repository we implement the following noises. Note that we want the expected <img alt="$\ell_2$" src="svgs/336fefe2418749fabf50594e52f7b776.svg" align="middle" width="13.40191379999999pt" height="22.831056599999986pt"/> distance of perturbations to be approximately the same for purposes of comparison, i.e. fix <img alt="$\frac{1}{d}\mathbb{E}[||x||_2^2] = \sigma^2$" src="svgs/8cfd7c775dc1c9365532c8edeeb94b28.svg" align="middle" width="102.39446745pt" height="27.77565449999998pt"/>. We derive the appropriate correspondences between distributions below.

<p align="center"><img alt="$$&#10;\begin{align*}&#10;p(x) &amp; = \mathrm{Normal}(0,\lambda^2 I) &amp; \mathbb{E}[||x||^2_2] &amp;= \lambda^2 d &amp; \lambda &amp; = \sigma\\&#10;p(x) &amp; = \mathrm{Laplace}(0, \lambda) &amp; \mathbb{E}[||x||_2^2]&amp; = 2\lambda^2d&amp; \lambda &amp;= \frac{1}{\sqrt 2}\sigma\\&#10;p(x) &amp; = \mathrm{Uniform(-\lambda,\lambda)}&amp; \mathbb{E}[||x||_2^2] &amp; = \frac{1}{3}\lambda^2d &amp; \lambda &amp; = \sqrt 3 \sigma\\&#10;p(x)&amp; = \mathrm{ExpInf}(\lambda,k)&amp; \mathbb{E}[||x||_2^2] &amp; \approx \frac{1}{3}d\lambda^2\gamma^2 &amp; \lambda &amp; = \frac{\sqrt 3}{\gamma}\sigma\approx \frac{\sqrt 3}{d}\sigma\\&#10;p(x) &amp; = \mathrm{Exp1}(\lambda,k) &amp; \mathbb{E}[||x||^2_2] &amp; \approx \frac{2}{d+1}\lambda^2\gamma^2 &amp; \lambda &amp; = \sqrt\frac{d(d+1)}{2}\frac{\sigma}{\gamma}\\&#10;p(x) &amp; = \mathrm{Exp2}(\lambda,k) &amp; \mathbb{E}[||x||_2^2] &amp; = \lambda^2\gamma^2 &amp; \lambda &amp; = \sqrt{d}\ \frac{\sigma}{\gamma} \\&#10;p(x) &amp; = \mathrm{Lomax}(\lambda,k)&amp;\mathbb{E}[||x||^2_2] &amp; = \frac{2\lambda^2d}{(k-1)(k-2)} &amp; \lambda &amp; = \sqrt{\frac{(k-1)(k-2)}{2}} \sigma&#10;\end{align*}&#10;$$" src="svgs/b2824c303e2d31ae63b496f0022e657e.svg" align="middle" width="540.2531590499999pt" height="288.62438879999996pt"/></p>

Here <img alt="$\gamma = \Gamma\left(\frac{d+1}{k}\right)/\Gamma\left(\frac{d}{k}\right)$" src="svgs/af5c8e5da313bc6a05bb574b122c8651.svg" align="middle" width="137.10824655pt" height="28.92634470000001pt"/>. Example:

```python
from src.noises import UniformNoise
noise = UniformNoise(1.0, device="cpu", dim=3072)

# training code, to generate samples
noisy_x = noise.sample(x)

# testing code, for L1 adversary
prob_lower_bound = noise.certify(prob_lower_bound, p=1)
```

#### References

[1] Cohen, J., Rosenfeld, E., and Kolter, Z. (2019). Certified Adversarial Robustness via Randomized Smoothing. In International Conference on Machine Learning, pp. 1310–1320.