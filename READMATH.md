### Robust Image Classification via Randomized Smoothing

Last update: November 2019.

---

We use randomized smoothing [1] to obtain certifiably robust image classifiers. In particular in this repository we are interested in robustness to $\ell_1$ perturbations of the input. Adversarial training is incorporated, as in [2].

Let $P_\theta(y|x)$ denote the categorical forecast for a particular input, parameterized by neural network $\theta \in \Theta$. We let $\arg\max P_\theta(y|x)$ denote the mode of the forecast and $\max P_\theta(y|x)$ denote its corresponding predicted probability.

We define a *soft smoothed classifier* as a mixture model of forecasts smoothed by zero-mean noise on the inputs, i.e.
$$
\mathbb{E}_{\epsilon}[P_\theta(y|x + \epsilon)]\quad\quad\epsilon\sim p(x)
$$

It turns out a classifier smoothed in this way has robustnessness properties in a Lipschitz sense. 

Now consider the zero-one loss (note that while this is a proper scoring rule, it is not *strictly* proper)
$$
\mathcal{S}(P_\theta, y) = 1\{ \arg\max P_\theta(y|x) = 1\}.
$$

We'd like to obtain guarantees on this loss. To do so we'll need to use a *hard smoothed classifier*, which we define as the mixture model of forecast modes when smoothed by zero-mean noise on the inputs, i.e.
$$
\mathbb{E}_{\epsilon}[1\{\arg\max P_\theta(y|x + \epsilon)\}]\quad\quad\epsilon\sim p(x),
$$
where $1\{\cdot\}$ denotes a categorical distribution with all mass on the appropriate category.

By interpreting this hard smoothed classifier as a soft smooth classifier, we can get guarantees on the modes that are predicted. 

In this repository we implement:

1. Laplace Noise


2. Uniform Noise


2. ExpInf Noise (with parameter $k$)


$$
p(x) \propto \exp(-(||x||_\infty/\sigma)^k)
$$

3. 



#### References

[1] Cohen, J., Rosenfeld, E., and Kolter, Z. (2019). Certified Adversarial Robustness via Randomized Smoothing. In International Conference on Machine Learning, pp. 1310–1320.

[2] Salman, H., Yang, G., Li, J., Zhang, P., Zhang, H., Razenshteyn, I., and Bubeck, S. (2019). Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers.

