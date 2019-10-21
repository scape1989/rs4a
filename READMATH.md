**Toward Robust Forecasting**

In this setting our output is in the form of an entire distribution $P_\theta$ over the output space.

It is reasonable to formulate an attack as the following.
$$
\underset{x':||x-x'|| < \epsilon}{\arg\max} D_{KL}(p_\theta(x) || p_\theta(x'))
$$
How would randomized smoothing help?