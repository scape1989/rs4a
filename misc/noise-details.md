Further details below.

**L-1 Matched Noises**

Fix $\frac{1}{d}\mathbb{E}[||x||_1] = \sigma$.
$$
\begin{align*}
p(x)& = \mathrm{Normal}(0,\lambda^2 I)&\mathbb{E}[||x||_1] & = \lambda d\sqrt{2/\pi} & \lambda & = \sqrt{\pi/2}\ \sigma\\
p(x) & = \mathrm{Laplace}(0,\lambda) & \mathbb{E}[||x||_1] & = \lambda d & \lambda & = \sigma\\
p(x) & = \mathrm{Uniform}(-\lambda,\lambda) & \mathbb{E}[||x||_1] &= \frac{1}{2}\lambda d & \lambda & = 2\sigma\\
p(x) & = \mathrm{ExpInf}(\lambda,k) &\mathbb{E}[||x||_1] & \approx \frac{1}{2}\lambda d \ast & \lambda & \approx \frac{2}{\ast} \sigma \approx \frac{2}{d}\sigma\\
p(x) & = \mathrm{Exp1(\lambda,k)} & \mathbb{E}{[||x||_1]}& = \lambda \ast & \lambda & \approx \frac{d}{\ast} \sigma \approx \sigma\\
p(x) & = \mathrm{Exp2}(\lambda,k) & \mathbb{E}[||x||_1] & = \sqrt{\frac{2d}{\pi}}\lambda\ast & \lambda & = \sqrt{\frac{\pi d}{2}}\frac{\sigma}{\ast}\\
p(x) & = \mathrm{Lomax}(\lambda,k) & \mathbb{E}[||x||_1] & = \frac{\lambda d}{k-1} & \lambda & = (k-1)\sigma
\end{align*}
$$
The expected absolute value of a Laplace distribution follows from [Liu and Kozubowski 2015]. 

**ExpInf Details**
$$
p(x) \propto \exp\left(-(||x||_\infty/\lambda)^k\right)
$$
In two dimensions, we can imagine level sets that are squares around the origin. We can divide the sampling process into first sampling a radius $r$, then sampling uniformly from a square at radius $r$.

Observe that for ExpInf noise, the probability of sampling a radius $r$ is proportional to the surface area of a hypercube with radius $r$ times the probability of the corresponding $||x||_\infty$, i.e. 
$$
\begin{align*}
p(r)&\propto (2d) (2r)^{d-1}e^{-(r/\lambda)^k}\partial r\\
	& \propto r^{d-1}e^{-(r/\lambda)^k}\partial r\\
p(z) &\propto z^{(d-1)/k}e^{-z/\lambda^k}\frac{1}{k}(z^{1/k-1})\partial z\\
&\propto z^{d/k-1}e^{-z/\lambda^k}\partial z
\end{align*}
$$
Note that above we use a change of variables to make the sampling process tractable.
$$
z = r^k \implies r = z^{1/k}\quad\quad\quad\quad \partial r = \frac{1}{k}(z^{1/k-1})\partial z
$$
We first sample $z\sim\mathrm{Gamma}(\alpha=d/k,\beta=(1/\lambda)^k)$. Then to sample from the square at radius $r$, we pick one dimension to set $x_i=r$ and sample uniformly $x_j\sim \mathrm{Uniform}\left(-r, r\right)$ for $j\neq i$.

Note a Gamma distribution with parameters $\alpha,\beta$ has $\mathbb{E}[z] =\alpha/\beta$ and $\mathrm{Var}[z] = \alpha/\beta^2$. In our case, 
$$
\begin{align*}
\mathbb{E}[z] & = \lambda^k d/k\\
\mathbb{E}[r] & = \lambda \Gamma\left(\frac{d+1}{k}\right)/\Gamma\left(\frac{d}{k}\right) \triangleq\lambda\ast
\end{align*}
$$
We can also derive the expected norms, noting that $\mathbb{E}[x_j^2] = \frac{1}{12}(2\lambda\ast)^2 = \frac{1}{3}\lambda^2\ast^2$.
$$
\begin{align*}
\mathbb{E}[||x||^2_2] & = \mathbb{E}\left[\mathbb{E}\left[||x||_2^2 \; |\; r\right]\right] \\
& = \mathbb{E}\left[\mathbb{E}\left[x_1^2+\dots+x_d^2 \;|\; r\right]\right]\\
& = (\lambda\ast)^2 + (d-1)\frac{1}{3}\lambda^2\ast^2\\
& \approx \frac{1}{3}d\lambda^2\ast^2 \\
\mathbb{E}[||x||_1]& =  \mathbb{E}\left[\mathbb{E}\left[||x||_1 \; |\; r\right]\right]\\
& = \lambda\ast + (d-1)\frac{1}{2}\lambda\ast\\
& = \frac{1}{2}(d+1)\lambda\ast\\
& \approx \frac{1}{2}d\lambda\ast
\end{align*}
$$

#### **Exp1 Details**

$$
p(x) \propto \exp\left(-(||x||_1/\lambda)^k\right)
$$

In two dimensions, we can imagine level sets that are $\ell_1$ balls around the origin. We can divide the sampling process into first sampling a radius $r$, then uniformly sampling from the $\ell_1$ ball at radius $r$.

We first sample $z\sim \mathrm{Gamma}(\alpha=d/k, \beta=(1/\lambda)^k)$. Then to sample from the corresponding $\ell_1$ ball we can sample from the simplex (equivalent to sampling from a Dirichlet distribution with $\alpha=1$) then set coordinates to $\pm 1$ uniformly at random and multiply by $z$.

We can also derive the expected norms,
$$
\begin{align*}
\mathbb{E}\left[||x||_1\right] & = \mathbb{E}[\mathbb{E}[||x||_1\;|\;r]]\\
															 & = \mathbb{E}[r] = \lambda \ast \\
\mathbb{E}[||x||^2_2] & = \mathbb{E}\left[\mathbb{E}[x_1^2 + \dots + x_d^2 | r]\right]\\
& = \frac{2}{d+1}(\lambda\ast)^2
\end{align*}
$$
Above we used the fact for a Dirichlet distributed variable $x$ with $\alpha=1$, we have 
$$
\begin{align*}
\mathbb{E}[x_i] & = \frac{1}{d} & \mathrm{Var}[x_i] &= \frac{d-1}{(d+1)d^2}\\
\mathbb{E}[x_i^2] & = \frac{2}{d(d+1)} & \mathbb{E}\left[\sum_{i=1}^d x_i^2\right] & = \frac{2}{d+1},
\end{align*}
$$
and scaling $x_i$ by the radius which has expected value $\lambda \ast$ yields second moments that scale as $\lambda^2\ast^2$.

**Exp2 Details**
$$
p(x) \propto \exp(-(||x||_2/\lambda)^k)
$$

In two dimensions, we can imagine level sets that are $\ell_2$ balls around the origin. We can divide the sampling process into first sampling a radius $r$, then uniformly sampling from the $\ell_2$ ball at radius $r$.

We first sample $z\sim\mathrm{Gamma}(\alpha=d/k, \beta=(1/\lambda)^k)$. Then to sample from the corresponding $\ell_2$ ball we simply sample $\tilde{x}_i \sim N(0,1)$ and normalize by dividing each by $\sqrt{\tilde{x}_1^2 +\dots+\tilde{x}_d^2}$, i.e.
$$
x_i = \frac{\tilde{x}_i}{\sqrt{||\tilde{x}||^2_2}}
$$
The expected norms are 
$$
\begin{align*}
\mathbb{E}[||x||_2^2] & = \mathbb{E}[\mathbb{E}[x_1^2+\dots+x_d^2|r]]\\
	& = (\lambda\ast)^2\\
\mathbb{E}[||x||_1] & = \mathbb{E}[\mathbb{E}[|x_1| + \dots + |x_d| | r,||\tilde{x}||_2^2]]\\
& = \sqrt{\frac{2d}{\pi}}\lambda\ast
\end{align*}
$$
Above we used the fact that $\mathbb{E}[r] = \lambda\ast$ and $\mathbb{E}[||\tilde{x}||_2^2] = d$, and the corresponding folded normal distribution has expected value $\mathbb{E}[|\tilde{x}_i|] = \sqrt{2/\pi}$.

**Loss Estimators**

Suppose we want to estimate the following, where the expectation is over $\delta \sim N(0,1)$ and $x$ is fixed.
$$
-\log \mathbb{E}[(x+\delta)^2] = -\log(x^2+1).
$$
The closed form expression arises from observing that $x+\delta^{(k)}\sim N(x,1)$.

Now we take Monte Carlo estimates $\delta_1,\dots,\delta_K$ and use the estimator
$$
-\log \frac{1}{K}\sum_{k=1}^K(x+\delta^{(k)})^2.
$$
*The estimator is asymptotically consistent*

As $K\rightarrow\infty$, the estimator converges to $-\log(x^2+1)$.

*The estimator is biased*

In general we have (due to Jensen's),
$$
\begin{align*}
\mathbb{E}_{\delta^{(k)}}\left[-\log \frac{1}{K}\sum_{k=1}^K (x+\delta^{(k)})^2\right] & \geq -\log \mathbb{E}_{\delta^{(k)}}\left[\frac{1}{K}\sum_{k=1}^K (x+\delta^{(k)})^2\right]\\
& = -\log \mathbb{E}_\delta[(x+\delta)^2]
\end{align*}
$$
*How much bias?*

To be more precise, a second-order Taylor expansion around $x$ yields
$$
\mathbb{E}[-\log (x)] \approx -\log \mathbb{E}[x] + \frac{\mathrm{Var}[x]}{2\mathbb{E}[x]^2}
$$
Applying that here, we have
$$
\begin{align*}
\mathbb{E}_{\delta^{(k)}}\left[-\log \frac{1}{K}\sum_{k=1}^K (x+\delta^{(k)})^2\right] & \approx -\log \mathbb{E}_{\delta^{(k)}}\left[\frac{1}{K}\sum_{k=1}^K (x+\delta^{(k)})^2\right] + \frac{\mathrm{Var}_{\delta^{(k)}}\left[\frac{1}{K}\sum_{k=1}^K (x+\delta^{(k)})^2\right]}{2\mathbb{E}_{\delta^{(k)}}\left[\frac{1}{K}\sum_{k=1}^K (x+\delta^{(k)})^2\right]^2}\\
& = -\log\mathbb{E}_\delta[(x+\delta^2)] + \frac{\mathrm{Var}_\delta[(x+\delta)^2]}{2K\mathbb{E}_\delta[(x+\delta^2)]^2}
\end{align*}
$$

So the bias scales as $1/K$.

**MaskNoise**

Suppose we want a classifier provably robust to $\ell_0$ perturbations of size $R$.

We construct a *smooth* classifier by random i.i.d. sub-sampling (i.e. masking), that is, noise such that
$$
\epsilon_i = \begin{cases}-x_i & \mbox{with probability } 1-p \\ 0 & \mbox{with probability } p .\end{cases}
$$
Without loss of generality suppose the $R$ adversarial perturbations occur at indices $1,\dots,R$. By a simple union bound, we have
$$
\mathrm{Pr}[x+\epsilon \mathrm{\ contains\ adv\ perturbation}]= \mathrm{Pr}[\epsilon_1=0 \cup \dots\cup\epsilon_R=0] \leq Rp.
$$
Now consider the hard smooth classifier 
$$
F_\mathrm{hard}(x) \triangleq \mathbb{E}_{\epsilon}[\arg\max P_\theta(y|x + \epsilon)]\quad\quad\epsilon\sim p(x),
$$
We know that at most $Rp$ of the resulting probability mass function can be moved adversarially. So the hard smooth classifier is robust to adversarial perturbations of size $R$ as long as
$$
p_A > \frac{1}{2}+Rp.
$$

The expected $\ell_2$ norm is,
$$
\begin{align*}
\mathbb{E}[||\epsilon||_2^2|x] & = \mathbb{E}[\epsilon_1^2+\dots+\epsilon_d^2|x]\\
& = (1-p)x_1^2 \dots + (1-p)x_d^2\\
& = (1-p)||x||_2^2\\
\mathbb{E}[||\epsilon||_2^2] & = (1-p)\mathbb{E}[||x||_2^2]
\end{align*}
$$

