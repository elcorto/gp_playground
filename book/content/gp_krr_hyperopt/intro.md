# Compare GPs and kernel ridge regression (KRR)

Show that GPs and KRR are the same w.r.t. weights and predictions. They only
differ in the way hyperparameter optimization is done.

In order to underline the latter, we perform a hyperopt for a sklearn KRR and
GP model using the same external optimizer which is not present in sklearn. We
use `scipy.optimizer.differential_evolution` as an example of a global
optimization method.

## Kernel / covariance function

We use the radial basis function (RBF) kernel function $\kappa(\cdot,\cdot)$,
also called squared-exponential kernel. The kernel matrix
is

```{math}
K_{ij} = \kappa(\ve x_i, \ve x_j)
```

Note that there are many other RBFs but `sklearn` only implements that one.

```py
from sklearn.gaussian_process.kernels import RBF
RBF(length_scale=...)
```

## `KernelRidge`

We solve

```{math}
    :label: e:krr_solve
    (\ma K + \eta\,\ma I)\,\ve\alpha = \ve y
```

for the weights $\ve\alpha$ (called `KernelRidge.alpha_` in `sklearn`).

We specify $\eta$ as `alpha`

```py
KernelRidge(alpha=noise_level, kernel=RBF(length_scale=length_scale))
```

Calling

```py
KernelRidge(...).fit(X, y)
```

solves {eq}`e:krr_solve` once.

## `GaussianProcessRegressor`

In addition to `RBF`, we can use a `WhiteKernel` "to learn global noise", so
the kernel we use is a combination of two kernels which are responsible for
modeling different aspects of the data (i.e. "kernel engineering"). The
resulting kernel matrix is the same as the above, where

```py
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)
```

results in

```{math}
    \ma K + \eta\,\ma I
```

and we solve {eq}`e:krr_solve` for $\ve\alpha$, also called `GaussianProcessRegressor.alpha_`.

### Differences to KRR

#### Noise in `WhiteKernel`

In contrast to `KernelRidge`, calling

```py
GaussianProcessRegressor(...).fit(X, y)
```

does not solve for the weights once, but by default
(`GaussianProcessRegressor(optimizer=...)` is not `None`) uses a local
optimizer to optimize all kernel hyperparameters, solving {eq}`e:krr_solve` in
each step.

One can also specify $\eta$ as regularization parameter as in `KernelRidge`

```py
GaussianProcessRegressor(alpha=noise_level)
```

(in fact the default is not zero but 1e-10), which has the exact same effect
*when solving for the weights*. But when calling
`GaussianProcessRegressor.predict()`, the noise value will not be present. For
more details, see [this section](s:pred_noise).

The `GaussianProcessRegressor` optimizer cannot optimize $\eta$ when
given as regularization parameter, since it only optimizes kernel
hyperparameters, which is why we have to sneak it in via
`WhiteKernel(noise_level=)` where we interpret it as noise, while setting the
regularization parameter `alpha=0`. This is a technicality and might be a bit
confusing since from a textbook point of view, $\eta$ is not a parameter
of any kernel.

#### Hyperopt objective function

The difference to KRR is that the GP implementation optimizes the kernel's
params, here $\ell$ and $\eta$ treated as kernel param, by maximization
of the log marginal likelihood (LML) while KRR needs to use cross validation.
Also we get `y_std` or `y_cov` if we want, so of course the GP is in general
the preferred solution. Additionally, the GP can use the LML's gradient to do
local optimization, which can be fast if the LML evaluation is fast and it can
be the global min if the LML surface is convex, at least the neighborhood of a
good start guess.

### GP optimizer

In addition to using `GaussianProcessRegressor`'s internal optimizer code path,
either by using the default `l_bfgs_b` local optimizer or by setting a custom one
using `optimizer=my_optimizer`, we show how to optimize the GP's
hyperparameters in the same way as any other model by setting
`GaussianProcessRegressor(optimizer=None)` in combination with an external
optimizer. With that, we can use either
`GaussianProcessRegressor(alpha=noise_level)`, i.e. treat it as
`KernelRidge(alpha=noise_level)`, or
`WhiteKernel(noise_level=noise_level)` and get the exact same results.

We define a custom GP optimizer using `scipy.optimize.differential_evolution`
(i) to show how this can be done in general and (ii) because the default local
optimizer (`l_bfgs_b`), also with `n_restarts_optimizer>0` can get stuck in local
optima or on flat plateaus sometimes. Sometimes because the start guess is
randomly selected from bounds (the docs are bleak on how to fix the RNG for
that, so we don't).

### Example results of optimized models

GP, using the internal optimizer API and `RBF+WhiteKernel`:

`k1` is the Gaussian RBF kernel. `length_scale` is the optimized kernel width
parameter. `k2` is the `WhiteKernel` with its optimized `noise_level` parameter.

```py
{'k1': RBF(length_scale=0.147),
 'k1__length_scale': 0.14696558218508174,
 'k1__length_scale_bounds': (1e-05, 2),
 'k2': WhiteKernel(noise_level=0.0882),
 'k2__noise_level': 0.08820850820059796,
 'k2__noise_level_bounds': (0.001, 1)}
```

Fitted GP weights can be accessed by

```py
GaussianProcessRegressor.alpha_
```

and optimized kernel hyper params by

```py
GaussianProcessRegressor.kernel_.k1.length_scale
GaussianProcessRegressor.kernel_.k2.noise_level
```

where trailing underscores denote values after calling `fit()`: weights
`alpha_` and hyperopt `kernel_`.

## Why the GP and KRR hyperparameters are different after optimization

We use both models to solve {eq}`e:krr_solve` and therefore the results of the
hyperopt ($\ell$ and $\eta$) should be the same ... which
they aren't.

The reason is, as mentioned above already, that KRR has to resort to something
like cross validation (CV) to get a useful optimization objective, while GPs
can use maximization of the LML (see also [this part of the `sklearn`
docs][sklearn_gr_krr]). They can be equivalent, given one performs a very
particular and super costly variant of CV involving an "exhaustive leave-p-out
cross-validation averaged over all values of p and all held-out test sets when
using the log posterior predictive probability as the scoring rule", see
https://arxiv.org/abs/1905.08737 for details. This is nice but hard to do in
practice. Instead, we use `KFold` (try to replace `KFold` by `LeavePOut` with
`p>1` and then wait ...). This basically means that any form of practically
usable CV is an approximation of the LML with varying quality.

We plot the CV and -LML surface as function of $\ell$ and $\eta$ on a log scale
to get a visual representation of the problem that we solve here. `sklearn`
uses the log of $\ell$ and $\eta$ internally because (see
`sklearn.gaussian_process.kernels.Kernel.theta`, where `theta=[length_scale,
noise_level]` here):

    Note that theta are typically the log-transformed values of the
    kernel's hyperparameters as this representation of the search space
    is more amenable for hyperparameter search, as hyperparameters like
    length-scales naturally live on a log-scale.

We do the same if `HyperOpt(..., logscale=True)`.

## Data scaling

For both KRR and GP below, we work with the same scaled data (esp. zero mean).
`KernelRidge` has no

* constant offset term to `fit()`, i.e. there is no `fit_intercept` as in `Ridge`
* `normalize_y` as in `GaussianProcessRegressor`, which is why we use
  `GaussianProcessRegressor(normalize_y=False)` to ensure a correct comparison.

Still KRR can fit the data when the mean is very non-zero (e.g. `y += 1000`)
since the hyperopt still finds correct params. Also with fixed $\ell$ and
$\eta$ KRR and GP still produce the same weights $\ve\alpha$ and predictions
because they both solve {eq}`e:krr_solve`. However in case of $y$ far away from
zero, the hyperopt for `GaussianProcessRegressor(normalize_y=False)` fails
because the LML is changed such that we can't find a global opt any longer even
for large param bounds up to say `[1e-10, 1000]` for both $\ell$ and $\eta$.
This is because with `normalize_y=True`, the GP implementation zeros the data
mean before doing anything since it implements Alg. 2.1 from
{cite}`rasmussen_2006_GaussianProcessesMachine` which assumes a zero mean
function in the calculation of weights and LML. In the prediction the mean is
added back at the end.


[sklearn_gr_krr]: https://scikit-learn.org/stable/modules/gaussian_process.html#comparison-of-gpr-and-kernel-ridge-regression
