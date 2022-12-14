---
jupytext:
  cell_metadata_filter: -all
  executable: /usr/bin/env python3
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell}

```

# Comparison of GP libraries

We perform calculations using different GP implementations, which may serve as
a reference.

We test

* [tinygp](https://github.com/dfm/tinygp)
* [sklearn](https://scikit-learn.org)
* [GPy](https://github.com/SheffieldML/GPy)
* [gpytorch](https://gpytorch.ai)

We perform the same calculations with all libraries and compare to the results
obtained by a straight forward implementation of the textbook equations from
{cite}`rasmussen_2006_GaussianProcessesMachine`.

The values of the hyperparameters $\ell$ and $\sigma_n^2$ are usually the
result of "fitting the GP model to data", which means optimizing the GP's log
marginal likelihood as a function of both (e.g. what `sklearn`'s
`GaussianProcessRegressor` does by default when `optimizer != None`).

To ensure accurate comparisons, we

* skip param optimization and instead fix $\ell$ (kernel) and
  $\sigma_n^2$ (likelihood) because
  * testing correct prediction code paths is orthogonal to how those are obtained
  * codes use different optimizers and/or convergence thresholds and/or start
    values, so optimized params might not be
    * from the same (local) optimum of the log marginal likelihood
    * equal enough numerically
* skip code-internal data normalization
* set regularization defaults to zero where needed to ensure that we only add
  $\sigma_n^2$ to the kernel matrix diag

First, we generate reference textbook results. We compare each tested GP library
against them using `numpy.testing` tools. Since we don't calculate with
actual training data, we generate random data inputs and targets $(\ve x_i
\in \mathbb R^D, y_i \in \mathbb R)$.

We calculate GP prior and posterior data in the `predict` and
`predict_noiseless` setting:

reference result name | meaning
-|-
text_pri          | prior, `predict_noiseless`
text_pri_noise    | prior, `predict`
text_pos          | posterior, `predict_noiseless`
text_pos_noise    | posterior, `predict`

Note that the `text_pri_noise` case is not useful in practice but we include
it since some libraries expose the possibility to construct this case.

```{code-cell}

import numpy as np

from common import (
    textbook_prior,
    textbook_posterior,
    textbook_posterior_noise,
    cov2std,
)


def rand(*size):
    return rng.uniform(size=size)


rng = np.random.default_rng(123)

X_train = rand(100, 5)
y_train = rand(100)
X_pred = rand(50, 5)
noise_level = 0.1
length_scale = 1

text_pri = textbook_prior(noise_level=0, length_scale=length_scale)(X_pred)
text_pri_noise = textbook_prior(
    noise_level=noise_level, length_scale=length_scale
)(X_pred)

text_pos = textbook_posterior(
    X_train, y_train, noise_level=noise_level, length_scale=length_scale
)(X_pred)
text_pos_noise = textbook_posterior_noise(
    X_train, y_train, noise_level=noise_level, length_scale=length_scale
)(X_pred)
```

## tinygp

```{code-cell}

# =========================================================================
# tinygp
# =========================================================================
import jax

jax.config.update("jax_enable_x64", True)
from tinygp import GaussianProcess
from tinygp.kernels import ExpSquared
from tinygp.solvers import DirectSolver


def compare_tinygp(gp, text_gp):
    y_mean, y_std, y_cov = text_gp
    np.testing.assert_allclose(y_mean, gp.loc)
    np.testing.assert_allclose(y_std, np.sqrt(gp.variance))
    np.testing.assert_allclose(y_cov, gp.covariance)


# prior w/o noise
gp = GaussianProcess(
    kernel=ExpSquared(scale=length_scale),
    X=X_pred,
    diag=0,
)
compare_tinygp(gp, text_pri)

# prior w/ noise
gp = GaussianProcess(
    kernel=ExpSquared(scale=length_scale),
    X=X_pred,
    diag=noise_level,
)
compare_tinygp(gp, text_pri_noise)

# posterior, like GPy predict_noiseless() = R&W textbook eqns
gp = GaussianProcess(
    kernel=ExpSquared(scale=length_scale),
    X=X_train,
    solver=DirectSolver,
    diag=noise_level,
)
cond = gp.condition(y_train, X_pred, diag=0)
compare_tinygp(cond.gp, text_pos)

# posterior, like GPy predict() -- add noise_level to y_cov's diag
gp = GaussianProcess(
    kernel=ExpSquared(scale=length_scale),
    X=X_train,
    solver=DirectSolver,
    diag=noise_level,
)
cond = gp.condition(y_train, X_pred, diag=noise_level)
compare_tinygp(cond.gp, text_pos_noise)
```

## GPy

```{code-cell}

# =========================================================================
# GPy
# =========================================================================

# Can't convince GPy to be more accurate than atol / rtol. There must be
# hidden jitter defaults lurking around.


def compare_gpy(pred_func, text_gp):
    gp_mean, gp_cov = pred_func(X_pred, full_cov=True)
    y_mean, y_std, y_cov = text_gp
    np.testing.assert_allclose(y_mean, gp_mean[:, 0])
    np.testing.assert_allclose(y_std, cov2std(gp_cov))
    np.testing.assert_allclose(y_cov, gp_cov, rtol=1e-4)


# posterior
import GPy

gpy_kernel = GPy.kern.RBF(
    input_dim=X_train.shape[1],
    lengthscale=length_scale,
    variance=1,
    inv_l=True,
)
gp = GPy.models.GPRegression(
    X_train,
    y_train[:, None],
    gpy_kernel,
    normalizer=False,
    noise_var=noise_level,
)

# predict_noiseless()
compare_gpy(gp.predict_noiseless, text_pos)

# predict(): y_cov has noise_level added to the diag
compare_gpy(gp.predict, text_pos_noise)
```

## GPyTorch

```{code-cell}

# =========================================================================
# GPyTorch
# =========================================================================

import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import Positive
import torch as T

# Despite all our efforts, can't convince gpytorch to be more accurate than
# atol / rtol set below. There must be hidden jitter defaults lurking around.
gpytorch.settings.verbose_linalg(True)
gpytorch.settings.cholesky_jitter(float=0, double=0, half=0)
gpytorch.settings.fast_computations(
    covar_root_decomposition=False, log_prob=False, solves=False
)
gpytorch.settings.fast_pred_var(False)
gpytorch.settings.linalg_dtypes(
    default=T.float64, symeig=T.float64, cholesky=T.float64
)


def compare_gpytorch(gp, text_gp):
    y_mean, y_std, y_cov = text_gp
    np.testing.assert_allclose(y_mean, gp.mean.detach().numpy())
    np.testing.assert_allclose(y_std, np.sqrt(gp.variance.detach().numpy()))
    np.testing.assert_allclose(
        y_cov, gp.covariance_matrix.detach().numpy(), rtol=1e-5
    )


def fixed(val):
    return Positive(initial_value=val, transform=None, inv_transform=None)


class ExactGPModel(gpytorch.models.ExactGP):
    """API:
    model.forward() -> prior
    model()         -> posterior
    """

    def __init__(self, X, y, likelihood):
        super().__init__(X, y, likelihood)
        kernel = gpytorch.kernels.RBFKernel(
            lengthscale_constraint=fixed(length_scale),
            eps=0,
        )
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

    def forward(self, X):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(X), self.covar_module(X)
        )


likelihood = GaussianLikelihood(noise_constraint=fixed(noise_level))
model = ExactGPModel(T.from_numpy(X_train), T.from_numpy(y_train), likelihood)

# prior w/o noise: model.forward()
gp = model.forward(T.from_numpy(X_pred))
compare_gpytorch(gp, text_pri)

# prior w/ noise: likelihood(model.forward())
gp = likelihood(model.forward(T.from_numpy(X_pred)))
compare_gpytorch(gp, text_pri_noise)

model.eval()
likelihood.eval()

# posterior, like GPy predict_noiseless(): model()
gp = model(T.from_numpy(X_pred))
compare_gpytorch(gp, text_pos)

# posterior, like GPy predict(): likelihood(model())
gp = likelihood(model(T.from_numpy(X_pred)))
compare_gpytorch(gp, text_pos_noise)
```

(s:gp_pred_comp_sklearn)=
## sklearn

```{code-cell}

# =========================================================================
# sklearn
# =========================================================================
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# WhiteKernel API:
#   k = WhiteKernel()
#
#   k(X) = np.eye(...) * noise_level
#   k(X, X) = np.zeros(...)


def compare_sklearn(gp, text_gp):
    y_mean, y_std, y_cov = text_gp
    gp_mean, gp_std = gp.predict(X_pred, return_std=True)
    _, gp_cov = gp.predict(X_pred, return_cov=True)
    np.testing.assert_allclose(y_mean, gp_mean)
    np.testing.assert_allclose(y_std, gp_std)
    np.testing.assert_allclose(y_cov, gp_cov)


# -------------------------------------------------------------------------
# noise as regularization param, no noise in kernel
# -------------------------------------------------------------------------

# prior
gp = GaussianProcessRegressor(
    kernel=RBF(length_scale=length_scale),
    alpha=noise_level,
    optimizer=None,
    normalize_y=False,
)
compare_sklearn(gp, text_pri)

# posterior, like GPy predict_noiseless
gp = gp.fit(X_train, y_train)
compare_sklearn(gp, text_pos)

# -------------------------------------------------------------------------
# noise as kernel param via WhiteKernel
# -------------------------------------------------------------------------

# prior, calling kernel(X) in predict() retains noise in kernel via
# WhiteKernel
gp = GaussianProcessRegressor(
    kernel=RBF(length_scale=length_scale)
    + WhiteKernel(noise_level=noise_level),
    alpha=0,
    optimizer=None,
    normalize_y=False,
)
compare_sklearn(gp, text_pri_noise)

# posterior, like GPy predict(), when using kernel(X, X) instead of kernel(X)
# in sklearn's predict() then this is equal to GPy predict_noiseless()
gp = gp.fit(X_train, y_train)
compare_sklearn(gp, text_pos_noise)
```
