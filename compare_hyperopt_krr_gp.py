#!/usr/bin/env python3

"""
Show how to hyper opt a sklearn KRR and GP model using the same external
optimizer which is not present in sklearn. We use
scipy.optimizer.differential_evolution as an example of a global optimization
method.

KernelRidge
-----------

Kernel Ridge Regression using a radial basis function (RBF) kernel function k,
here a Gaussian one, also called squared-exponential kernel:
RBF(length_scale=p). The kernel matrix is

    K_ij = k(x_i, x_j) = exp(-0.5 * d_ij^2 / p^2)
    d_ij = ||x_i - x_j||_2

Note that there are many other RBFs but sklearn only implements that one.

In the KRR setting, we have an L2 regularization term in the loss (also called
"weight decay")

    r * ||w||_2^2

with the weights w, which results in a kernel matrix used in training where we
add r to the diagonal (L2 regularization variant of Tikhonov regularization)

    K_ii += r

We solve

    (K + r I)^-1 w = y

to get w (called alpha_ in sklearn).

GaussianProcessRegressor
------------------------

In addition to RBF, we can use a WhiteKernel "to learn global noise", so the
kernel we use is a combination of two kernels which are responsible for
modeling different aspects of the data (i.e. "kernel engineering"). The
resulting kernel matrix is the same as the above, i.e.
RBF(length_scale=p)+WhiteKernel(noise_level=r) does

    K_ii += r

The difference to KRR is that the GP implementation optimizes the kernel's
params (p,r) by maximization of the log marginal likelihood (LML) while KRR
needs to use CV. Also we get y_std or y_cov if we want, so of course the GP is
in general the preferred solution. Additionally, the GP can use the LML's
gradient to do local optimization, which can be fast if the LML evaluation is
fast and it can be the global min if the LML surface is convex, at least the
neighborhood of a good start guess.

GP optimizer
------------

One can also specify r as GaussianProcessRegressor(alpha=r) if interpreted as
regularization parameter, in fact the default is not zero but 1e-10. However
the GP optimizer cannot optimize this, since it only optimizes kernel
hyperparameters, which is why we sneak it in via WhiteKernel(noise_level=r)
where we interpret it as noise, while setting alpha=0.

We show how to optimize the GP's hyperparameters in the same way as any other
model by using GaussianProcessRegressor(optimizer=None) in combination with an
external optimizer. With that, we can use either
GaussianProcessRegressor(alpha=r) or WhiteKernel(noise_level=r) and get the
exact same results.

We define a custom GP optimizer using scipy.optimize.differential_evolution()
(i) to show how this can be done in general and (ii) because the default local
optimizer (l_bfgs_b), also with n_restarts_optimizer > 0 can get stuck in local
optima or on flat plateaus sometimes. Sometimes because the start guess is
randomly selected from bounds (the docs are bleak on how to fix the RNG for
that, so we don't).

Example results of optimized models
-----------------------------------

GP, using the internal optimizer API and RBF+WhiteKernel:

k1 is the Gaussian RBF kernel. length_scale is the optimized kernel width
parameter. k2 is the WhiteKernel with its optimized noise_level parameter.

{'k1': RBF(length_scale=0.147),
 'k1__length_scale': 0.14696558218508174,
 'k1__length_scale_bounds': (1e-05, 2),
 'k2': WhiteKernel(noise_level=0.0882),
 'k2__noise_level': 0.08820850820059796,
 'k2__noise_level_bounds': (0.001, 1)}

Fitted GP weights can be accessed by
    GaussianProcessRegressor.alpha_
and optimized kernel hyper params by
    GaussianProcessRegressor.kernel_.k1.length_scale
    ...
(trailing underscores denote values after fitting (weights alpha_) and hyper
opt (kernel_)).

Why the GP and KRR hyperparameters are different after optimization
-------------------------------------------------------------------

We use both models to solve

    (K + r I)^-1 w = y

and therefore the results of the hyperopt (p and r) should be the same ... which
they aren't.

The reason is that KRR (see also [1]) has to resort to something like CV to get
a useful optimization objective to find p and r, while GPs can use maximization
of the LML. They can be equivalent, given one performs a very particular and
super costly variant of CV involving an "exhaustive leave-p-out
cross-validation averaged over all values of p and all held-out test sets when
using the log posterior predictive probability as the scoring rule", see
https://arxiv.org/abs/1905.08737 for details. This is nice but hard to do in
practice. Instead, we use KFold (try to replace KFold by LeavePOut() with p>1
and then wait ...). This basically means that any form of practically usable CV
is an approximation of the LML with varying quality.

We plot the CV and -LML surface as function of p and r on a log scale to get a
visual representation of the problem that we solve here. sklearn uses the log
of p and r internally because (see
sklearn.gaussian_process.kernels.Kernel.theta, theta=[p,r] here):

    Note that theta are typically the log-transformed values of the
    kernel's hyperparameters as this representation of the search space
    is more amenable for hyperparameter search, as hyperparameters like
    length-scales naturally live on a log-scale.

We do the same if HyperOpt(..., logscale=True).

Data scaling
------------
For both KRR and GP below, we work with the same scaled data (esp. zero mean).
KernelRidge has no

* constant offset term to fit, i.e. there is no fit_intercept as in Ridge
* normalize_y as in GaussianProcessRegressor, which is why we use
  GaussianProcessRegressor(normalize_y=False) to ensure a correct comparison.

Still KRR can fit the data when the mean is very non-zero (e.g. y += 1000)
since the hyperopt still finds correct params. Also with fixed p and r KRR and
GP still produce the same weights and predictions because they solve the same
equation for the weights. However in this case the hyperopt for
GaussianProcessRegressor(normalize_y=False) fails because the LML is changed
such that we can't find a global opt any longer even for large param bounds up
to say [1e-10, 1000] for p and r. This is because whith normalize_y=True, the
GP implementation zeros the data mean before doing anything since it
implements alg. 2.1 from Rasmussen & Williams which assumes a zero mean
function in the calculation of weights and LML. In the prediction the mean is
added back at the end.

refs
----
[1] https://scikit-learn.org/stable/modules/gaussian_process.html#comparison-of-gpr-and-kernel-ridge-regression
[2] http://www.gaussianprocess.org/gpml/ (Rasmussen & Williams 2006)
[3] https://probml.github.io/pml-book/book1.html (Murphy 2022)
[4] https://gregorygundersen.com/blog/2020/01/06/kernel-gp-regression
[5] https://peterroelants.github.io/posts/gaussian-process-tutorial/
"""

import itertools
import multiprocessing as mp

##import multiprocess as mp

import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from matplotlib import cm, ticker, colors

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.kernel_ridge import KernelRidge

from icecream import ic


def gt_func(x):
    """Ground truth"""
    return np.sin(x) * np.exp(-0.1 * x) + 10


def noise(x, rng, noise_level=0.1):
    """Gaussian noise."""
    # variable noise
    ##return rng.normal(loc=0, scale=noise_level, size=x.shape[0]) * np.exp(0.1*x) / 5
    # constant noise
    return rng.normal(loc=0, scale=noise_level, size=x.shape[0])


def transform_1d(scaler, x):
    assert x.ndim == 1
    return scaler.transform(x.reshape(-1, 1))[:, 0]


def de_callback(xk, convergence=None):
    """Callback for differential_evolution that prints the best individual per
    iteration."""
    ##ic(xk)


def de_callback_logscale(xk, convergence=None):
    """Callback for differential_evolution that prints the best individual per
    iteration.

    Since GaussianProcessRegressor's hyper optimizer code path internally works
    with log(p) (= xk[0]) and log(r) (= xk[1]), we need to exp() them before
    printing.

    We also do that in HyperOpt if logscale=True.
    """
    ##ic(np.exp(xk))


def gp_optimizer(obj_func, initial_theta, bounds):
    """Custom optimizer for GaussianProcessRegressor using
    differential_evolution.

    Ignore initial_theta since we need only bounds for differential_evolution.
    """
    # Avoid pickle error when using multiprocessing in
    # differential_evolution(..., workers=-1). We'd use
    # https://github.com/uqfoundation/multiprocess instead of the stdlib's
    # multiprocessing to work around that but sadly differential_evolution()
    # uses the latter internally, so we're stuck with that.
    global _gp_obj_func_wrapper

    def _gp_obj_func_wrapper(params):
        ##print(f"{obj_func(initial_theta)=}")
        # obj_func(theta, eval_gradient=True) hard-coded in
        # GaussianProcessRegressor, so it always returns the function value and
        # grad. However, we only need the function's value in
        # differential_evolution() below. obj_func = -log_marginal_likelihood.
        ##val, grad = obj_func(params)
        return obj_func(params)[0]

    opt_result = differential_evolution(
        ##lambda params: obj_func(params)[0],  # nope, no pickle for you
        _gp_obj_func_wrapper,
        bounds=bounds,
        callback=de_callback_logscale,
        **de_kwds_common,
    )
    return opt_result.x, opt_result.fun


def simple_cv(model, X, y, cv):
    """Same as

        -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error")

    but much faster because we bypass the rich API of
    cross_val_score and can thus skip many many checks.
    """
    errs = np.empty((cv.get_n_splits(X),), dtype=float)
    for ii, tup in enumerate(cv.split(X)):
        idxs_train, idxs_test = tup
        fm = model.fit(X[idxs_train, ...], y[idxs_train, ...])
        d = fm.predict(X[idxs_test, ...]) - y[idxs_test, ...]
        # MSE
        errs[ii] = np.dot(d, d) / len(d)
    return errs


class HyperOpt:
    """Optimize hyper params of a sklearn model using
    differential_evolution."""

    def __init__(self, bounds, get_model, logscale=False):
        self.bounds = bounds
        self.get_model = get_model
        self.logscale = logscale

    def obj_func(self, params, X, y):
        raise NotImplementedError

    def fit(self, X, y, return_params=False):
        global _ho_obj_func_wrapper

        if self.logscale:
            bounds_transform = lambda x: np.log(x)
            params_transform = lambda x: np.exp(x)
            callback = de_callback_logscale
        else:
            bounds_transform = lambda x: x
            params_transform = lambda x: x
            callback = de_callback

        def _ho_obj_func_wrapper(params):
            return self.obj_func(params_transform(params), X, y)

        opt_result = differential_evolution(
            _ho_obj_func_wrapper,
            bounds=bounds_transform(bounds),
            callback=callback,
            **de_kwds_common,
        )
        params = params_transform(opt_result.x)
        f = self.get_model(params).fit(X, y)
        if return_params:
            return f, params
        else:
            return f


class HyperOptKRR(HyperOpt):
    def __init__(self, *args, seed=None, **kwds):
        super().__init__(*args, **kwds)
        self.seed = seed

    def obj_func(self, params, X, y):
        cv = KFold(n_splits=5, random_state=self.seed, shuffle=True)
        return simple_cv(self.get_model(params), X, y, cv=cv).mean()


class HyperOptGP(HyperOpt):
    def obj_func(self, params, X, y):
        return -self.get_model(params).fit(X, y).log_marginal_likelihood()


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # setup data
    # -------------------------------------------------------------------------
    seed = 123
    rng = np.random.default_rng(seed=seed)

    # Equidistant x points: constant y_std (from GP) in-distribution
    ##x = np.linspace(0, 30, 60)
    #
    # Random x points for varying y_std. For some reason the results vary
    # depending on whether we sort the points. Shouldn't be the case?
    ##x = rng.uniform(0, 30, 60)
    x = np.sort(rng.uniform(0, 30, 60), axis=0)
    xspan = x.max() - x.min()
    xi = np.linspace(x.min() - 0.3 * xspan, x.max() + 0.3 * xspan, len(x) * 10)
    y = gt_func(x) + noise(x, rng, noise_level=0.1)
    yi_gt = gt_func(xi)

    # Data scaling.
    in_scaler = StandardScaler().fit(x.reshape(-1, 1))
    out_scaler = StandardScaler().fit(y.reshape(-1, 1))
    x = transform_1d(in_scaler, x)
    xi = transform_1d(in_scaler, xi)
    y = transform_1d(out_scaler, y)
    yi_gt = transform_1d(out_scaler, yi_gt)
    X = x[:, None]
    XI = xi[:, None]

    # -------------------------------------------------------------------------
    # Sanity check KernelRidge API: alpha added to diag of kernel matrix.
    # -------------------------------------------------------------------------
    param_p = 1
    param_r = 0.1
    kp = RBF(length_scale=param_p)
    f_krr_1 = KernelRidge(alpha=param_r, kernel=kp).fit(X, y)
    f_krr_2 = KernelRidge(alpha=0, kernel="precomputed").fit(
        kp(X, X) + np.eye(X.shape[0]) * param_r, y
    )

    np.testing.assert_allclose(f_krr_1.dual_coef_, f_krr_2.dual_coef_)
    np.testing.assert_allclose(f_krr_1.predict(XI), f_krr_2.predict(kp(XI, X)))

    # -------------------------------------------------------------------------
    # Show that WhiteKernel(noise_level=) is equal to regularization param
    # alpha=param_r in both sklearn models. No hyperopt just yet
    # (GaussianProcessRegressor(optimizer=None)). Use fixed hyper params=[p,r].
    # -------------------------------------------------------------------------
    f_gp_kp = GaussianProcessRegressor(
        kernel=kp,
        optimizer=None,
        normalize_y=False,
        alpha=param_r,
    ).fit(X, y)

    f_gp_kpr = GaussianProcessRegressor(
        kernel=RBF(length_scale=param_p) + WhiteKernel(noise_level=param_r),
        optimizer=None,
        normalize_y=False,
        alpha=0,
    ).fit(X, y)

    np.testing.assert_allclose(f_gp_kp.alpha_, f_gp_kpr.alpha_)
    np.testing.assert_allclose(f_gp_kp.predict(XI), f_gp_kpr.predict(XI))

    np.testing.assert_allclose(f_gp_kp.alpha_, f_krr_1.dual_coef_)
    np.testing.assert_allclose(f_gp_kp.predict(XI), f_krr_1.predict(XI))

    # -------------------------------------------------------------------------
    # Non-zero mean for fixed p and r.
    # -------------------------------------------------------------------------
    f_krr_nzm = KernelRidge(alpha=param_r, kernel=kp).fit(X, y + 1000)
    f_gp_nzm = GaussianProcessRegressor(
        kernel=RBF(length_scale=param_p) + WhiteKernel(noise_level=param_r),
        optimizer=None,
        normalize_y=False,
        alpha=0,
    ).fit(X, y + 1000)

    np.testing.assert_allclose(f_gp_nzm.alpha_, f_krr_nzm.dual_coef_)
    np.testing.assert_allclose(f_gp_nzm.predict(XI), f_krr_nzm.predict(XI))

    # -------------------------------------------------------------------------
    # hyperopt gp
    #
    # Show 4 ways to opt [p,r]. One using the default optimizer for
    # reference. Then 3 ways to use the custom differential evolution (DE) based one
    # using the HyperOpt helper class.
    #
    # DE results must be exactly equal if we use logscale=True, i.e. what
    # GaussianProcessRegressor does internally, so these tests only check
    # different code paths doing the same operations (of course also we fix all
    # RNG seeds, so DE is reproducible).
    #
    # With logscale=False params are not equal but very close, also because we
    # use polish=True which adds a final local optimizer run starting from the
    # best DE result where we assume that we're close to the global opt and
    # things are convex-ish.
    # -------------------------------------------------------------------------
    param_p_bounds = (1e-5, 10)
    param_r_bounds = (1e-10, 10)
    bounds = [param_p_bounds, param_r_bounds]

    de_kwds_common = dict(
        polish=True,
        disp=False,
        atol=0,
        tol=0.001,
        popsize=20,
        maxiter=10000,
        workers=-1,
        updating="deferred",
        seed=seed,
    )

    # Internal optimizer API. Use default local optimizer (BFGS).
    #
    # For the wider bounds above the -LML shows big flat plateaus and so the
    # local optimizer always goes off into the wild blue yonder with
    # n_restarts_optimizer=0 (the default).
    #
    ##ic("opt gp internal default RBF + WhiteKernel ...")
    f_gp_0 = GaussianProcessRegressor(
        kernel=RBF(length_scale_bounds=param_p_bounds)
        + WhiteKernel(
            noise_level_bounds=param_r_bounds,
        ),
        n_restarts_optimizer=5,
        normalize_y=False,
        alpha=0,
    ).fit(X, y)
    params_gp_0 = np.array(
        [
            f_gp_0.kernel_.k1.length_scale,
            f_gp_0.kernel_.k2.noise_level,
        ]
    )
    ic(params_gp_0)

    # Internal optimizer API. Use differential_evolution.
    #
    ##ic("opt gp internal RBF + WhiteKernel ...")
    f_gp_1 = GaussianProcessRegressor(
        kernel=RBF(length_scale_bounds=param_p_bounds)
        + WhiteKernel(
            noise_level_bounds=param_r_bounds,
        ),
        n_restarts_optimizer=0,
        optimizer=gp_optimizer,
        normalize_y=False,
        alpha=0,
    ).fit(X, y)

    params_gp_1 = np.array(
        [
            f_gp_1.kernel_.k1.length_scale,
            f_gp_1.kernel_.k2.noise_level,
        ]
    )
    ic(params_gp_1)

    # External optimizer using HyperOpt helper class.
    #
    # RBF + WhiteKernel
    #
    ##ic("opt gp external RBF + WhiteKernel ...")
    get_model_gp_2 = lambda params: GaussianProcessRegressor(
        kernel=RBF(length_scale=params[0])
        + WhiteKernel(noise_level=params[1]),
        optimizer=None,
        normalize_y=False,
        alpha=0,
    )
    f_gp_2, params_gp_2 = HyperOptGP(
        get_model=get_model_gp_2,
        bounds=bounds,
        logscale=True,
    ).fit(X, y, return_params=True)
    ic(params_gp_2)

    # External optimizer using HyperOpt helper class.
    #
    # RBF, alpha
    #
    ##ic("opt gp external RBF + alpha ...")
    get_model_gp_3 = lambda params: GaussianProcessRegressor(
        kernel=RBF(length_scale=params[0]),
        optimizer=None,
        normalize_y=False,
        alpha=params[1],
    )
    f_gp_3, params_gp_3 = HyperOptGP(
        get_model=get_model_gp_3,
        bounds=bounds,
        logscale=True,
    ).fit(X, y, return_params=True)
    ic(params_gp_3)

    np.testing.assert_allclose(params_gp_1, params_gp_2)
    np.testing.assert_allclose(params_gp_1, params_gp_3)

    np.testing.assert_allclose(f_gp_1.alpha_, f_gp_2.alpha_)
    np.testing.assert_allclose(f_gp_1.alpha_, f_gp_3.alpha_)

    np.testing.assert_allclose(f_gp_1.predict(XI), f_gp_2.predict(XI))
    np.testing.assert_allclose(f_gp_1.predict(XI), f_gp_3.predict(XI))

    # External optimizer using HyperOpt helper class.
    #
    # RBF, alpha
    # logscale=False
    #
    ##ic("opt gp external RBF + alpha nolog ...")
    get_model_gp_3_nolog = lambda params: GaussianProcessRegressor(
        kernel=RBF(length_scale=params[0]),
        optimizer=None,
        normalize_y=False,
        alpha=params[1],
    )
    f_gp_3_nolog, params_gp_3_nolog = HyperOptGP(
        get_model=get_model_gp_3_nolog,
        bounds=bounds,
        logscale=False,
    ).fit(X, y, return_params=True)
    ic(params_gp_3_nolog)

    # -------------------------------------------------------------------------
    # hyperopt krr
    #
    # KRR params will be different b/c CV != log_marginal_likelihood
    # -------------------------------------------------------------------------
    get_model_krr = lambda params: KernelRidge(
        alpha=params[1], kernel=RBF(length_scale=params[0])
    )

    ##ic("opt krr RBF + alpha ...")
    f_krr, params_krr = HyperOptKRR(
        bounds=bounds,
        get_model=get_model_krr,
        logscale=True,
        seed=seed,
    ).fit(X, y, return_params=True)
    ic(params_krr)

    ##ic("opt krr RBF + alpha nolog ...")
    f_krr_nolog, params_krr_nolog = HyperOptKRR(
        bounds=bounds,
        get_model=get_model_krr,
        logscale=False,
        seed=seed,
    ).fit(X, y, return_params=True)
    ic(params_krr_nolog)

    # -------------------------------------------------------------------------
    # Plot functions, data and GP's std
    # -------------------------------------------------------------------------
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["font.size"] = 18

    yi_krr = f_krr.predict(XI)
    yi_gp, yi_gp_std = f_gp_1.predict(XI, return_std=True)
    ##yi_gp, yi_gp_std = f_gp_3_nolog.predict(XI, return_std=True)

    fig1, axs = plt.subplots(
        nrows=3,
        sharex=True,
        gridspec_kw=dict(height_ratios=[1, 0.3, 0.3]),
        figsize=(15, 10),
    )
    axs[0].plot(x, y, "o", color="tab:gray", alpha=0.5)
    axs[0].plot(xi, yi_krr, label="krr", color="tab:red")
    axs[0].plot(xi, yi_gp, label="gp", color="tab:green")
    axs[0].fill_between(
        xi,
        yi_gp - 2 * yi_gp_std,
        yi_gp + 2 * yi_gp_std,
        alpha=0.2,
        color="tab:gray",
        label=r"gp $\pm 2\,\sigma$",
    )

    yspan = y.max() - y.min()
    axs[0].plot(
        xi, yi_gt, label="ground truth g(x)", color="tab:gray", alpha=0.5
    )
    axs[0].set_ylim(y.min() - 0.1 * yspan, y.max() + 0.1 * yspan)

    diff_krr = yi_gt - yi_krr
    diff_gp = yi_gt - yi_gp
    msk = (xi > x.min()) & (xi < x.max())
    lo = min(diff_krr[msk].min(), diff_gp[msk].min())
    hi = max(diff_krr[msk].max(), diff_gp[msk].max())
    span = hi - lo
    axs[1].plot(xi, diff_krr, label="krr - g(x)", color="tab:red")
    axs[1].plot(xi, diff_gp, label="gp - g(x)", color="tab:green")
    axs[1].set_ylim((lo - 0.1 * span, hi + 0.1 * span))

    axs[2].plot(xi, yi_gp_std, label=r"gp $\sigma$")

    for ax in axs:
        ax.legend()

    # Could re-use HyperOpt instances from above, but re-create here for
    # clarity.
    ##ic("Plot hyperopt objective functions")
    nsample = 50
    nlevels = 50
    z_log = False
    param_p = np.logspace(*np.log10(param_p_bounds), nsample)
    param_r = np.logspace(*np.log10(param_r_bounds), nsample)
    grid = np.array(list(itertools.product(param_p, param_r)))
    fig2, axs2d = plt.subplots(nrows=1, ncols=2, figsize=(18, 8))
    fig3, axs3d = plt.subplots(
        nrows=1, ncols=2, figsize=(18, 8), subplot_kw={"projection": "3d"}
    )

    # zmax for linear z scale (see below)
    cases = dict(gp=dict(zmax=3e-10), krr=dict(zmax=1e-5))

    for icol, name in enumerate(cases):
        ax2d = axs2d[icol]
        ax3d = axs3d[icol]
        ax2d.set_title(name)
        ax3d.set_title(name)
        if name == "krr":
            ho = HyperOptKRR(
                bounds=bounds,
                get_model=get_model_krr,
                seed=seed,
            )

            def func(params):
                return ho.obj_func(params, X, y)

            params_opt = params_krr
        elif name == "gp":

            # Use a fitted GP object from above that has RBF+WhiteKernel such
            # that we can call log_marginal_likelihood() with a length 2 param
            # array.
            #
            def func(params):
                return -f_gp_1.log_marginal_likelihood(np.log(params))

            # Use HyperOpt API defined above.
            #
            ##ho = HyperOptGP(
            ##    get_model=get_model_gp_2,
            ##    bounds=bounds,
            ##)

            ##def func(params):
            ##    return ho.obj_func(params, X, y)

            params_opt = params_gp_1

        with mp.Pool(mp.cpu_count()) as pool:
            zz = np.array(pool.map(func, grid))

        # z log scale looks nice but is hard to interpret, also it heavily
        # depends on eps of course. When not using a z log scale we need to cut
        # off at z >= zmax to visualize low z value regions where the global
        # mins live. Note that zmax depends on bounds and thus on the range of
        # z values. Compared to krr, the GP's LML is essentially flat around
        # the min. Still DE and the local optimizer find the same min.
        if z_log:
            eps = 0.01
            zz -= zz.min() - eps
            zz /= zz.max()
        else:
            zmax = cases[name]["zmax"]
            zz -= zz.min()
            zz /= zz.max()
            zz = np.ma.masked_where(zz > zmax, zz)

        _X, _Y = np.meshgrid(param_p, param_r, indexing="ij")
        Z = zz.reshape((_X.shape[0], _X.shape[1]))

        if z_log:
            levels = np.logspace(
                np.log10(zz.min()), np.log10(zz.max()), nlevels
            )
            pl2d = ax2d.contourf(
                _X, _Y, Z, levels=levels, norm=colors.LogNorm()
            )
            pl3d = ax3d.plot_surface(
                np.log10(_X), np.log10(_Y), np.log10(Z), cmap=cm.viridis
            )
        else:
            pl2d = ax2d.contourf(_X, _Y, Z, levels=nlevels)
            pl3d = ax3d.plot_surface(
                np.log10(_X),
                np.log10(_Y),
                Z,
                cmap=cm.viridis,
            )

        fig2.colorbar(pl2d, ax=ax2d)
        fig3.colorbar(pl3d, ax=ax3d)
        ax2d.plot(*params_opt, "o", ms=10, color="white")
        if name == "gp":
            ax2d.plot(*params_gp_0, "*", ms=10, color="black")
        ax2d.set_xlabel(r"$p$")
        ax2d.set_ylabel(r"$r$")
        ax3d.set_xlabel(r"$\log_{10}(p)$")
        ax3d.set_ylabel(r"$\log_{10}(r)$")
        ax2d.set_xscale("log")
        ax2d.set_yscale("log")

    plt.show()
