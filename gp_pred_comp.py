#!/usr/bin/env python3

from functools import partial

from scipy.spatial.distance import cdist
from scipy import linalg
import numpy as np


def rbf_gauss(rsq, length_scale):
    return np.exp(-0.5 * rsq / length_scale**2.0)


def kernel(X1, X2, **kwds):
    return rbf_gauss(cdist(X1, X2, metric="sqeuclidean"), **kwds)


def sym_solve(A, b):
    return linalg.solve(A, b, assume_a="sym")


def cov2std(y_cov):
    return np.sqrt(np.diag(y_cov))


def textbook_prior(noise_level=0, **kernel_kwds):
    """
    R&W 2006, eq. 2.17 (if noise_level=0)
        X*        = X_pred
        K(X*, X*) = K_pp
    """
    kern = partial(kernel, **kernel_kwds)

    def predict(X_pred):
        K_pp = kern(X_pred, X_pred)
        K_pp_noise = K_pp + np.eye(X_pred.shape[0]) * noise_level
        y_mean = np.zeros((X_pred.shape[0],))
        y_cov = K_pp_noise
        y_std = cov2std(y_cov)
        return y_mean, y_std, y_cov

    return predict


def textbook_posterior(X_train, y_train, *, noise_level=0, **kernel_kwds):
    """
    R&W 2006, eqs. 2.23, 2.24
        X         = X_train
        X*        = X_pred
        K(X, X)   = K_tt
        K(X*, X)  = K_pt
        K(X*, X*) = K_pp
    """

    kern = partial(kernel, **kernel_kwds)
    K_tt = kern(X_train, X_train)
    K_tt_noise = K_tt + np.eye(X_train.shape[0]) * noise_level
    alpha = sym_solve(K_tt_noise, y_train)

    def predict(X_pred):
        K_pt = kern(X_pred, X_train)
        K_pp = kern(X_pred, X_pred)
        y_mean = K_pt @ alpha
        ##y_cov = K_pp - K_pt @ linalg.inv(K_tt_noise) @ K_pt.T
        y_cov = K_pp - K_pt @ sym_solve(K_tt_noise, K_pt.T)
        y_std = cov2std(y_cov)
        return y_mean, y_std, y_cov

    return predict


def textbook_posterior_noise(*args, **kwds):
    """
    Same as textbook_posterior, only that

        y_cov += np.eye(X_pred.shape[0]) * noise_level
    """

    def predict(X_pred):
        y_mean, _, cov = textbook_posterior(*args, **kwds)(X_pred)
        y_cov = cov + np.eye(X_pred.shape[0]) * kwds["noise_level"]
        y_std = cov2std(y_cov)
        return y_mean, y_std, y_cov

    return predict


if __name__ == "__main__":

    rng = np.random.default_rng(123)

    def rand(*size):
        return rng.uniform(size=size)

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

    # =========================================================================
    # gpytorch
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
        np.testing.assert_allclose(
            y_std, np.sqrt(gp.variance.detach().numpy())
        )
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
    model = ExactGPModel(
        T.from_numpy(X_train), T.from_numpy(y_train), likelihood
    )

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

    # =========================================================================
    # sklearn plotting, 1D data, WhiteKernel w/ fixed noise_level, optimize
    # length_scale now, show two cases: interpolation (noise_level=0) and
    # regression (noise_level > 0) and in each predict vs. predict_noiseless
    # =========================================================================

    import matplotlib.pyplot as plt

    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["font.size"] = 18

    from sklearn.preprocessing import StandardScaler

    def transform_labels(name):
        lmap = dict(
            noise_level=r"$\sigma_n^2$",
            length_scale=r"$\ell$",
            y_std=r"$\sigma$",
            y_mean=r"$\mu$",
        )
        return lmap.get(name, name)

    def gt_func(x):
        """Ground truth"""
        return np.sin(x) * np.exp(-0.1 * x) + 10

    def transform_1d(scaler, x):
        assert x.ndim == 1
        return scaler.transform(x.reshape(-1, 1))[:, 0]

    seed = 123
    rng = np.random.default_rng(seed=seed)

    x = np.sort(rng.uniform(0, 5, 5), axis=0)
    xspan = x.max() - x.min()
    xi = np.linspace(x.min() - 0.3 * xspan, x.max() + 0.3 * xspan, len(x) * 50)
    y = gt_func(x)

    in_scaler = StandardScaler().fit(x.reshape(-1, 1))
    out_scaler = StandardScaler().fit(y.reshape(-1, 1))
    x = transform_1d(in_scaler, x)
    xi = transform_1d(in_scaler, xi)
    y = transform_1d(out_scaler, y)
    X = x[:, None]
    XI = xi[:, None]

    fig, axs = plt.subplots(
        nrows=3,
        ncols=2,
        gridspec_kw=dict(height_ratios=[1, 0.3, 0.3]),
        figsize=(15, 10),
        sharex=True,
    )

    for icol, noise_level in enumerate([0, 0.1]):
        gp = GaussianProcessRegressor(
            kernel=RBF(length_scale_bounds=[1e-5, 10])
            ##+ WhiteKernel(noise_level_bounds=[1e-18, 2]),
            + WhiteKernel(noise_level=noise_level, noise_level_bounds="fixed"),
            n_restarts_optimizer=5,
            alpha=0,
            normalize_y=False,
        )

        gp.fit(X, y)
        length_scale = gp.kernel_.k1.length_scale

        axs[0, icol].set_title(
            f"{transform_labels('noise_level')}={noise_level}   "
            f"{transform_labels('length_scale')}={length_scale:.5f}"
        )
        ##axs[0, icol].plot(xi, gp.sample_y(XI, 10), color="tab:gray", alpha=0.3)
        # hack for getting lablels right
        samples = gp.sample_y(XI, 10)
        for ii, yy in enumerate(samples.T):
            axs[0, icol].plot(
                xi,
                yy,
                color="tab:gray",
                alpha=0.3,
                label=("posterior samples" if ii == 0 else "_"),
            )

        # posterior, predict()
        y_mean, y_cov = gp.predict(XI, return_cov=True)
        y_std_p = cov2std(y_cov)
        y_mean_ref, y_std_ref, y_cov_ref = textbook_posterior_noise(
            X, y, noise_level=noise_level, length_scale=length_scale
        )(XI)
        np.testing.assert_allclose(y_mean, y_mean_ref, rtol=0, atol=1e-9)
        np.testing.assert_allclose(y_std_p, y_std_ref, rtol=0, atol=1e-9)
        np.testing.assert_allclose(y_cov, y_cov_ref, rtol=0, atol=1e-9)

        y_std_label = transform_labels("y_std")
        axs[0, icol].plot(
            xi, y_mean, lw=3, color="tab:red", label=transform_labels("y_mean")
        )
        axs[0, icol].fill_between(
            xi,
            y_mean - 2 * y_std_p,
            y_mean + 2 * y_std_p,
            alpha=0.1,
            color="tab:cyan",
            label=rf"$\pm$ 2 {y_std_label} predict",
        )

        # posterior, predict_noiseless(), re-use y_cov from above;
        # gp.kernel_.k2.noise_level == noise_level (fixed)
        y_cov -= np.eye(XI.shape[0]) * gp.kernel_.k2.noise_level
        y_std_pn = cov2std(y_cov)
        _, y_std_ref, y_cov_ref = textbook_posterior(
            X, y, noise_level=noise_level, length_scale=length_scale
        )(XI)
        np.testing.assert_allclose(y_std_pn, y_std_ref, rtol=0, atol=1e-9)
        np.testing.assert_allclose(y_cov, y_cov_ref, rtol=0, atol=1e-9)

        axs[0, icol].fill_between(
            xi,
            y_mean - 2 * y_std_pn,
            y_mean + 2 * y_std_pn,
            alpha=0.1,
            color="tab:orange",
            label=rf"$\pm$ 2 {y_std_label} predict_noiseless",
        )

        axs[0, icol].plot(x, y, "o", ms=10)

        axs[1, icol].plot(
            xi, y_std_p, color="tab:cyan", label=f"{y_std_label} predict"
        )
        axs[1, icol].plot(
            xi,
            y_std_pn,
            color="tab:orange",
            label=f"{y_std_label} predict_noiseless",
        )
        axs[1, icol].set_ylim(-0.1, 1.1)

        axs[2, icol].plot(
            xi,
            y_std_p - y_std_pn,
            label=f"{y_std_label} predict - predict_noiseless",
        )
        axs[2, icol].set_ylim(-0.1, 0.3)

    for ax in axs[:, 1]:
        ax.legend()

    ##fig.savefig("pics/gp.png")
    plt.show()
