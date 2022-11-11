#!/usr/bin/env python3

from functools import partial

from box import Box

import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import matplotlib

from common import (
    textbook_prior,
    textbook_posterior,
    textbook_posterior_noise,
    cov2std,
    sample_y,
)

# =========================================================================
# sklearn plotting, 1D data, WhiteKernel w/ fixed noise_level; cases:
# * prior (with noise_level=0)
# then posterior: optimize length_scale, noise_level fixed
# * interpolation (noise_level=0)
# * regression (noise_level > 0) predict_noiseless
# * regression (noise_level > 0) predict
# =========================================================================


plt.rcParams["figure.autolayout"] = True
plt.rcParams["font.size"] = 18


def transform_labels(name):
    lmap = dict(
        noise_level=r"$\sigma_n^2$",
        length_scale=r"$\ell$",
        y_std_pn=r"$\sqrt{\mathrm{diag}(\Sigma)}$",
        y_std_p=r"$\sqrt{\mathrm{diag}(\Sigma + \sigma_n^2\,I)}$",
        y_mean=r"$\mu$",
    )
    return lmap.get(name, name)


def gt_func(x):
    """Ground truth"""
    return np.sin(x) * np.exp(-0.1 * x) + 10


def transform_1d(scaler, x):
    assert x.ndim == 1
    return scaler.transform(x.reshape(-1, 1))[:, 0]


def make_axs_itr(axs):
    if isinstance(axs, matplotlib.axes.Axes):
        return [axs]
    else:
        iter(axs)
        return axs


def calc_gp(*, pri_post, noise_level, pred_mode, XI, X=None, y=None):
    if pri_post == "pri":
        length_scale = 0.5
        gp = GaussianProcessRegressor(
            kernel=RBF(length_scale=length_scale)
            + WhiteKernel(noise_level=noise_level),
            alpha=0,
            normalize_y=False,
        )
    elif pri_post == "post":
        gp = GaussianProcessRegressor(
            kernel=RBF(length_scale_bounds=[1e-5, 10])
            + WhiteKernel(noise_level=noise_level, noise_level_bounds="fixed"),
            n_restarts_optimizer=5,
            alpha=0,
            normalize_y=False,
        )

        gp.fit(X, y)
        length_scale = gp.kernel_.k1.length_scale
        # gp.kernel_.k2.noise_level == noise_level (fixed)
    else:
        raise ValueError(f"unknown {pri_post=}")

    y_mean, y_cov = gp.predict(XI, return_cov=True)

    if pred_mode == "p":
        post_ref_func = textbook_posterior_noise
    elif pred_mode == "pn":
        post_ref_func = textbook_posterior
        y_cov -= np.eye(XI.shape[0]) * noise_level
    else:
        raise ValueError(f"unknown {pred_mode=}")

    y_std_label = transform_labels(f"y_std_{pred_mode}")
    y_std = cov2std(y_cov)

    if pri_post == "pri":
        y_mean_ref, y_std_ref, y_cov_ref = textbook_prior(
            noise_level=noise_level, length_scale=length_scale
        )(XI)
    else:
        y_mean_ref, y_std_ref, y_cov_ref = post_ref_func(
            X, y, noise_level=noise_level, length_scale=length_scale
        )(XI)

    np.testing.assert_allclose(y_mean, y_mean_ref, rtol=0, atol=1e-9)
    np.testing.assert_allclose(y_std, y_std_ref, rtol=0, atol=1e-9)
    np.testing.assert_allclose(y_cov, y_cov_ref, rtol=0, atol=1e-9)

    samples = sample_y(y_mean, y_cov, 10, random_state=123)

    if noise_level == 0:
        cov_title = r"$\Sigma=K'' - K'\,K^{-1}\,K'^\top$"
    else:
        cov_title = r"$\Sigma=K'' - K'\,(K+\sigma_n^2\,I)^{-1}\,K'^\top$"

    cov_title += "\n" + rf"$\sigma$={y_std_label}"

    return Box(
        y_mean=y_mean,
        y_cov=y_cov,
        y_std=y_std,
        samples=samples,
        cov_title=cov_title,
        length_scale=length_scale,
        noise_level=noise_level,
        y_std_label=y_std_label,
        pri_post=pri_post,
    )


def plot_gp(*, box: Box, ax, xi, std_color="tab:orange", x=None, y=None):
    ax.set_title(
        f"{transform_labels('noise_level')}={box.noise_level}   "
        f"{transform_labels('length_scale')}={box.length_scale:.5f}"
        "\n" + box.cov_title
    )

    samples_kind = "prior" if box.pri_post == "pri" else "posterior"
    for ii, yy in enumerate(box.samples.T):
        ax.plot(
            xi,
            yy,
            color="tab:gray",
            alpha=0.3,
            label=(f"{samples_kind} samples" if ii == 0 else "_"),
        )

    ax.plot(
        xi,
        box.y_mean,
        lw=3,
        color="tab:red",
        label=transform_labels("y_mean"),
    )

    if box.pri_post == "post":
        ax.plot(x, y, "o", ms=10)

    ax.fill_between(
        xi,
        box.y_mean - 2 * box.y_std,
        box.y_mean + 2 * box.y_std,
        alpha=0.1,
        color=std_color,
        label=rf"$\pm$ 2 {box.y_std_label}",
    )


if __name__ == "__main__":

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

    # -----------------------------------------------------------------------
    # prior
    # -----------------------------------------------------------------------
    d_pri_0 = calc_gp(
        pri_post="pri",
        noise_level=0,
        pred_mode="pn",
        XI=XI,
    )

    fig, ax = plt.subplots(figsize=(10.1, 9.7))
    plot_gp(box=d_pri_0, ax=ax, xi=xi)

    # -----------------------------------------------------------------------
    # posterior
    # -----------------------------------------------------------------------
    noise_level = 0.3
    calc_gp_post = partial(calc_gp, X=X, y=y, XI=XI)
    d_post_0 = calc_gp_post(
        pri_post="post",
        noise_level=0,
        pred_mode="pn",
    )

    d_post_p = calc_gp_post(
        pri_post="post",
        noise_level=noise_level,
        pred_mode="p",
    )

    d_post_pn = calc_gp_post(
        pri_post="post",
        noise_level=noise_level,
        pred_mode="pn",
    )

    fig, axs = plt.subplots(
        nrows=3,
        ncols=3,
        gridspec_kw=dict(height_ratios=[1, 0.3, 0.3]),
        figsize=(30, 15),
        sharex=True,
    )

    y_std_p_color = "tab:cyan"
    y_std_pn_color = "tab:orange"
    plot_gp_post = partial(plot_gp, x=x, y=y, xi=xi)
    plot_gp_post(box=d_post_0, ax=axs[0, 0])
    plot_gp_post(box=d_post_pn, ax=axs[0, 1])
    plot_gp_post(box=d_post_p, ax=axs[0, 2], std_color=y_std_p_color)

    dd = d_post_pn
    axs[0, 2].fill_between(
        xi,
        dd.y_mean - 2 * dd.y_std,
        dd.y_mean + 2 * dd.y_std,
        alpha=0.1,
        color=y_std_pn_color,
        label=rf"$\pm$ 2 {dd.y_std_label}",
    )

    for dd, color in [(d_post_p, y_std_p_color), (d_post_pn, y_std_pn_color)]:
        axs[1, 2].plot(xi, dd.y_std, label=dd.y_std_label, color=color)

    aa = d_post_p
    bb = d_post_pn
    axs[2, 2].plot(
        xi, aa.y_std - bb.y_std, label=f"{aa.y_std_label} - {bb.y_std_label}"
    )

    for ax in axs[1:, :2].flat:
        ax.set_visible(False)

    for ax in [axs[0, 1], axs[0, 2], axs[1, 2], axs[2,2]]:
        ax.legend(loc="upper right")

    for ax in axs[:, 0]:
        ax.set_ylim(-3, 3)

    for ax in axs[1, :].flat:
        ax.set_ylim(-0.1, 1.3)

    for ax in axs[2, :].flat:
        ax.set_ylim(-0.1, 0.5)

    plt.show()
