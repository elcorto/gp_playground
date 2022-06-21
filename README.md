# Compare GP prediction APIs

We test calculating GP prior and posterior predictions using

* Rasmussen & Williams (R&W 2006) textbook equations
* [tinygp](https://github.com/dfm/tinygp) 0.2.2
* sklearn 1.1.0
* GPy 1.10.0
* gpytorch 1.6.0

using the Gaussian radial basis function (a.k.a. squared exponential)
as covariance ("kernel") function

$$k(\mathbf x_i, \mathbf x_j) = \exp\left(-\frac{\lVert\mathbf x_i - \mathbf x_j\rVert_2^2}{2 \ell^2}\right)$$

with random data inputs $\mathbf x_i \in \mathbb R^D$ and targets
$y_i \in \mathbb R$.

Notation:

* kernel length scale parameter: $\ell$ = `length_scale` (as in sklearn)
* likelihood variance: $\sigma_n^2$ = `noise_level` (as in sklearn)
* posterior predictive variance: $\sigma^2$

We show the difference between "noisy" and "noiseless" predictions w.r.t. the
covariance matrix and how to obtain both with the different libraries listed
above. The textbook equations serve as a reference. In short, when learning a
noise model ($\sigma_n^2>0$, e.g. using a `WhiteKernel` component in
`sklearn`), then there are two flavors of the posterior predictive's covariance
matrix. Borrowing from the `GPy` library's naming scheme, we have

* `predict_noiseless`: $\Sigma = \text{cov}(\mathbf f_*)$
* `predict`: $\Sigma = \text{cov}(\mathbf f_*) + \sigma_n^2\mathbf I$

where $\text{cov}(\mathbf f_*)$ is the posterior predictive covariance matrix
(R&W 2006, eq. 2.24). When doing interpolation ($\sigma_n=0$) then both
$\Sigma$ matrices are equal. $\ell$ and $\sigma_n^2$ are usually the result of
"fitting the GP model to data", which means optimizing the GP's log marginal
likelihood as a function of both (e.g. what `sklearn`'s
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

At the very end, we do a $\ell$ optimization using 1$D$ toy data and `sklearn`
for two noise cases with fixed $\sigma_n^2$: interpolation ($\sigma_n$ = 0) and
regression ($\sigma_n$ > 0) and for each `predict` vs. `predict_noiseless`,
which results in a plot like this.

![](pics/gp.png)

The difference in $\sigma$ between `predict` vs. `predict_noiseless`
is not constant even though the constant $\sigma_n^2$ is added to the diagonal
because of the $\sqrt{\cdot}$ in

$$\sigma = \sqrt{\text{diag}(\text{cov}(\mathbf f_*) + \sigma_n^2\mathbf I)}$$

# Install packages

```sh
# https://virtualenvwrapper.readthedocs.io/en/latest/
$ mkvirtualenv gp_pred_comp
```

else

```
$ python3 -m venv gp_pred_comp && . ./gp_pred_comp/bin/activate
```

Then install some variant of torch (CPU, GPU) plus the packages that we test
here. The torch install line is just an example, please check the pytorch
website for more.

```
(gp_pred_comp) $ pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
(gp_pred_comp) $ pip install -r requirements.txt
```

# Resources

* [Rasmussen & Williams (R&W 2006) textbook](http://www.gaussianprocess.org/gpml)
* <https://distill.pub/2019/visual-exploration-gaussian-processes> and refs
  linked from there, in particular:
* <https://infallible-thompson-49de36.netlify.app>
* This repo was inspired by a [discussion over at the sklearn issue
  tracker](https://github.com/scikit-learn/scikit-learn/issues/22945). Thanks!
