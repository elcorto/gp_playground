# Compare GP prediction APIs

* Rasmussen & Williams (R&W 2006) textbook equations
* [tinygp](https://github.com/dfm/tinygp) 0.2.2 (also wins price for best API!)
* sklearn 1.1.0
* GPy 1.10.0
* gpytorch 1.6.0

We use the Gaussian RBF (squared exponential) covariance function.

We show the difference between "noisy" and "noiseless" predictions w.r.t. the
covariance matrix and how to obtain both with the different libraries listed
above. In short, when learning a noise model (the likelihood variance
(`noise_level`) is nonzero, e.g. using a `WhiteKernel` component in `sklearn`),
then there are two flavors of the posterior predictive's covariance matrix.
Borrowing from the `GPy` library's naming scheme, we have

* `predict`: `cov = cov(f*) + noise_level * I`
* `predict_noiseless`: `cov = cov(f*)`

where `cov(f*)` is the posterior predictive covariance matrix (R&W 2006, eq.
2.24) and `I` is the identity matrix. When doing interpolation
(`noise_level`=0) then both are equal.

To ensure accurate comparisons, we

* skip param optimization and instead fix `length_scale` (kernel) and
  `noise_level` (likelihood) because
  * testing correct prediction code paths is orthogonal to how kernel params
    and the likelihood `noise_level` are obtained
  * codes use different optimizers and/or convergence thresholds and/or start
    values, so optimized params might not be
    * from the same (local) optimum of the log marginal likelihood
    * equal enough numerically
* skip code-internal data normalization
* set regularization defaults to zero where needed to ensure that we only add
  `noise_level` to the kernel matrix diag

At the very end, we do a `length_scale` optimization using `sklearn` and two
noise cases with fixed `noise_level`: interpolation (`noise_level` = 0) and
regression (`noise_level` > 0) and for each `predict` vs. `predict_noiseless`,
which results in a plot like this.

![](pics/gp.png)

The difference in `y_std` between `predict` vs. `predict_noiseless` is not
constant even though the constant `noise_level` is added to the diagonal
because of the `sqrt()` in

`y_std = sqrt(diag(cov(f*) + noise_level * I))` .

# Install packages

```sh
# https://virtualenvwrapper.readthedocs.io/en/latest/
$ mkvirtualenv gp_pred_comp
```

else

```
$ python3 -m venv gp_pred_comp && . ./gp_pred_comp/bin/activate
```

Then

```
(gp_pred_comp) $ pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
(gp_pred_comp) $ pip install tinygp Gpy scikit-learn gpytorch matplotlib
```

# Resources

* [Rasmussen & Williams (R&W 2006) textbook](http://www.gaussianprocess.org/gpml)
* <https://distill.pub/2019/visual-exploration-gaussian-processes> and refs
  linked from there, in particular:
* <https://infallible-thompson-49de36.netlify.app>
* This repo was inspired by a [discussion over at the sklearn issue
  tracker](https://github.com/scikit-learn/scikit-learn/issues/22945). Thanks!
