# Compare GP prediction APIs

* Rasmussen & Williams (R&W 2006) textbook equations
* [tinygp](https://github.com/dfm/tinygp) 0.2.2 (also wins price for best API!)
* sklearn 1.1.dev0
* GPy 1.10.0
* gpytorch 1.6.0

Use Gaussian RBF (squared exponential) kernel. Show the difference between
"noisy" and "noiseless" predictions.

To ensure accurate comparisons, we

* skip param optimization and instead fix `length_scale` (kernel) and
  `noise_level` (likelihood) b/c
  * testing correct prediction code paths is orthogonal to how kernel params
    and the likelihood `noise_level` are obtained
  * codes use different optimizers and/or convergence thresholds and/or start
    values
  * optimized `length_scale` and `noise_level` may carry numerical noise
  * we'd need to check whether `length_scale` and `noise_level` are from the
    same optimum of the log marginal likelihood
* skip code-internal data normalization
* set regularization defaults to zero where needed to ensure that we only add
  `noise_level` to the kernel matrix diag

At the very end, we do a `length_scale` optimization using `sklearn` and two
noise cases with fixed `noise_level`: interpolation (`noise_level` = 0) and
regression (`noise_level` > 0) and for each predict vs. predict_noiseless,
which results in a plot like this.

![](pics/gp.png)

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
(gp_pred_comp) $ pip install tinygp Gpy sklearn gpytorch matplotlib
```

# Resources

* [Rasmussen & Williams (R&W 2006) textbook](http://www.gaussianprocess.org/gpml)
* <https://distill.pub/2019/visual-exploration-gaussian-processes> and refs
  linked from there, in particular:
* <https://infallible-thompson-49de36.netlify.app>
