(s:gp_pred_noise)=
# The role of noise in GPs

We show the difference between "noisy" and "noiseless" predictions w.r.t. the
posterior predictive covariance matrix. When learning a noise model
($\sigma_n^2>0$, e.g. using a `WhiteKernel` component in `sklearn`), then there
are two flavors of that covariance matrix. Borrowing from the `GPy` library's
naming scheme, we have

* `predict_noiseless`: $\cov(\predve f_*)$
* `predict`: $\cov(\predve f_*) + \sigma_n^2\,\ma I$

where $\cov(\predve f_*)$ is the posterior predictive covariance matrix
({cite}`rasmussen_2006_GaussianProcessesMachine` eq. 2.24). These lead to
different uncertainty estimates and posterior samples, as will be shown.


## GP intro

Unless stated otherwise, we use the Gaussian radial basis function (a.k.a.
squared exponential) as covariance ("kernel") function

$$\kappa(\ve x_i, \ve x_j) = \exp\left(-\frac{\lVert\ve x_i - \ve x_j\rVert_2^2}{2\,\ell^2}\right)$$


Notation:

* RBF kernel length scale parameter: $\ell$ = `length_scale` (as in sklearn)
* likelihood variance $\sigma_n^2$ (a.k.a. "noise level") in GPs,
  regularization parameter $\lambda$ in KRR: $\eta$ = `noise_level` (as in `sklearn`)
* posterior predictive variance: $\sigma^2$



## Contents

```{tableofcontents}
```
