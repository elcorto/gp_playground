(s:pred_noise)=
# The role of noise in GPs

We show the difference between "noisy" and "noiseless" predictions w.r.t. the
posterior predictive covariance matrix. When learning a noise model
($\sigma_n^2>0$, e.g. using a `WhiteKernel` component in `sklearn`), then there
are two flavors of that covariance matrix. Borrowing from the `GPy` library's
naming scheme, we have

* `predict_noiseless`: $\cov(\ve f_*)$
* `predict`: $\cov(\ve f_*) + \sigma_n^2\,\ma I$

where $\cov(\ve f_*)$ is the posterior predictive covariance matrix
({cite}`rasmussen_2006_GaussianProcessesMachine` eq. 2.24). These lead to
different uncertainty estimates and posterior samples, as will be shown.

**Contents**

```{tableofcontents}
```
