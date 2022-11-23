**About**

This project explores selected aspects of Gaussian processes (GPs) by
implementing them (a.k.a. "learning by hacking").[^fn:excuse]

**Content**

```{tableofcontents}
```


**GP intro**

Unless stated otherwise, we use the Gaussian radial basis function (a.k.a.
squared exponential) as covariance ("kernel") function

$$\kappa(\ve x_i, \ve x_j) = \exp\left(-\frac{\lVert\ve x_i - \ve x_j\rVert_2^2}{2\,\ell^2}\right)$$


Notation:

* RBF kernel length scale parameter: $\ell$ = `length_scale` (as in sklearn)
* likelihood variance $\sigma_n^2$ (a.k.a. "noise level") in GPs,
  regularization parameter $\lambda$ in KRR: $\eta$ = `noise_level` (as in `sklearn`)
* posterior predictive variance: $\sigma^2$



**Resources**

* {cite}`rasmussen_2006_GaussianProcessesMachine, murphy_2023_ProbabilisticMachineLearningAdvanced, goertler_2019_VisualExplorationGaussiana, deisenroth_2020_PracticalGuideGaussian, kanagawa_2018_GaussianProcessesKernel`
* [Blog post about GP vs. KRR][gp_krr_blog]
* [This section](s:pred_noise) was inspired by a [discussion over at the sklearn issue
  tracker][sklearn_issue]. Thanks!

```{bibliography}
```

[gp_krr_blog]: https://gregorygundersen.com/blog/2020/01/06/kernel-gp-regression
[sklearn_issue]: https://github.com/scikit-learn/scikit-learn/issues/22945
[^fn:excuse]: And is an excuse to play with JupyterBook.

